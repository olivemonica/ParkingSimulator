import os
import time
import functools
import numpy as np
import os.path as osp
import tensorflow as tf
from baselines import logger
from collections import deque
from baselines.common import explained_variance, set_global_seeds
from baselines.common.policies import build_policy
from baselines.common.runners import AbstractEnvRunner
from baselines.common.tf_util import get_session, save_variables, load_variables
from baselines.common.mpi_adam_optimizer import MpiAdamOptimizer

from mpi4py import MPI
from baselines.common.tf_util import initialize
from baselines.common.mpi_util import sync_from_root


class PPO(object):
    """
    We use this object to :
    __init__:
    - Creates the step_model
    - Creates the train_model

    train():
    - Make the training part (feedforward and retropropagation of gradients)

    save/load():
    - Save load the model
    """

    def __init__(self, policy, ob_space, ac_space, nbatch_act, nbatch_train,
                 nsteps, ent_coef, vf_coef, max_grad_norm, *args):
        sess = get_session()

        with tf.variable_scope('ppo2_model', reuse=tf.AUTO_REUSE):
            # CREATE OUR TWO MODELS
            # act_model that is used for sampling
            act_model = policy(nbatch_act, 1, sess)

            # Train model for training
            train_model = policy(nbatch_train, nsteps, sess)

        # CREATE THE PLACEHOLDERS
        A = train_model.pdtype.sample_placeholder([None])
        ADV = tf.placeholder(tf.float32, [None])
        R = tf.placeholder(tf.float32, [None])
        # Keep track of old actor
        OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
        # Keep track of old critic
        OLDVPRED = tf.placeholder(tf.float32, [None])
        LR = tf.placeholder(tf.float32, [])
        # Cliprange
        CLIPRANGE = tf.placeholder(tf.float32, [])

        neglogpac = train_model.pd.neglogp(A)

        # Calculate the entropy
        # Entropy is used to improve exploration by limiting the
        # premature convergence to suboptimal policy.
        entropy = tf.reduce_mean(train_model.pd.entropy())

        # CALCULATE THE LOSS
        '''
        Total loss = (Policy gradient loss -
                      entropy * entropy coefficient +
                      Value coefficient * value loss)
        '''

        # Clip the value to reduce variability during Critic training
        # Get the predicted value
        vpred = train_model.vf
        vpredclipped = OLDVPRED + tf.clip_by_value(train_model.vf - OLDVPRED,
                                                   - CLIPRANGE,
                                                   CLIPRANGE)
        # Unclipped value
        vf_losses1 = tf.square(vpred - R)
        # Clipped value
        vf_losses2 = tf.square(vpredclipped - R)

        vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))

        # Calculate ratio (pi current policy / pi old policy)
        ratio = tf.exp(OLDNEGLOGPAC - neglogpac)

        # Defining Loss = - J is equivalent to max J
        pg_losses = -ADV * ratio

        pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)

        # Final PG loss
        pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
        approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
        clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))

        # Total loss
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef

        # UPDATE THE PARAMETERS USING LOSS
        # 1. Get the model parameters
        params = tf.trainable_variables('ppo2_model')
        # 2. Build our trainer
        trainer = MpiAdamOptimizer(MPI.COMM_WORLD, learning_rate=LR, epsilon=1e-5)
        # 3. Calculate the gradients
        grads_and_var = trainer.compute_gradients(loss, params)
        grads, var = zip(*grads_and_var)

        if max_grad_norm is not None:
            # Clip the gradients (normalize)
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads_and_var = list(zip(grads, var))
        # zip aggregate each gradient with parameters associated
        # For instance zip(ABCD, xyza) => Ax, By, Cz, Da

        _train = trainer.apply_gradients(grads_and_var)

        def train(lr, cliprange, obs, returns, masks, actions, values, neglogpacs, states=None):
            # Here we calculate advantage A(s,a) = R + yV(s') - V(s)
            # Returns = R + yV(s')
            advs = returns - values

            # Normalize the advantages
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)
            td_map = {train_model.X: obs, A: actions, ADV: advs, R: returns, LR: lr,
                      CLIPRANGE: cliprange, OLDNEGLOGPAC: neglogpacs, OLDVPRED: values}
            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
            return sess.run(
                [pg_loss, vf_loss, entropy, approxkl, clipfrac, _train],
                td_map
            )[:-1]
        self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']

        self.train = train
        self.train_model = train_model
        self.act_model = act_model
        self.value = act_model.value
        self.initial_state = act_model.initial_state

        self.save = functools.partial(save_variables, sess=sess)
        self.load = functools.partial(load_variables, sess=sess)

        if MPI.COMM_WORLD.Get_rank() == 0:
            initialize()
        global_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="")
        sync_from_root(sess, global_variables)  # pylint: disable=E1101

    def step(self, obs, S=None, M=None):
        # Function defined just to apply tanh to actions
        actions, values, states, neglogpacs = self.act_model.step(obs, S=S, M=M)
        return np.tanh(actions), values, states, neglogpacs

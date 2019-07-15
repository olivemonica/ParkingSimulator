import argparse
from baselines.common.misc_util import boolean_flag
import baselines.common.tf_util as U
from baselines import logger
from baselines.common.policies import build_policy
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from .ppo import PPO
import tensorflow as tf
import pickle
from mpi4py import MPI
from Environment import ParkingSimulator
import os
from copy import copy
import warnings
import time


def test(load_ckpt, rand_spot, sim_mode, special_eps, state_type, num_eps, env, **kwargs):
    rank = MPI.COMM_WORLD.Get_rank()
    # Handling where to save model output

    if load_ckpt is not None:
        if not os.path.exists(load_ckpt):
            raise RuntimeError('"{}" is not a valid directory'.format(load_ckpt))
    else:
        raise RuntimeError('You must provide a directory containing the model file to load')

    if sim_mode != 'continuous':
        warnings.warn("The ppo algorithm assumes a continuous action space,"
                      "changed sim_mode to 'continuous'")
        sim_mode = 'continuous'

    # Create envs.
    '''
    timestep_as_channel = True if state_type == 'visual' else None
    env = ParkingSimulator.ParkingSimulator(rand_spot=rand_spot,
                                            sim_mode=sim_mode,
                                            special_eps=special_eps,
                                            state_type=state_type,
                                            gamma=kwargs.get('gamma', 0.99),
                                            timestep_as_channel=timestep_as_channel)
    '''

    def make_env():
        return copy(dummy_env)
    #env = DummyVecEnv([make_env])

    gamma = kwargs.get('gamma', 0.99)
    lam = kwargs.get('lambda', 1.)
    max_grad_norm = 0.5

    network_kwargs = {}
    if state_type == 'visual':
        network = 'cnn'
    else:
        network = 'mlp'
        network_kwargs['num_hidden'] = max(kwargs['actor_hidden_size'])
        network_kwargs['num_layers'] = len(kwargs['actor_hidden_size'])
        network_kwargs['layer_norm'] = kwargs['layer_norm']

    policy = build_policy(env, network, **network_kwargs)

    # Get state_space and action_space
    ob_space = env.observation_space
    ac_space = env.action_space

    # Instantiate the model object (that creates act_model and train_model)
    def make_model():
        return PPO(policy=policy, ob_space=ob_space, ac_space=ac_space,
                   nbatch_act=1, nbatch_train=0,
                   nsteps=0, ent_coef=0, vf_coef=0,
                   max_grad_norm=0)

    # import pdb; pdb.set_trace()
    model = make_model()
    model.load(load_ckpt)

    obs = env.reset()
    env.render()

    #for i in reversed(range(5)):
    #    print(i + 1)
    #    time.sleep(1)

    for e in range(num_eps):
        done = False
        ep_len = 0
        ep_rew = 0

        while not done:

            states = model.initial_state
            dones = [False]

            actions, _, states, _ = model.step(obs, S=states, M=dones)

            if env.in_spot:
                actions[0, 0] = -1

            obs, reward, done, infos = env.step(actions[0])

            env.render()

            ep_len += 1
            ep_rew += reward

            done = (done or ep_len >= 200)
        obs = env.reset()


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--algorithm', help='learning algorithm to be applied',
                        type=str, default='ddpg')
    parser.add_argument('--load_ckpt', help='model to be loaded initially',
                        type=str, default='latest_model')
    parser.add_argument('--sim_mode', help='continuous or discrete parking simulator',
                        type=str, default='continuous')
    boolean_flag(parser, 'special_eps', default=False)
    boolean_flag(parser, 'rand_spot', default=True)
    boolean_flag(parser, 'layer-norm', default=True)
    boolean_flag(parser, 'render', default=True)
    parser.add_argument('--actor_h1', help='Actor layer 1 hidden size', type=int, default=400)
    parser.add_argument('--actor_h2', help='Actor layer 2 hidden size', type=int, default=300)
    parser.add_argument('--critic_h1', help='Critic layer 1 hidden size', type=int, default=400)
    parser.add_argument('--critic_h2', help='Critic layer 2 hidden size', type=int, default=300)
    parser.add_argument('--nb_episodes', help='Number of episodes to view', type=int, default=20)
    parser.add_argument('--max_ep_len', help='Max length of each episode', type=int, default=150)
    args = parser.parse_args()
    dict_args = vars(args)

    dict_args['actor_hidden_size'] = [dict_args['actor_h1'], dict_args['actor_h2']]
    dict_args['critic_hidden_size'] = [dict_args['critic_h1'], dict_args['critic_h2']]
    del dict_args['actor_h1']
    del dict_args['actor_h2']
    del dict_args['critic_h1']
    del dict_args['critic_h2']

    return dict_args


if __name__ == '__main__':
    args = parse_args()
    if MPI.COMM_WORLD.Get_rank() == 0:
        logger.configure()
    # Run actual script.
    main(**args)

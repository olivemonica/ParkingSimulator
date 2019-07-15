import argparse
from baselines.common.misc_util import boolean_flag
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines import logger
from mpi4py import MPI
from Environment import ParkingSimulator
import os
import time
import pickle
import warnings
from copy import copy

ALGS = ['ddpg', 'ppo']


def main(algorithm, load_ckpt, rand_spot, sim_mode, special_eps, state_type,
         nenvs=1, name=None, **kwargs):

    # Handling where to save model output
    if name is not None:
        model_dir = os.path.join(os.getcwd(), 'experiments', algorithm, name)
    else:
        date_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
        model_dir = os.path.join(os.getcwd(), 'experiments', algorithm, date_time)
    
    if not os.path.exists(os.path.dirname(model_dir)):
        os.makedirs(os.path.dirname(model_dir))
    kwargs['model_dir'] = model_dir

    logger.configure(dir=model_dir)
    logger.info(locals())

    if load_ckpt is not None:
        ckpt_dir = os.path.join(os.getcwd(), 'experiments', algorithm, load_ckpt)
        if not os.path.exists(ckpt_dir):
            raise RuntimeError('"{}" is not a valid directory'.format(ckpt_dir))
        kwargs['ckpt_dir'] = ckpt_dir
    else:
        kwargs['ckpt_dir'] = None

    if state_type == 'visual':
        from xvfbwrapper import Xvfb
        vdisplay = Xvfb()
        vdisplay.start()

    # Based on algorithm requested, call a script
    if algorithm == 'ddpg':
        # check if user specified wrong simulator type
        if sim_mode != 'continuous':
            warnings.warn("The ddpg algorithm assumes a continuous action space,"
                          "changed sim_mode to 'continuous'")
            sim_mode = 'continuous'

        # Create envs.
        env = ParkingSimulator.ParkingSimulator(rand_spot=rand_spot,
                                                sim_mode=sim_mode,
                                                special_eps=special_eps,
                                                state_type=state_type,
                                                render_ep=kwargs.get('render', False),
                                                gamma=kwargs.get('gamma', 0.99))

        del kwargs['lambda']
        from models.ddpg import main
        main.run(env, **kwargs)

    elif algorithm == 'ppo':
        if sim_mode != 'continuous':
            warnings.warn("The ppo algorithm assumes a continuous action space,"
                          "changed sim_mode to 'continuous'")
            sim_mode = 'continuous'
        # Create envs.
        timestep_as_channel = True if state_type == 'visual' else None
        dummy_env = ParkingSimulator.ParkingSimulator(rand_spot=rand_spot,
                                                      sim_mode=sim_mode,
                                                      special_eps=special_eps,
                                                      state_type=state_type,
                                                      render_ep=kwargs.get('render', False),
                                                      gamma=kwargs.get('gamma', 0.99),
                                                      timestep_as_channel=timestep_as_channel)

        def make_env():
            return copy(dummy_env)
        env = DummyVecEnv([make_env for e in range(nenvs)])
        #env._render()

        '''
        ppo_kwargs = {}
        ppo_kwargs['env'] = env
        ppo_kwargs['total_timesteps'] = (kwargs['nb_epochs'] *
                                         kwargs['nb_epoch_cycles'] *
                                         kwargs['nb_train_steps'])
        if kwargs['nb_eval_steps'] > 0:
            ppo_kwargs['eval_env'] = copy(env)
        ppo_kwargs['seed'] = kwargs['seed']
        ppo_kwargs['nsteps'] = kwargs['nb_train_steps']
        ppo_kwargs['noptepochs'] = kwargs['nb_epoch_cycles']
        # nminibatches = 4
        ppo_kwargs['save_interval'] = kwargs['save_every']
        ppo_kwargs['load_path'] = kwargs.get('load_ckpt', None)
        ppo_kwargs['lr'] = kwargs['actor_lr']
        ppo_kwargs['gamma'] = kwargs.get('gamma', 0.99)

        ppo_kwargs['lam'] = kwargs['lambda']
        ppo_kwargs['max_grad_norm'] = 0.5

        ppo_kwargs['network'] = 'mlp'
        network_kwargs = {}
        network_kwargs['num_layers'] = len(kwargs['actor_hidden_size'])
        network_kwargs['num_hidden'] = max(kwargs['actor_hidden_size'])
        network_kwargs['layer_norm'] = kwargs['layer_norm']
        ppo_kwargs['**network_kwargs'] = network_kwargs

        from models.ppo import main
        main.learn(**ppo_kwargs)
        '''
        total_timesteps = (kwargs['nb_epochs'] *
                           kwargs['nb_epoch_cycles'] *
                           kwargs['nb_train_steps'])
        if kwargs.get('nb_eval_steps', 0) > 0:
            eval_env = copy(env)
        else:
            eval_env = None
        seed = kwargs['seed']
        nsteps = kwargs['nb_train_steps']
        noptepochs = kwargs['nb_epoch_cycles']
        # nminibatches = 4
        save_interval = kwargs['save_every']
        load_path = kwargs.get('load_ckpt', None)
        lr = kwargs['actor_lr']
        gamma = kwargs.get('gamma', 0.99)

        lam = kwargs['lambda']
        max_grad_norm = 0.5

        network_kwargs = {}
        if state_type == 'visual':
            network = 'cnn'
            # network_kwargs['rf'] = 3
        else:
            network = 'mlp'
            network_kwargs['num_hidden'] = max(kwargs['actor_hidden_size'])
            network_kwargs['num_layers'] = len(kwargs['actor_hidden_size'])
            network_kwargs['layer_norm'] = kwargs['layer_norm']

        from models.ppo import main
        model = main.learn(network=network, env=env, eval_env=eval_env,
                           total_timesteps=total_timesteps, nsteps=nsteps, noptepochs=noptepochs,
                           lr=lr, gamma=gamma, lam=lam, max_grad_norm=max_grad_norm,
                           save_interval=save_interval, load_path=load_path, model_dir=model_dir,
                           seed=seed, **network_kwargs)

        model.save(model_dir)
        pickle.dump(kwargs, open(os.join(model_dir, 'hyperparams.p'), 'wb'))

    else:
        raise RuntimeError('"{}" is not a supported algorithm'.format(algorithm))

    if vdisplay is not None:
        vdisplay.stop()


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--algorithm', help='learning algorithm to be applied',
                        type=str, default='ddpg', choices=ALGS)
    parser.add_argument('--load_ckpt', help='model to be loaded initially', type=str, default=None)
    parser.add_argument('--sim_mode', help='continuous or discrete parking simulator',
                        type=str, default='continuous')
    parser.add_argument('--state_type', help='visual or summary state',
                        type=str, default='summary', choices=['summary', 'visual'])
    parser.add_argument('--rand_spot', help='whether the spot should be randomly '
                                            'placed at each episode', type=bool, default=False)
    parser.add_argument('--nenvs', help='number of parallel envs', type=int, default=1)
    boolean_flag(parser, 'special_eps', default=False)
    boolean_flag(parser, 'render-eval', default=False)
    boolean_flag(parser, 'layer-norm', default=True)
    boolean_flag(parser, 'render', default=False)
    boolean_flag(parser, 'normalize-returns', default=False)
    boolean_flag(parser, 'normalize-observations', default=True)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--actor_h1', help='Actor layer 1 hidden size', type=int, default=100)
    parser.add_argument('--actor_h2', help='Actor layer 2 hidden size', type=int, default=100)
    parser.add_argument('--critic_h1', help='Critic layer 1 hidden size', type=int, default=100)
    parser.add_argument('--critic_h2', help='Critic layer 2 hidden size', type=int, default=100)
    parser.add_argument('--critic-l2-reg', type=float, default=1e-2)
    parser.add_argument('--batch-size', type=int, default=128)  # per MPI worker
    parser.add_argument('--actor-lr', type=float, default=1e-5)
    parser.add_argument('--critic-lr', type=float, default=1e-4)
    boolean_flag(parser, 'popart', default=False)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lambda', type=float, default=0.95)
    parser.add_argument('--reward-scale', type=float, default=1.)
    parser.add_argument('--clip-norm', type=float, default=None)
    parser.add_argument('--save-every', type=int, default=5)  # how often to save (epochs)
    parser.add_argument('--nb-epochs',
                        type=int, default=500)  # with default settings, perform 1M steps total
    parser.add_argument('--nb-epoch-cycles', type=int, default=20)
    parser.add_argument('--nb-train-steps',
                        type=int, default=200)  # per epoch cycle and MPI worker
    parser.add_argument('--nb-eval-steps', type=int, default=100)  # per epoch cycle and MPI worker
    parser.add_argument('--nb-rollout-steps',
                        type=int, default=100)  # per epoch cycle and MPI worker
    # choices are adaptive-param_xx, ou_xx, normal_xx, none
    parser.add_argument('--noise-type', type=str, default='adaptive-param_0.2')
    parser.add_argument('--name', type=str, default=None)
    boolean_flag(parser, 'evaluation', default=False)
    args = parser.parse_args()
    dict_args = vars(args)

    dict_args['actor_hidden_size'] = [dict_args['actor_h1'], dict_args['actor_h2']]
    dict_args['critic_hidden_size'] = [dict_args['critic_h1'], dict_args['critic_h2']]
    del dict_args['actor_h1']
    del dict_args['actor_h2']
    del dict_args['critic_h1']
    del dict_args['critic_h2']

    # Don't allow for evaluation
    del dict_args['evaluation']

    return dict_args


if __name__ == '__main__':
    args = parse_args()
    if MPI.COMM_WORLD.Get_rank() == 0:
        logger.configure()
    # Run actual script.
    main(**args)

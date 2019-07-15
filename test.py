import argparse
from baselines.common.misc_util import boolean_flag
import baselines.common.tf_util as U
from baselines import logger
import tensorflow as tf
import pickle
from mpi4py import MPI
from Environment import ParkingSimulator
import os
import warnings


def main(algorithm, load_ckpt, rand_spot, sim_mode, special_eps, state_type, nb_episodes, **kwargs):
    rank = MPI.COMM_WORLD.Get_rank()

    if load_ckpt is not None:
        ckpt_dir = os.path.join(os.getcwd(), 'experiments', algorithm, load_ckpt)
        if not os.path.exists(ckpt_dir):
            raise RuntimeError('"{}" is not a valid directory'.format(ckpt_dir))
        kwargs['ckpt_dir'] = ckpt_dir
    else:
        raise RuntimeError('You must provide a directory containing the model file to load')

    

    # Based on algorithm requested, call a script
    if algorithm == 'ddpg':
        all_rewards = pickle.load(open(os.path.join(ckpt_dir, 'rewards.p'), 'rb'))

        # check if user specified wrong simulator type
        if sim_mode != 'continuous':
            warnings.warn("The ddpg algorithm assumes a continuous action space,"
                          "changed sim_mode to 'continuous'")
        # Create envs.
        env = ParkingSimulator.ParkingSimulator(rand_spot=rand_spot,
                                                sim_mode=sim_mode,
                                                special_eps=special_eps,
                                                gamma=1)

        from models.ddpg.ddpg import DDPG
        from models.ddpg.models import Actor, Critic
        actor_hidden_size = kwargs['actor_hidden_size']
        critic_hidden_size = kwargs['critic_hidden_size']
        actor = Actor(env.action_space.shape[-1],
                      h1=actor_hidden_size[0], h2=actor_hidden_size[1],
                      layer_norm=kwargs['layer_norm'])
        critic = Critic(h1=critic_hidden_size[0], h2=critic_hidden_size[1],
                        layer_norm=kwargs['layer_norm'])
        agent = DDPG(actor, critic, None, env.observation_space.shape, env.action_space.shape,
                     gamma=1, tau=0, normalize_returns=False,
                     normalize_observations=True, batch_size=1,
                     action_noise=None, param_noise=None, critic_l2_reg=0,
                     actor_lr=1e-4, critic_lr=1e-3, enable_popart=False, clip_norm=False,
                     reward_scale=1)

        with U.single_threaded_session() as sess:
            # Prepare everything.
            if rank == 0:
                saver = tf.train.Saver()
            agent.initialize(sess, ckpt_dir, saver)
            sess.graph.finalize()

            agent.reset()

            ep_info = []
            for ep in range(kwargs['nb_episodes']):
                # obs = env._reset(len(all_rewards))
                obs = env._reset(1000)
                done = False
                ep_len = 0
                ep_reward = 0

                while not done and ep_len < kwargs['max_ep_len']:
                    action, _ = agent.pi(obs, apply_noise=False, compute_Q=False)
                    new_obs, r, done, substep_time = env._step(action)
                    env._render()

                    obs = new_obs
                    # Update tracking
                    ep_reward += r
                    ep_len += 1

                ep_info.append({'reward': ep_reward,
                                'length': ep_len})

            pickle.dump(ep_info, open(os.path.join(ckpt_dir, 'test_rewards.p'), 'wb'))
    elif algorithm == 'ppo':
        env = ParkingSimulator.ParkingSimulator(rand_spot=rand_spot,
                                                sim_mode=sim_mode,
                                                special_eps=special_eps,
                                                state_type=state_type,
                                                gamma=kwargs.get('gamma', 0.99),
                                                timestep_as_channel=False,
                                                reset_ep=40000)

        from models.ppo import test
        test.test(ckpt_dir, rand_spot, sim_mode, special_eps, state_type,
                  num_eps=nb_episodes, env=env, **kwargs)
    else:
        raise RuntimeError('"{}" is not a supported algorithm'.format(algorithm))


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--algorithm', help='learning algorithm to be applied',
                        type=str, default='ddpg')
    parser.add_argument('--load_ckpt', help='model to be loaded initially',
                        type=str, default='latest_model')
    parser.add_argument('--sim_mode', help='continuous or discrete parking simulator',
                        type=str, default='continuous')
    parser.add_argument('--state_type', help='visual or summary state',
                        type=str, default='summary', choices=['summary', 'visual'])
    boolean_flag(parser, 'special_eps', default=False)
    boolean_flag(parser, 'rand_spot', default=False)
    boolean_flag(parser, 'layer-norm', default=True)
    boolean_flag(parser, 'render', default=True)
    parser.add_argument('--actor_h1', help='Actor layer 1 hidden size', type=int, default=100)
    parser.add_argument('--actor_h2', help='Actor layer 2 hidden size', type=int, default=100)
    parser.add_argument('--critic_h1', help='Critic layer 1 hidden size', type=int, default=100)
    parser.add_argument('--critic_h2', help='Critic layer 2 hidden size', type=int, default=100)
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

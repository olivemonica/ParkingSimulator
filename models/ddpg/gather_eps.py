import os
import time
from collections import deque
import pickle

from .ddpg import DDPG
import baselines.common.tf_util as U

from baselines import logger
import numpy as np
import tensorflow as tf
from mpi4py import MPI

import numpy as np
import json

def train(env, nb_epochs, nb_epoch_cycles, render_eval, reward_scale, render, param_noise, actor, critic,
    normalize_returns, normalize_observations, critic_l2_reg, actor_lr, critic_lr, action_noise,
    popart, gamma, clip_norm, nb_train_steps, nb_rollout_steps, nb_eval_steps, batch_size, memory,
    model_dir, ckpt_dir, save_every, tau=0.001, eval_env=None, param_noise_adaption_interval=50):
    rank = MPI.COMM_WORLD.Get_rank()

    assert (np.abs(env.action_space.low) == env.action_space.high).all()  # we assume symmetric actions.
    max_action = env.action_space.high
    print('max_action: {}'.format(max_action))
    logger.info('scaling actions by {} before executing in env'.format(max_action))
    agent = DDPG(actor, critic, memory, env.state_space.shape, env.action_space.shape,
        gamma=gamma, tau=tau, normalize_returns=normalize_returns, normalize_observations=normalize_observations,
        batch_size=batch_size, action_noise=action_noise, param_noise=param_noise, critic_l2_reg=critic_l2_reg,
        actor_lr=actor_lr, critic_lr=critic_lr, enable_popart=popart, clip_norm=clip_norm,
        reward_scale=reward_scale)
    #logger.info('Using agent with the following configuration:')
    #logger.info(str(agent.__dict__.items()))

    # Set up logging stuff only for a single worker.
    if rank == 0:
        saver = tf.train.Saver()
    else:
        saver = None

    if ckpt_dir is None:
        all_rewards = []
    else:
        all_rewards = pickle.load(open(os.path.join(ckpt_dir, 'rewards.p'),'rb'))

    step = 0
    episode = len(all_rewards)
    eval_episode_rewards_history = deque(maxlen=100)
    episode_rewards_history = deque(maxlen=100)
    with U.single_threaded_session() as sess:
        # Prepare everything.
        agent.initialize(sess,ckpt_dir,saver)
        sess.graph.finalize()

        agent.reset()
        obs = env._reset(len(all_rewards))
        #obs = env._reset(200)
        start_dist = obs[5]
        start_angle = obs[6]
        ep_observations = []
        ep_actions = []
        ep_rewards = []
        eps = 0

        if eval_env is not None:
            eval_obs = eval_env._reset(0)
        done = False
        episode_reward = 0.
        episode_step = 0
        episodes = 0
        t = 0

        epoch = 0
        start_time = time.time()

        ep_times = {'prediction': [], 'stepping': [], 'substep': [], 'training':[]}

        epoch_episode_rewards = []
        epoch_episode_steps = []
        epoch_episode_eval_rewards = []
        epoch_episode_eval_steps = []
        epoch_start_time = time.time()
        epoch_actions = []
        epoch_qs = []
        epoch_episodes = 0
        for epoch in range(nb_epochs):
            if epoch % save_every == 0:
                print('Saving')
                agent.saver.save(agent.sess,os.path.join(model_dir, 'model.ckpt'))
                pickle.dump(all_rewards,open(os.path.join(model_dir, 'rewards.p'),'wb'))
                pickle.dump(ep_times,open(os.path.join(model_dir, 'op_times.p'),'wb'))

            for cycle in range(nb_epoch_cycles):
                #create lists to save operation times
                pred_times = []
                step_times = []
                substep_times = [[],[],[]]
                train_times = []

                # Perform rollouts.
                for t_rollout in range(nb_rollout_steps):

                    '''
                    # Only apply noise if performance in last hundred episodes was poor
                    if len(all_rewards) >= 1000 and len([r for r in all_rewards[-500:] if r[2]]) > 400:
                        apply_noise = False
                    else:
                        apply_noise = True
                    '''
                    apply_noise = True

                    # Predict next action.
                    act_start = time.time()
                    action, q = agent.pi(obs, apply_noise=apply_noise, compute_Q=True)
                    pred_times.append(time.time() - act_start)

                    assert action.shape == env.action_space.shape

                    if env.special_ep:
                        # action[1] = np.clip(action[1],-0.001,0.001)
                        action[1] = 0

                        if not env.corner_in_spot:
                            upper_lim = 2
                            lower_lim = 1.5
                        elif env.corner_in_spot and not env.in_spot:
                            upper_lim = 1
                            lower_lim = 0.5
                        else:
                            upper_lim = 0
                            lower_lim = -1

                        if env.car.speed > upper_lim:
                            action[0] = np.clip(np.minimum(action[0],
                                                -0.4 - np.random.exponential(0.3)), -1, 1)
                        if env.car.speed < lower_lim:
                            action[0] = np.clip(np.maximum(action[0],
                                                0.4 + np.random.exponential(0.3)), -1, 1)
                        # print(action)
                    # Execute next action.
                    if rank == 0 and render:
                        env._render()
                    assert max_action.shape == action.shape
                    step_start = time.time()
                    new_obs, r, done, substep_time = env._step(max_action * action)  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
                    step_times.append(time.time() - step_start)
                    ep_observations.append(np.expand_dims(np.array(obs), axis=0))
                    ep_actions.append(np.expand_dims(np.array(max_action * action), axis=0))
                    ep_rewards.append(r)

                    #print(len(substep_times),substep_time)
                    for i in range(len(substep_time)):
                        substep_times[i].append(substep_time[i])

                    t += 1
                    if rank == 0 and render:
                        env._render()
                    episode_reward += r
                    episode_step += 1

                    # Book-keeping.
                    epoch_actions.append(action)
                    epoch_qs.append(q)
                    agent.store_transition(obs, action, r, new_obs, done)
                    obs = new_obs

                    if done or episode_step > 100:
                        # Episode done.
                        epoch_episode_rewards.append(episode_reward)
                        episode_rewards_history.append(episode_reward)
                        epoch_episode_steps.append(episode_step)
                        epoch_episodes += 1
                        episodes += 1

                        if not env.special_ep:
                            all_rewards.append([episodes,
                                                episode_reward,
                                                env.parked,
                                                start_dist,
                                                obs[5],
                                                start_angle,
                                                env.special_ep])
                            '''
                            if env.parked:
                                print('------------PARKED------------')
                            else:
                                print('Episode Failed')
                            '''
                        '''
                        if env.special_ep and env.parked:
                            print('Special Episode Parked')
                        '''

                        episode_reward = 0.
                        episode_step = 0


                        agent.reset()
                        obs = env._reset(len(all_rewards))

                        # save episode
                        ep_dict = {'observations': np.concatenate(ep_observations).tolist(),
                                   'actions': np.concatenate(ep_actions).tolist(),
                                   'rewards': ep_rewards,
                                   'length': len(ep_observations)}
                        ep_file = 'episode_' + str(eps) + '.json'
                        ep_path = os.path.join(model_dir, ep_file)
                        with open(ep_path, 'w') as fp:
                            json.dump(ep_dict, fp)

                        # reset episode trackers
                        eps += 1
                        ep_observations = []
                        ep_actions = []
                        ep_rewards = []

                        start_dist = obs[5]
                        start_angle = obs[6]

                #print('Episode: {}, reward: {}'.format(cycle,np.round(episode_reward,2)))

                # Train.
                epoch_actor_losses = []
                epoch_critic_losses = []
                epoch_adaptive_distances = []
                for t_train in range(nb_train_steps):
                    # Adapt param noise, if necessary.
                    if memory.nb_entries >= batch_size and t_train % param_noise_adaption_interval == 0:
                        distance = agent.adapt_param_noise()
                        epoch_adaptive_distances.append(distance)

                    train_start = time.time()
                    cl, al = agent.train()
                    train_times.append(time.time() - train_start)

                    epoch_critic_losses.append(cl)
                    epoch_actor_losses.append(al)
                    agent.update_target_net()

                # Evaluate.
                eval_episode_rewards = []
                eval_qs = []
                if eval_env is not None:
                    eval_episode_reward = 0.
                    for t_rollout in range(nb_eval_steps):
                        eval_action, eval_q = agent.pi(eval_obs, apply_noise=False, compute_Q=True)
                        eval_obs, eval_r, eval_done, eval_info = eval_env.step(max_action * eval_action)  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
                        if render_eval:
                            eval_env._render()
                        eval_episode_reward += eval_r

                        eval_qs.append(eval_q)
                        if eval_done:
                            eval_obs = eval_env._reset(0)
                            eval_episode_rewards.append(eval_episode_reward)
                            eval_episode_rewards_history.append(eval_episode_reward)
                            eval_episode_reward = 0.
                ep_times['prediction'].append(np.mean(pred_times))
                ep_times['stepping'].append(np.mean(step_times))
                #new_substeptimes = [np.mean(substep_times),np.std(substep_times),np.max(substep_times)]
                new_substeptimes = []
                for i in range(len(substep_times)):
                    st = np.mean(substep_times[i])
                    new_substeptimes.append(st)
                #print(new_substeptimes)
                ep_times['substep'].append(new_substeptimes)
                ep_times['training'].append(np.mean(train_times))

            mpi_size = MPI.COMM_WORLD.Get_size()
            # Log stats.
            # XXX shouldn't call np.mean on variable length lists
            duration = time.time() - start_time
            stats = agent.get_stats()
            combined_stats = stats.copy()
            combined_stats['rollout/return'] = np.mean(epoch_episode_rewards)
            combined_stats['rollout/return_history'] = np.mean(episode_rewards_history)
            combined_stats['rollout/episode_steps'] = np.mean(epoch_episode_steps)
            combined_stats['rollout/actions_mean'] = np.mean(epoch_actions)
            combined_stats['rollout/Q_mean'] = np.mean(epoch_qs)
            combined_stats['train/loss_actor'] = np.mean(epoch_actor_losses)
            combined_stats['train/loss_critic'] = np.mean(epoch_critic_losses)
            combined_stats['train/param_noise_distance'] = np.mean(epoch_adaptive_distances)
            combined_stats['total/duration'] = duration
            combined_stats['total/steps_per_second'] = float(t) / float(duration)
            combined_stats['total/episodes'] = episodes
            combined_stats['rollout/episodes'] = epoch_episodes
            combined_stats['rollout/actions_std'] = np.std(epoch_actions)
            # Evaluation statistics.
            if eval_env is not None:
                combined_stats['eval/return'] = eval_episode_rewards
                combined_stats['eval/return_history'] = np.mean(eval_episode_rewards_history)
                combined_stats['eval/Q'] = eval_qs
                combined_stats['eval/episodes'] = len(eval_episode_rewards)
            def as_scalar(x):
                if isinstance(x, np.ndarray):
                    assert x.size == 1
                    return x[0]
                elif np.isscalar(x):
                    return x
                else:
                    raise ValueError('expected scalar, got %s'%x)
            combined_stats_sums = MPI.COMM_WORLD.allreduce(np.array([as_scalar(x) for x in combined_stats.values()]))
            combined_stats = {k : v / mpi_size for (k,v) in zip(combined_stats.keys(), combined_stats_sums)}

            # Total statistics.
            combined_stats['total/epochs'] = epoch + 1
            combined_stats['total/steps'] = t

            for key in sorted(combined_stats.keys()):
                logger.record_tabular(key, combined_stats[key])
            logger.dump_tabular()
            logger.info('')
            logdir = logger.get_dir()
            if rank == 0 and logdir:
                if hasattr(env, 'get_state'):
                    with open(os.path.join(logdir, 'env_state.pkl'), 'wb') as f:
                        pickle.dump(env.get_state(), f)
                if eval_env and hasattr(eval_env, 'get_state'):
                    with open(os.path.join(logdir, 'eval_env_state.pkl'), 'wb') as f:
                        pickle.dump(eval_env.get_state(), f)

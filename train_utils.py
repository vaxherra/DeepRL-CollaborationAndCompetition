from collections import deque
import numpy as np
import torch


def _get_actions(states, add_noise, agent_0, agent_1):
    """
    Get actions for each two agents and return a single concatenated array with both actions

    :param states: current state of the environment
    :param add_noise: boolean to add noise to the actions
    :param agent_0: agent 0
    :param agent_1: agent 1

    """
    action_0 = agent_0.act(states, add_noise)    # agent 0 chooses an action
    action_1 = agent_1.act(states, add_noise)    # agent 1 chooses an action
    return np.concatenate((action_0, action_1), axis=0).flatten()


def maddpg(env, brain_name, agent_0, agent_1, n_episodes=2000, train_mode=True, print_every=10, scoring_episodes=100,
           add_noise: bool = True, target_score=0.5):
    """Multi-Agent Deep Deterministic Policy Gradient (MADDPG)

    :param env: environment
    :param brain_name: brain name
    :param agent_0: agent 0
    :param agent_1: agent 1
    :param n_episodes: maximum number of training episodes
    :param max_t: maximum number of timesteps per episode
    :param train_mode: boolean to set the environment to train mode
    :param print_every: interval to print the average score
    :param scoring_episodes: number of episodes to average the score over
    :param add_noise: boolean to add noise to the actions
    :param target_score: target score to solve the environment


    :return: scores: list of scores from each episode, scores_window: list of scores from the last scoring_episodes
    """

    num_agents = 2
    scores_queue = deque(maxlen=scoring_episodes)
    scores_arr = []
    average_score = []

    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=train_mode)[brain_name]  # reset environment
        states = np.reshape(env_info.vector_observations, (1, 48))  # get states
        agent_0.reset()
        agent_1.reset()
        scores = np.zeros(num_agents)
        while True:
            actions = _get_actions(states, add_noise, agent_0, agent_1)  # choose agent actions
            env_info = env.step(actions)[brain_name]  # send agents' actions to the environment
            next_states = np.reshape(env_info.vector_observations, (1, 48))  # get agents' next states
            rewards = env_info.rewards  # get rewards
            done = env_info.local_done  # episode state
            agent_0.step(states, actions, rewards[0], next_states, done, 0)  # learn agent 1
            agent_1.step(states, actions, rewards[1], next_states, done, 1)  # Learn agent 2
            scores += np.max(rewards)  # add the best score
            states = next_states  # roll over states to next time step
            if np.any(done):  # exit loop if episode finished
                break

        episode_best_score = np.max(scores)
        scores_queue.append(episode_best_score)
        scores_arr.append(episode_best_score)
        average_score.append(np.mean(scores_queue))

        # print results
        if i_episode % print_every == 0:
            print(
                'Episodes {:0>4d}-{:0>4d}\t Highest Reward: {:.3f}\t Lowest Reward: {:.3f}\t Average Score: {:.3f}'.format(
                    i_episode - print_every, i_episode, np.max(scores_arr[-print_every:]),
                    np.min(scores_arr[-print_every:]), average_score[-1]))

        # determine if environment is solved and keep best performing models
        if average_score[-1] >= target_score:
            print('<-- Environment solved in {:d} episodes! \
                \n<-- Average Score: {:.3f} over past {:d} episodes'.format(
                i_episode - scoring_episodes, average_score[-1], scoring_episodes))
            # save weights
            torch.save(agent_0.actor_local.state_dict(), 'checkpoint_actor_0.pth')
            torch.save(agent_0.critic_local.state_dict(), 'checkpoint_critic_0.pth')
            torch.save(agent_1.actor_local.state_dict(), 'checkpoint_actor_1.pth')
            torch.save(agent_1.critic_local.state_dict(), 'checkpoint_critic_1.pth')
            break

    return scores_arr, average_score
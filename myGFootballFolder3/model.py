import gfootball.env as football_env
from network import Net
import numpy as np
import math
import torch
import torch.nn as nn
# global_environment_name = "11_vs_11_stochastic" ## select the scenario you want to run
# numEpisodes = 10 # number of episodes to run with one network
# net = Net() # temporary network. This will eventually be passed to the function named model


def model_func(net,numEpisodes=10,global_environment_name="11_vs_11_stochastic"):

    observations = []
    envs = [football_env.create_environment(env_name=global_environment_name, stacked=True, logdir='/tmp/football', write_goal_dumps=False, representation="extracted", write_full_episode_dumps=False, render=False) for i in range(numEpisodes)]
    for env in envs:
        env.reset()
        obs, rew, done, info = env.step(11) # passing short pass as first action to all environments
        mobs = obs.transpose(2,1,0)
        observations.append(mobs)
        
    steps = 1

    observations = np.array(observations)
    batch = torch.from_numpy(observations)
    actions = torch.argmax(net(batch),dim=1)


    while True:
        observations = []
        ### CODESMELL ####
        rewards = [] # assuming that the reward at the last step is the reward for the entire episode
        for id,env in enumerate(envs):
            obs, rew, done, info = env.step(actions[id].item()) # passing short pass as first action to all environments
            mobs = obs.transpose(2,1,0)
            observations.append(mobs)
            rewards.append(rew)

        steps += 1
        # print(steps)
        if done:
            break

        observations = np.array(observations)
        actions = torch.argmax(net(torch.from_numpy(observations)),dim=1)

    return np.mean(np.array(rewards))
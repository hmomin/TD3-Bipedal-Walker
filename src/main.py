import csv
import gymnasium as gym
from Agent import Agent
from os import path
from time import time

# HYPERPARAMETERS BELOW
gamma = 0.99  # discount factor for rewards
learningRate = 3e-4  # learning rate for actor and critic networks
tau = 0.005  # tracking parameter used to update target networks slowly
actionSigma = 0.1  # contributes noise to deterministic policy output
trainingSigma = 0.2  # contributes noise to target actions
trainingClip = 0.5  # clips target actions to keep them close to true actions
miniBatchSize = 100  # how large a mini-batch should be when updating
policyDelay = 2  # how many steps to wait before updating the policy
resume = True  # resume from previous checkpoint if possible?
render = True  # render out the environment on-screen?

envName = "BipedalWalker-v3"

for trial in range(64):
    renderMode = "human" if render else None
    env = gym.make(envName, render_mode=renderMode)
    env.name = envName + "_" + str(trial)
    csvName = env.name + "-data.csv"
    agent = Agent(env, learningRate, gamma, tau, resume)
    state, info = env.reset()
    step = 0
    runningReward = None

    # determine the last episode if we have saved training in progress
    numEpisode = 0
    if resume and path.exists(csvName):
        fileData = list(csv.reader(open(csvName)))
        lastLine = fileData[-1]
        numEpisode = int(lastLine[0])

    start_time = time()

    while numEpisode <= 2000:
        # choose an action from the agent's policy
        action = agent.getNoisyAction(state, actionSigma)
        # take a step in the environment and collect information
        nextState, reward, terminated, truncated, info = env.step(action)
        # store data in buffer
        agent.buffer.store(state, action, reward, nextState, terminated)

        if terminated or truncated:
            elapsed_time = time() - start_time
            start_time = time()
            print(f"training episode: {elapsed_time} s")
            numEpisode += 1
            # evaluate the deterministic agent on a test episode
            sumRewards = 0.0
            state, info = env.reset()
            terminated = truncated = False
            while not terminated and not truncated:
                action = agent.getDeterministicAction(state)
                nextState, reward, terminated, truncated, info = env.step(action)
                if render:
                    env.render()
                state = nextState
                sumRewards += reward
            elapsed_time = time() - start_time
            start_time = time()
            print(f"testing episode: {elapsed_time} s")
            state, info = env.reset()
            # keep a running average to see how well we're doing
            runningReward = (
                sumRewards
                if runningReward is None
                else runningReward * 0.99 + sumRewards * 0.01
            )
            # log progress in csv file
            fields = [numEpisode, sumRewards, runningReward]
            with open(env.name + "-data.csv", "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(fields)
            agent.save()
            # print episode tracking
            print(
                f"episode {numEpisode:6d} --- "
                + f"total reward: {sumRewards:7.2f} --- "
                + f"running average: {runningReward:7.2f}",
                flush=True,
            )
        else:
            state = nextState
        step += 1

        shouldUpdatePolicy = step % policyDelay == 0
        agent.update(miniBatchSize, trainingSigma, trainingClip, shouldUpdatePolicy)

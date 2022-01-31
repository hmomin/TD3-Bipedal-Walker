import csv
import gym
from Agent import Agent
from os import path

# HYPERPARAMETERS BELOW
gamma = 0.99         # discount factor for rewards
learningRate = 3e-4  # learning rate for actor and critic networks
tau = 0.005          # tracking parameter used to update target networks slowly
actionSigma = 0.1    # contributes noise to deterministic policy output
trainingSigma = 0.2  # contributes noise to target actions
trainingClip = 0.5   # clips target actions to keep them close to the true actions
miniBatchSize = 100  # how large a mini-batch should be when updating
policyDelay = 2      # how many steps to wait before updating the policy
resume = True        # resume from previous checkpoint if possible?
render = False       # render out the environment on-screen?

envName = "BipedalWalker-v3"

for trial in range(12):
    env = gym.make(envName)
    env.name = envName + "_" + str(trial)
    csvName = env.name + '-data.csv'
    agent = Agent(env, learningRate, gamma, tau, resume)
    state = env.reset()
    step = 0
    runningReward = None

    # determine the last episode if we have saved training in progress
    numEpisode = 0
    if path.exists(csvName):
        fileData = list(csv.reader(open(csvName)))
        lastLine = fileData[-1]
        lastEpisode = int(lastLine[0])
        numEpisode = lastEpisode + 1

    while numEpisode <= 2000:
        # choose an action from the agent's policy
        action = agent.getNoisyAction(state, actionSigma)
        # take a step in the environment and collect information
        nextState, reward, done, info = env.step(action)
        # store data in buffer
        agent.buffer.store(state, action, reward, nextState, done)

        if done:
            numEpisode += 1
            # evaluate the deterministic agent on a test episode
            sumRewards = 0
            state = env.reset()
            done = False
            while not done:
                action = agent.getDeterministicAction(state)
                nextState, reward, done, info = env.step(action)
                if render:
                    env.render()
                state = nextState
                sumRewards += reward
            state = env.reset()
            # keep a running average to see how well we're doing
            runningReward = sumRewards\
                if runningReward is None\
                else runningReward*0.99 + sumRewards*0.01
            # log progress in csv file
            fields = [numEpisode, sumRewards, runningReward]
            with open(env.name + '-data.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(fields)
            agent.save()
            # print episode tracking
            print(
                f"episode {numEpisode:6d} --- total reward: {sumRewards:7.2f} --- running average: {runningReward:7.2f}"
            )
        else:
            state = nextState
        step += 1
        
        shouldUpdatePolicy = step % policyDelay == 0
        agent.update(miniBatchSize, trainingSigma, trainingClip, shouldUpdatePolicy)
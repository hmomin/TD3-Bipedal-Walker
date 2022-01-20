import csv
import gym
from Agent import Agent

envName = "BipedalWalker-v3"
# HYPERPARAMETERS BELOW
gamma = 0.99         # discount factor for rewards
learningRate = 1e-3  # learning rate for actor and critic networks
tau = 0.005          # tracking parameter used to update target networks slowly
actionSigma = 0.1    # contributes noise to deterministic policy output
trainingSigma = 0.2  # contributes noise to target actions
trainingClip = 0.5   # clips target actions to keep them close to the true actions
miniBatchSize = 100  # how large a mini-batch should be when updating
policyDelay = 2      # how many steps to wait before updating the policy
saveDelay = 10000    # how many steps to wait before saving the agent networks
resume = True        # resume from previous checkpoint if possible?
render = True        # render out the game on-screen?

env = gym.make(envName)
env.name = envName
agent = Agent(env, learningRate, gamma, tau, resume)
state = env.reset()
numEpisode = 0
step = 0
runningReward = None

while True:
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
        # keep a running average to see how well we're doing
        runningReward = sumRewards\
            if runningReward is None\
            else runningReward*0.99 + sumRewards*0.01
        # episode tracking
        print(
            f"episode {numEpisode:6d} --- total reward: {sumRewards:7.2f} --- running average: {runningReward:7.2f}"
        )
        # log progress in csv file
        fields = [numEpisode, sumRewards, runningReward]
        with open(env.name + '-data.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(fields)
        state = env.reset()
    else:
        state = nextState
    step += 1
    
    shouldUpdatePolicy = step % policyDelay == 0
    agent.update(miniBatchSize, trainingSigma, trainingClip, shouldUpdatePolicy)
    if step % saveDelay == 0:
        agent.save()
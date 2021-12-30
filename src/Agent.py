import numpy as np
import os
import pickle
import torch as T
import torch.nn as nn
from copy import deepcopy
from gym.core import Env
from Buffer import Buffer
from Network import Network

class Agent():
    def __init__(
        self, env: Env, learningRate: float, gamma: float, tau: float,
        shouldLoad: bool=True, saveFolder: str='networks'
    ):
        self.observationDim = env.observation_space.shape[0]
        self.actionDim = env.action_space.shape[0]
        self.gamma = gamma
        self.tau = tau
        self.buffer = Buffer(self.observationDim, self.actionDim)
        # check if the saveFolder path exists
        if not os.path.isdir(saveFolder):
            os.mkdir(saveFolder)
        self.envName = os.path.join(saveFolder, env.name + '.')
        name = self.envName
        # initialize the actor and critics
        self.actor = pickle.load(open(name + 'Actor', 'rb'))\
            if shouldLoad and os.path.exists(name + 'Actor') else Network(
                [self.observationDim, 256, 256, self.actionDim],
                nn.Tanh,
                learningRate
            )
        self.critic1 = pickle.load(open(name + 'Critic1', 'rb'))\
            if shouldLoad and os.path.exists(name + 'Critic1') else Network(
                [self.observationDim + self.actionDim, 256, 256, 1],
                nn.Identity,
                learningRate
            )
        self.critic2 = pickle.load(open(name + 'Critic2', 'rb'))\
            if shouldLoad and os.path.exists(name + 'Critic2') else Network(
                [self.observationDim + self.actionDim, 256, 256, 1],
                nn.Identity,
                learningRate
            )
        # create target networks
        self.targetActor = pickle.load(open(name + 'TargetActor', 'rb'))\
            if shouldLoad and os.path.exists(name + 'TargetActor') else\
            deepcopy(self.actor)
        self.targetCritic1 = pickle.load(open(name + 'TargetCritic1', 'rb'))\
            if shouldLoad and os.path.exists(name + 'TargetCritic1') else\
            deepcopy(self.critic1)
        self.targetCritic2 = pickle.load(open(name + 'TargetCritic2', 'rb'))\
            if shouldLoad and os.path.exists(name + 'TargetCritic2') else\
            deepcopy(self.critic2)
    
    def getNoisyAction(self, state: np.ndarray, sigma: float) -> np.ndarray:
        deterministicAction = self.getDeterministicAction(state)
        noise = np.random.normal(0, sigma, deterministicAction.shape)
        return np.clip(deterministicAction + noise, -1, +1)
    
    def getDeterministicAction(self, state: np.ndarray) -> np.ndarray:
        actions: T.Tensor = self.actor.forward(T.tensor(state).cuda())
        return actions.cpu().detach().numpy()
    
    def update(
        self, miniBatchSize: int, trainingSigma: float, trainingClip: float,
        updatePolicy: bool
    ):
        # randomly sample a mini-batch from the replay buffer
        miniBatch = self.buffer.getMiniBatch(miniBatchSize)
        # create tensors to start generating computational graph
        states = T.tensor(miniBatch["states"], requires_grad=True).cuda()
        actions = T.tensor(miniBatch["actions"], requires_grad=True).cuda()
        rewards = T.tensor(miniBatch["rewards"], requires_grad=True).cuda()
        nextStates = T.tensor(miniBatch["nextStates"], requires_grad=True).cuda()
        dones = T.tensor(miniBatch["doneFlags"], requires_grad=True).cuda()
        # compute the targets
        targets = self.computeTargets(
            rewards, nextStates, dones, trainingSigma, trainingClip
        )
        # do a single step on each critic network
        Q1Loss = self.computeQLoss(self.critic1, states, actions, targets)
        self.critic1.gradientDescentStep(Q1Loss, True)
        Q2Loss = self.computeQLoss(self.critic2, states, actions, targets)
        self.critic2.gradientDescentStep(Q2Loss)
        if updatePolicy:
            # do a single step on the actor network
            policyLoss = self.computePolicyLoss(states)
            self.actor.gradientDescentStep(policyLoss)
            # update target networks
            self.updateTargetNetwork(self.targetActor, self.actor)
            self.updateTargetNetwork(self.targetCritic1, self.critic1)
            self.updateTargetNetwork(self.targetCritic2, self.critic2)
    
    def computeTargets(
        self, rewards: T.Tensor, nextStates: T.Tensor, dones: T.Tensor,
        trainingSigma: float, trainingClip: float
    ) -> T.Tensor:
        targetActions = self.targetActor.forward(nextStates.float())
        # create additive noise for target actions
        noise = np.random.normal(0, trainingSigma, targetActions.shape)
        clippedNoise = T.tensor(np.clip(noise, -trainingClip, +trainingClip)).cuda()
        targetActions = T.clip(targetActions + clippedNoise, -1, +1)
        # compute targets
        targetQ1Values = T.squeeze(
            self.targetCritic1.forward(T.hstack([nextStates, targetActions]).float())
        )
        targetQ2Values = T.squeeze(
            self.targetCritic2.forward(T.hstack([nextStates, targetActions]).float())
        )
        targetQValues = T.minimum(targetQ1Values, targetQ2Values)
        return rewards + self.gamma*(1 - dones)*targetQValues
    
    def computeQLoss(
        self, network: Network, states: T.Tensor, actions: T.Tensor, targets: T.Tensor
    ) -> T.Tensor:
        # compute the MSE of the Q function with respect to the targets
        QValues = T.squeeze(network.forward(T.hstack([states, actions]).float()))
        return T.square(QValues - targets).mean()

    def computePolicyLoss(self, states: T.Tensor): 
        actions = self.actor.forward(states.float())
        QValues = T.squeeze(self.critic1.forward(T.hstack([states, actions]).float()))
        return -QValues.mean()

    def updateTargetNetwork(self, targetNetwork: Network, network: Network):
        with T.no_grad():
            for targetParameter, parameter in zip(
                targetNetwork.parameters(), network.parameters()
            ):
                targetParameter.mul_(1 - self.tau)
                targetParameter.add_(self.tau*parameter)

    def save(self):
        name = self.envName
        pickle.dump(self.actor, open(name + 'Actor', 'wb'))
        pickle.dump(self.critic1, open(name + 'Critic1', 'wb'))
        pickle.dump(self.critic2, open(name + 'Critic2', 'wb'))
        pickle.dump(self.targetActor, open(name + 'TargetActor', 'wb'))
        pickle.dump(self.targetCritic1, open(name + 'TargetCritic1', 'wb'))
        pickle.dump(self.targetCritic2, open(name + 'TargetCritic2', 'wb'))
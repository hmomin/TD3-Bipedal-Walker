import numpy as np
import torch as T


class Buffer:
    def __init__(
        self,
        observationDim: int,
        actionDim: int,
        device: T.device,
        size: int = 1_000_000,
    ):
        self.device = device
        # use a fixed-size buffer to prevent constant list instantiations
        self.states = T.zeros((size, observationDim), device=self.device)
        self.actions = T.zeros((size, actionDim), device=self.device)
        self.rewards = T.zeros(size, device=self.device)
        self.nextStates = T.zeros((size, observationDim), device=self.device)
        self.doneFlags = T.zeros(size, device=self.device)
        # use a pointer to keep track of where in the buffer we are
        self.pointer = 0
        # use current size to ensure we don't train on any non-existent data points
        self.currentSize = 0
        self.size = size

    def store(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        nextState: np.ndarray,
        doneFlag: bool,
    ):
        # store all the data for this transition
        ptr = self.pointer
        tensorState = T.tensor(state, device=self.device)
        tensorAction = T.tensor(action, device=self.device)
        tensorNextState = T.tensor(nextState, device=self.device)
        self.states[ptr, :] = tensorState
        self.actions[ptr, :] = tensorAction
        self.rewards[ptr] = reward
        self.nextStates[ptr, :] = tensorNextState
        self.doneFlags[ptr] = float(doneFlag)
        # update the pointer and current size
        self.pointer = (self.pointer + 1) % self.size
        self.currentSize = min(self.currentSize + 1, self.size)

    def getMiniBatch(self, size: int) -> dict[str, T.Tensor]:
        # ensure size is not bigger than the current size of the buffer
        size = min(size, self.currentSize)
        # generate random indices
        indices = T.randint(0, self.currentSize, (size,), device=self.device)
        # return the mini-batch of transitions
        return {
            "states": self.states[indices, :],
            "actions": self.actions[indices, :],
            "rewards": self.rewards[indices],
            "nextStates": self.nextStates[indices, :],
            "doneFlags": self.doneFlags[indices],
        }

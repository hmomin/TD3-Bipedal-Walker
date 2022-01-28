import numpy as np

class Buffer():
    def __init__(self, observationDim: int, actionDim: int, size: int=1_000_000):
        # use a fixed-size buffer to prevent constant list instantiations
        self.states = np.zeros((size, observationDim))
        self.actions = np.zeros((size, actionDim))
        self.rewards = np.zeros(size)
        self.nextStates = np.zeros((size, observationDim))
        self.doneFlags = np.zeros(size)
        # use a pointer to keep track of where in the buffer we are
        self.pointer = 0
        # use current size to ensure we don't train on any non-existent data points
        self.currentSize = 0
        self.size = size
    
    def store(
        self, state: np.ndarray, action: np.ndarray, reward: float, nextState: np.ndarray,
        doneFlag: bool
    ):
        # store all the data for this transition
        ptr = self.pointer
        self.states[ptr] = state
        self.actions[ptr] = action
        self.rewards[ptr] = reward
        self.nextStates[ptr] = nextState
        self.doneFlags[ptr] = doneFlag
        # update the pointer and current size
        self.pointer = (self.pointer + 1) % self.size
        self.currentSize = min(self.currentSize + 1, self.size)
    
    def getMiniBatch(self, size: int) -> dict:
        # ensure size is not bigger than the current size of the buffer
        size = min(size, self.currentSize)
        # generate random indices
        indices = np.random.choice(self.currentSize, size, replace=False)
        # return the mini-batch of transitions
        return {
            "states": self.states[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "nextStates": self.nextStates[indices],
            "doneFlags": self.doneFlags[indices],
        }
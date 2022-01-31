<p align="center">
    <img src="https://dr3ngl797z54v.cloudfront.net/bipedal-walker-logo.png" width="100%" alt="bipedal-walker-logo">
</p>

# Introduction

This script trains an agent with Twin Delayed DDPG (TD3) to solve the Bipedal Walker challenge from OpenAI.

In order to run this script, [NumPy](https://numpy.org/install/), the [OpenAI Gym toolkit](https://gym.openai.com/docs/), and [PyTorch](https://pytorch.org/get-started/locally/) will need to be installed.

Each step through the Bipedal Walker environment takes the general form:

```python
state, reward, done, info = env.step(action)
```

and the goal is for the agent to take actions that maximize the cumulative reward achieved for the episode's duration. In this specific environment, the state and action space are continuous and the state space is 8-dimensional while the action space is 4-dimensional. The state space consists of position and velocity measurements of the walker and its joints, while the action space consists of motor torques that can be applied to the four controllable joints.

Since the action space is continuous, a naive application of vanilla policy gradient would likely have relatively poor or limited performance in practice. In environments involving continuous action spaces, it is often preferable to make use of DDPG or TD3 among other deep reinforcement learning (DRL) algorithms, since these two have been specifically designed to make use of continuous action spaces.

To learn more about how the agent receives rewards, see [here](https://gym.openai.com/envs/BipedalWalker-v2/).

# Algorithm

A detailed discussion of the TD3 algorithm with proper equation typesetting is provided in the supplemental material [here](https://github.com/hmomin/TD3-Bipedal-Walker/blob/main/TD3-supplemental-material.pdf).

# Results

Solving the Bipedal Walker challenge requires training the agent to safely walk all the way to the end of the platform without falling over and while using as little motor torque as possible. The agent's ability to do this was quite abysmal in the beginning.

<p align="center">
    <img src="https://dr3ngl797z54v.cloudfront.net/bipedal-walker-failure.gif" width="80%" alt="failure...">'
</p>

After training the agent overnight on a GPU, it could gracefully complete the challenge with ease!

<p align="center">
    <img src="https://dr3ngl797z54v.cloudfront.net/bipedal-walker-success.gif" width="80%" alt="success!">
</p>

Below, the performance of the agent over 12 trial runs is documented. The shaded region represents a standard deviation of the average evaluation over all trials. The curve has been smoothed with a Savitzky-Golay filter for visual clarity.

<p align="center">
    <img src="https://dr3ngl797z54v.cloudfront.net/bipedal-walker-12-trial-avg.png" width="80%" alt="training-results">
</p>

# References

- [Continuous Control With Deep Reinforcement Learning - Lillicrap et al.](https://arxiv.org/abs/1509.02971)
- [Addressing Function Approximation Error in Actor-Critic Methods - Fujimoto et al.](https://arxiv.org/abs/1802.09477)

# License

All files in the repository are under the MIT license.
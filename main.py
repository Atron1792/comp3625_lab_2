from gymnasium.envs.classic_control.cartpole import CartPoleEnv
from agent import CartPoleAgent
import numpy as np
import scipy
import matplotlib.pyplot as plt
np.set_printoptions(precision=2)


def test_agent(parameters, render=False):
    """
    This function tests a CartPoleAgent with the given weights.
    The agent is tested over 10 episodes, and the total number of steps achieved is returned.
    """

    max_steps_per_episode = 500
    cumulative_reward = 0

    for rep in range(10):
        agent = CartPoleAgent(parameters)
        env = CartPoleEnv(render_mode="human" if render else None)
        

        observation, info = env.reset()

        for step in range(max_steps_per_episode):
            action = agent.get_action(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            cumulative_reward += reward
            if terminated or truncated:
                break

    print(f'tested parameters: {parameters}, cumulative reward: {cumulative_reward}')
    return cumulative_reward



# watch how an agent with randomly-initialized parameters does:
# test_agent(parameters=np.random.uniform(-1, 1, size=5), render=True)


# Write a search to find the best parameters for the CartPoleAgent.
params = [ 0.0, 0.0, 0.0, 0.0, 0.0]
rewards_over_time = []

def maximize(parameters):
    reward = test_agent(parameters, render=False)
    rewards_over_time.append(reward)
    return -reward

scipy.optimize.minimize(maximize, x0=params, method='Nelder-Mead')

# ploting
plt.plot(rewards_over_time)
plt.xlabel("Number of Parameter Set Evaluations")
plt.ylabel("Cumulative Reward")
plt.title("Cart-Pole Performance")
plt.savefig("performance_over_time.png")
plt.close()
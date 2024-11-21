import numpy as np
import random
import gym
from gym import spaces
import matplotlib.pyplot as plt

class GridEnv(gym.Env):
    def __init__(self):
        super(GridEnv, self).__init__()
        self.grid_size = 4
        self.start_state = (0, 0)
        self.goal_state = (3, 3)
        self.current_state = self.start_state
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(self.grid_size * self.grid_size)

    def reset(self):
        self.current_state = self.start_state
        return self.current_state

    def step(self, action):
        i, j = self.current_state
        if action == 0 and i > 0:
            next_state = (i - 1, j)
        elif action == 1 and i < self.grid_size - 1:
            next_state = (i + 1, j)
        elif action == 2 and j > 0:
            next_state = (i, j - 1)
        elif action == 3 and j < self.grid_size - 1:
            next_state = (i, j + 1)
        else:
            next_state = self.current_state

        # Calculate rewards
        if next_state == self.goal_state:
            reward = 10
            done = True
        elif next_state == self.current_state:
            reward = -5
            done = False
        else:
            reward = -1
            done = False

        self.current_state = next_state
        return self.current_state, reward, done, {}

    def render(self, mode='human'):
        grid = np.zeros((self.grid_size, self.grid_size), dtype=str)
        grid[:] = '.'
        grid[self.current_state] = 'A'
        grid[self.goal_state] = 'G'
        print("\n".join([" ".join(row) for row in grid]) + "\n")

def sarsa(env, episodes=500, alpha=0.5, gamma=0.9, epsilon=0.1):
    q_table = np.zeros((env.grid_size, env.grid_size, env.action_space.n))
    steps_per_episode = []
    rewards_per_episode = []

    for episode in range(episodes):
        state = env.reset()
        action = choose_action(state, q_table, epsilon)
        total_reward = 0
        steps = 0

        done = False
        while not done:
            next_state, reward, done, _ = env.step(action)
            next_action = choose_action(next_state, q_table, epsilon)

            # SARSA update rule
            q_table[state][action] += alpha * (
                reward + gamma * q_table[next_state][next_action] - q_table[state][action]
            )

            state = next_state
            action = next_action
            total_reward += reward
            steps += 1

        steps_per_episode.append(steps)
        rewards_per_episode.append(total_reward)
        print(f"SARSA | Episode {episode + 1}: Steps = {steps}, Cumulative Reward = {total_reward}")

    return steps_per_episode, rewards_per_episode

def expected_sarsa(env, episodes=500, alpha=0.5, gamma=0.9, epsilon=0.1):
    q_table = np.zeros((env.grid_size, env.grid_size, env.action_space.n))
    steps_per_episode = []
    rewards_per_episode = []

    for episode in range(episodes):
        state = env.reset()
        action = choose_action(state, q_table, epsilon)
        total_reward = 0
        steps = 0

        done = False
        while not done:
            next_state, reward, done, _ = env.step(action)

            # Expected SARSA update rule
            expected_value = np.mean(q_table[next_state])
            q_table[state][action] += alpha * (
                reward + gamma * expected_value - q_table[state][action]
            )

            action = choose_action(next_state, q_table, epsilon)
            state = next_state
            total_reward += reward
            steps += 1

        steps_per_episode.append(steps)
        rewards_per_episode.append(total_reward)
        print(f"Expected SARSA | Episode {episode + 1}: Steps = {steps}, Cumulative Reward = {total_reward}")

    return steps_per_episode, rewards_per_episode


def bellman_q_learning(env, episodes=500, alpha=0.5, gamma=0.9, epsilon=0.1):
    q_table = np.zeros((env.grid_size, env.grid_size, env.action_space.n))
    steps_per_episode = []
    rewards_per_episode = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0

        done = False
        while not done:
            action = choose_action(state, q_table, epsilon)
            next_state, reward, done, _ = env.step(action)
            next_max_q = np.max(q_table[next_state])
            q_table[state][action] += alpha * (
                reward + gamma * next_max_q - q_table[state][action]
            )
            state = next_state
            total_reward += reward
            steps += 1

        steps_per_episode.append(steps)
        rewards_per_episode.append(total_reward)
        print(f"Bellman Q-Learning | Episode {episode + 1}: Steps = {steps}, Cumulative Reward = {total_reward}")

    return steps_per_episode, rewards_per_episode

def choose_action(state, q_table, epsilon=0.1):
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, 3)
    return np.argmax(q_table[state])

def main():
    env = GridEnv()
    env.render()

    print("How many episodes to run?")
    episodes = int(input())

    # Run SARSA
    print("Running SARSA...")
    sarsa_steps, sarsa_rewards = sarsa(env, episodes=episodes)

    # Run Expected SARSA
    print("Running Expected SARSA...")
    expected_sarsa_steps, expected_sarsa_rewards = expected_sarsa(env, episodes=episodes)

    print("Running Bellman Equation ...")
    belman_steps, belman_rewards = bellman_q_learning(env, episodes=episodes)

    # Display results
    print("\nSARSA:")
    print("Steps : ",sarsa_steps)
    print("Rewards : ", sarsa_rewards)

    print("\nExpected SARSA:")
    print("Steps : ", expected_sarsa_steps)
    print("Rewards : ", expected_sarsa_rewards)

    print("\nBellman Equation:")
    print("Steps : ", belman_steps)
    print("Rewards : ", belman_rewards)

    plt.figure(figsize=(12, 6))
    plt.title(f"Number of episodes = {episodes}")
    plt.subplot(1, 2, 1)
    plt.plot(sarsa_steps, label="SARSA")
    plt.plot(expected_sarsa_steps, label="Expected SARSA")
    plt.plot(belman_steps, label="Bellman Equation")
    plt.title("Steps to Goal")
    plt.xlabel("Episodes")
    plt.ylabel("Steps")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(sarsa_rewards, label="SARSA")
    plt.plot(expected_sarsa_rewards, label="Expected SARSA")
    plt.plot(belman_rewards, label="Bellman Equation")
    plt.title("Cumulative Rewards")
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

import numpy as np
import matplotlib.pyplot as plt
from distributed_q_learner import DistributedQLearner
from policy import EpsGreedyQPolicy
from matrix_game import MatrixGame
import pandas as pd

if __name__ == '__main__':
    nb_episode = 40000

    actions = np.arange(3)
    agent1 = DistributedQLearner(policy=EpsGreedyQPolicy(epsilon=.04), actions=actions)
    agent2 = DistributedQLearner(policy=EpsGreedyQPolicy(epsilon=.04), actions=actions)

    game = MatrixGame()
    for episode in range(nb_episode):
        action1 = agent1.act()
        action2 = agent2.act()

        _, r1, r2 = game.step(action1, action2)

        agent1.observe(reward=r1)
        agent2.observe(reward=r2)

    print("Policy")
    print(agent1.pi)
    print(agent2.pi)

    # moving average reward
    average_rewrad_history1 = pd.Series(agent1.reward_history).rolling(50).mean().tolist()
    average_rewrad_history2 = pd.Series(agent2.reward_history).rolling(50).mean().tolist()

    plt.plot(np.arange(len(average_rewrad_history1)),average_rewrad_history1, label="agent1")
    plt.plot(np.arange(len(average_rewrad_history2)),average_rewrad_history2, label="agent2")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.savefig("result.jpg")
    plt.show()

from tqdm import tqdm
import numpy as np
import ipdb

class MatrixGame():
    def __init__(self):
        self.reward_matrix = self._create_reward_table()

    def step(self, action1, action2):
        r1, r2 = self.reward_matrix[action1][action2]

        return None, r1, r2


    def _create_reward_table(self):
        reward_matrix = [
                            [[11, 11], [-30, -30], [0, 0]],
                            [[-30, -30], [7, 7], [6, 6]], 
                            [[0, 0], [0, 0], [5, 5]], 
                        ]

        return reward_matrix

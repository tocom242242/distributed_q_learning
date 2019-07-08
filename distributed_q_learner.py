import numpy as np

class DistributedQLearner():
    def __init__(self, 
                 policy=None, 
                 gamma=0.99, 
                 ini_state="nonstate", 
                 actions=None, 
                 alpha_decay_rate=None, 
                 epsilon_decay_rate=None):

        self.gamma = gamma
        self.policy = policy
        self.reward_history = []
        self.actions = actions
        self.gamma = gamma
        self.alpha_decay_rate = alpha_decay_rate
        self.epsilon_decay_rate = epsilon_decay_rate
        self.previous_action_id = None
        self.q_values = self._init_q_values()

        self.state = ini_state
        self.pi = {}
        self.pi[ini_state] = self._init_pi_values()
        self.pi_history = []
        self.v = {}

    def _init_pi_values(self):
        pi = np.repeat(1.0/len(self.actions), len(self.actions))

        return pi

    def _init_q_values(self):
        q_values = {}

        return q_values

    def act(self, training=True):
        if training:
            action_id = self.policy.select_action(self.pi[self.state])
            self.previous_action_id = action_id
            action = self.actions[action_id]
            self.previous_action = action
        else:
            action_id = self.policy.select_greedy_action(self.pi)
            action = self.actions[action_id]

        return action

    def observe(self, state="nonstate", reward=None, opponent_action=None, is_learn=True):
        if is_learn:
            self.reward_history.append(reward)
            self.check_new_state(state)
            self.learn(state, reward)

    def learn(self, state, reward):
        max_q = max([self.q_values[(state, action1)] for action1 in self.actions])
        q = reward + self.gamma*max_q
        if q > self.q_values[(state, self.previous_action)]:
            self.q_values[(state, self.previous_action)] = q

        action_argmax_pi = 0
        max_pi_value = 0
        for action1 in self.actions:
            if max_pi_value < self.pi[state][action1]:
                action_argmax_pi = action1
                max_pi_value  = self.pi[state][action1]

        max_q = max([self.q_values[(state, action1)] for action1 in self.actions])
        if self.q_values[(state, action_argmax_pi)] != max_q:
            actions_argmax_qs = [action1 for action1 in self.actions if self.q_values[(state, action1)] == max_q]
            actions_argmax_q = np.random.choice(actions_argmax_qs)
            for a, _ in enumerate(self.pi[state]):
                if actions_argmax_q == a:
                    self.pi[state][a] = 1
                else:
                    self.pi[state][a] = 0

    def check_new_state(self, state):
        for action1 in self.actions:
            if state not in self.pi.keys():
                self.pi[state] = np.array([np.random.random() for _ in self.actions])
            if (state, action1) not in self.q_values.keys():
                self.q_values[(self.state, action1)] = -10000

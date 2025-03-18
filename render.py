# render one episode of the UCBmixLCB agent
import gymnasium as gym
import numpy as np

# Define an agent that uses the UCBmixLCB algorithm to solve the Mars Rover problem.
class UCBmixLCB:
    def __init__(self, n_states, n_actions, horizon, c=2):
        self.n_states = n_states
        self.n_actions = n_actions
        self.H = horizon
        self.c = c
        self.nsa = np.zeros((horizon, n_states, n_actions))
        self.nsas = np.zeros((horizon, n_states, n_actions, n_states))
        self.R = np.zeros((horizon, n_states, n_actions))
        self.Q = np.ones((horizon, n_states, n_actions))
        self.N = np.zeros((horizon, n_states, n_actions))
        self.t = 0

    def ucb_select_action(self, h, state):
        # self.t += 1
        ucb_values = self.Q[h, state] + self.c * np.sqrt(1 / (self.N[h, state] + 1e-5))
        return np.argmax(ucb_values)
    
    def lcb_select_action(self, h, state):
        # self.t += 1
        lcb_values = self.Q[h, state] - self.c * np.sqrt(1 / (self.N[h, state] + 1e-5))
        # if there are still actions with 0 counts, select them
        if np.any(self.N[h, state] == 0):
            return np.argmin(self.N[h, state])
        return np.argmax(lcb_values)
    
    def greedy_select_action(self, h, state):
        return np.argmax(self.Q[h, state])
    
    def update(self, h, state, action, reward, next_state):
        self.nsa[h, state, action] += 1
        self.nsas[h, state, action, next_state] += 1
        self.N[h, state, action] += 1
        self.R[h, state, action] += (reward - self.R[h, state, action]) / self.nsa[h, state, action]

    def Q_value_iteration(self):
        # update Q values by value iteration
        for h in range(self.H - 1, -1, -1):
            for state in range(self.n_states):
                for action in range(self.n_actions):
                    if h == self.H - 1:
                        self.Q[h, state, action] = self.R[h, state, action]
                    else:
                        if self.nsa[h, state, action] == 0:
                            continue
                        self.Q[h, state, action] = self.R[h, state, action] + np.sum(self.nsas[h, state, action] / self.nsa[h, state, action] * np.max(self.Q[h + 1], axis=1))

# Create the environment for FrozenLake-v0
desc=["SFFF", "FHFH", "FFFH", "HFFG"]
holes = [5, 7, 11, 12]

env = gym.make("FrozenLake-v1", desc=desc, map_name="4x4", is_slippery=True, render_mode="human")
H = 20
C = 20
mix_agent = UCBmixLCB(env.observation_space.n, env.action_space.n, H, C)
mix_agent.Q = np.load("mix_agent.npy")

ucb_agent = UCBmixLCB(env.observation_space.n, env.action_space.n, H, C)
ucb_agent.Q = np.load("ucb_agent.npy")

greedy_agent = UCBmixLCB(env.observation_space.n, env.action_space.n, H, C)
greedy_agent.Q = np.load("greedy_agent.npy")

state = env.reset()[0]
for h in range(H):
    print("Step", h)
    action = ucb_agent.greedy_select_action(h, state)
    next_state, _, done, _, _ = env.step(action)
    env.render()
    state = next_state
    if done:
        break
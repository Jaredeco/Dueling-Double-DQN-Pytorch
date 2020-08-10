import os
import numpy as np
import torch as T
from torch import nn
import torch.nn.functional as F
from torch import optim
import gym
from collections import deque
import random

class DuelingDDQN(nn.Module):
    def __init__(self, lr, n_actions, input_dims):
        super(DuelingDDQN, self).__init__()
        self.fc1 = nn.Linear(input_dims, 512)
        self.V = nn.Linear(512, 1)
        self.A = nn.Linear(512, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        flat1 = F.relu(self.fc1(state))
        V = self.V(flat1)
        A = self.A(flat1)
        return V, A

class Agent():
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims,
                 mem_size, batch_size, eps_min=0.01, eps_dec=5e-7,
                 replace=1000):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace
        self.action_space = [i for i in range(self.n_actions)]
        self.learn_step_counter = 0

        self.memory = deque(maxlen=mem_size)

        self.q_eval = DuelingDDQN(self.lr, self.n_actions,
                                   input_dims=self.input_dims,)

        self.q_next = DuelingDDQN(self.lr, self.n_actions,
                                   input_dims=self.input_dims)

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor([observation],dtype=T.float).to(self.q_eval.device)
            _, advantage = self.q_eval.forward(state)
            action = T.argmax(advantage).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def store_transition(self, state, action, reward, state_, done):
        self.memory.append((state, action, reward, state_, done))

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
                        if self.epsilon > self.eps_min else self.eps_min


    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()
        batch = random.sample(self.memory, self.batch_size)
        state, action, reward, new_state, done = [x[0] for x in batch], [x[1] for x in batch], [x[2] for x in batch], [x[3] for x in batch], [x[4] for x in batch]

        states = T.tensor(state).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        dones = T.tensor(done).to(self.q_eval.device)
        actions = T.tensor(action).to(self.q_eval.device)
        states_ = T.tensor(new_state).to(self.q_eval.device)

        indices = np.arange(self.batch_size)

        V_s, A_s = self.q_eval.forward(states)
        V_s_, A_s_ = self.q_next.forward(states_)

        V_s_eval, A_s_eval = self.q_eval.forward(states_)

        q_pred = T.add(V_s,
                        (A_s - A_s.mean(dim=1, keepdim=True)))[indices, actions]
        q_next = T.add(V_s_,
                      (A_s_ - A_s_.mean(dim=1, keepdim=True)))

        q_eval = T.add(V_s_eval, (A_s_eval - A_s_eval.mean(dim=1,keepdim=True)))

        max_actions = T.argmax(q_eval, dim=1)

        q_next[dones] = 0.0
        q_target = rewards + self.gamma*q_next[indices, max_actions]

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()


env = gym.make('LunarLander-v2')
num_games = 250

agent = Agent(gamma=0.99, epsilon=1.0, lr=5e-4,
              input_dims=8, n_actions=4, mem_size=100000, eps_min=0.01,
              batch_size=64, eps_dec=1e-3, replace=100)

scores = []
eps_history = []
n_steps = 0

for i in range(num_games):
    done = False
    observation = env.reset()
    score = 0

    while not done:
        action = agent.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        score += reward
        agent.store_transition(observation, action,
                                reward, observation_, int(done))
        agent.learn()

        observation = observation_

    scores.append(score)
    avg_score = np.mean(scores[max(0, i-100):(i+1)])
    print('episode: ', i,'score %.1f ' % score,
         ' average score %.1f' % avg_score,
        'epsilon %.2f' % agent.epsilon)
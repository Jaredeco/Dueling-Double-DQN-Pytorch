import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random


class DQN(nn.Module):
    def __init__(self, input_dim, n_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 30, 5)
        self.conv3 = nn.Conv2d(30, 40, 5)
        self.fc1 = nn.Linear(40 * 6 * 6, 256)
        self.fc2 = nn.Linear(256, 256)
        self.A = nn.Linear(256, n_actions)
        self.V = nn.Linear(256, 1)
        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters())
        self.cuda()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        A = self.A(x)
        V = self.V(x)
        return V, A


class Agent:
    def __init__(self, input_dims, n_actions):
        self.gamma = 0.99
        self.epsilon = 1
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = 64
        self.eps_min = 0.01
        self.eps_dec = 5e-5
        self.replace_target_net_int = 10
        self.action_space = [i for i in range(self.n_actions)]
        self.learn_step_counter = 0
        self.max_size = 100000
        self.memory = deque(maxlen=self.max_size)
        self.q_eval = DQN(self.input_dims, self.n_actions)
        self.q_next = DQN(self.input_dims, self.n_actions)

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = torch.tensor([observation], dtype=torch.float).cuda()
            _, advantage = self.q_eval(state)
            action = torch.argmax(advantage).item()
        else:
            action = np.random.choice(self.action_space)
        return action

    def store_transition(self, state, action, reward, state_, done):
        self.memory.append((state, action, reward, state_, done))

    def replace_target_network(self):
        if not self.learn_step_counter % self.replace_target_net_int:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()
        batch = random.sample(self.memory, self.batch_size)
        state, action, reward, new_state, done = [x[0] for x in batch], [x[1] for x in batch], [x[2] for x in batch], [
                x[3] for x in batch], [x[4] for x in batch]

        states = torch.tensor(state, dtype=torch.float32).cuda()
        rewards = torch.tensor(reward, dtype=torch.float32).cuda()
        dones = torch.tensor(done, dtype=torch.long).cuda()
        actions = torch.tensor(action, dtype=torch.long).cuda()
        states_ = torch.tensor(new_state, dtype=torch.float32).cuda()
        indices = np.arange(self.batch_size)

        V_s, A_s = self.q_eval(states)
        V_s_, A_s_ = self.q_next(states_)
        V_s_eval, A_s_eval = self.q_eval(states_)
        q_pred = torch.add(V_s, (A_s - A_s.mean(dim=1, keepdim=True)))[indices, actions]
        q_next = torch.add(V_s_, (A_s_ - A_s_.mean(dim=1, keepdim=True)))
        q_eval = torch.add(V_s_eval, (A_s_eval - A_s_eval.mean(dim=1, keepdim=True)))
        max_actions = torch.argmax(q_eval, dim=1)
        q_next[dones] = 0.0

        q_target = rewards + self.gamma * q_next[indices, max_actions]
        loss = self.q_eval.loss(q_pred, q_target).cuda()
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1
        self.decrement_epsilon()

    def save_models(self):
        torch.save(self.q_eval.state_dict(), "q_eval.pth")
        torch.save(self.q_next.state_dict(), "q_next.pth")

    def load_models(self):
        self.q_eval.load_state_dict(torch.load("q_eval.pth"))
        self.q_next.load_state_dict(torch.load("q_next.pth"))



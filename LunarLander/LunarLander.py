import gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.alpha = alpha

    def push(self, state, action, reward, next_state, done):
        max_priority = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)

        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.pos]

        probs = priorities ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        batch = list(zip(*samples))
        return (batch, indices, weights)

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, priority in zip(batch_indices, batch_priorities):
            self.priorities[idx] = priority

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.017):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))

        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / np.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def forward(self, input):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return nn.functional.linear(input, weight, bias)

class RainbowDQN(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(RainbowDQN, self).__init__()

        self.feature = nn.Sequential(
            nn.Linear(num_inputs, 128),
            nn.ReLU()
        )

        # Value Stream
        self.value_stream = nn.Sequential(
            NoisyLinear(128, 128),
            nn.ReLU(),
            NoisyLinear(128, 1)
        )

        # Advantage Stream
        self.advantage_stream = nn.Sequential(
            NoisyLinear(128, 128),
            nn.ReLU(),
            NoisyLinear(128, num_actions)
        )

    def forward(self, x):
        x = self.feature(x)
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        q = value + advantage - advantage.mean()
        return q

    def reset_noise(self):
        for name, module in self.named_children():
            if isinstance(module, NoisyLinear):
                module.reset_noise()
            elif isinstance(module, nn.Sequential):
                for layer in module:
                    if isinstance(layer, NoisyLinear):
                        layer.reset_noise()

ENV_NAME = "LunarLander-v2"
SEED = 42
GAMMA = 0.99
BATCH_SIZE = 64
BUFFER_SIZE = 100000
LEARNING_RATE = 1e-4
TARGET_UPDATE_FREQ = 1000
NUM_FRAMES = 500000
ALPHA = 0.6  # PER hyperparameter
BETA_START = 0.4
BETA_FRAMES = NUM_FRAMES
N_STEPS = 3  # Multi-step returns

env = gym.make(ENV_NAME)
env.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.n

policy_net = RainbowDQN(num_inputs, num_actions)
target_net = RainbowDQN(num_inputs, num_actions)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
replay_buffer = PrioritizedReplayBuffer(BUFFER_SIZE, alpha=ALPHA)

writer = SummaryWriter()

class MultiStepBuffer:
    def __init__(self, n_steps, gamma):
        self.n_steps = n_steps
        self.gamma = gamma
        self.reset()

    def reset(self):
        self.states = deque(maxlen=self.n_steps)
        self.actions = deque(maxlen=self.n_steps)
        self.rewards = deque(maxlen=self.n_steps)
        self.next_states = deque(maxlen=self.n_steps)
        self.dones = deque(maxlen=self.n_steps)

    def append(self, state, action, reward, next_state, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)

    def get(self):
        R = sum([self.rewards[i] * (self.gamma ** i) for i in range(len(self.rewards))])
        return (self.states[0], self.actions[0], R, self.next_states[-1], self.dones[-1])

def update_target():
    target_net.load_state_dict(policy_net.state_dict())

def compute_td_loss(batch_size, beta):
    (transitions, indices, weights) = replay_buffer.sample(batch_size, beta)
    states, actions, rewards, next_states, dones = transitions

    states = torch.FloatTensor(np.array(states))
    next_states = torch.FloatTensor(np.array(next_states))
    actions = torch.LongTensor(actions)
    rewards = torch.FloatTensor(rewards)
    dones = torch.FloatTensor(dones)
    weights = torch.FloatTensor(weights)

    q_values = policy_net(states)
    next_q_values = policy_net(next_states)
    next_q_state_values = target_net(next_states)

    q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    next_actions = next_q_values.max(1)[1]
    next_q_value = next_q_state_values.gather(1, next_actions.unsqueeze(1)).squeeze(1)
    expected_q_value = rewards + (GAMMA ** N_STEPS) * next_q_value * (1 - dones)

    loss = (q_value - expected_q_value.detach()).pow(2) * weights
    prios = loss + 1e-5
    loss = loss.mean()

    optimizer.zero_grad()
    loss.backward()
    replay_buffer.update_priorities(indices, prios.data.cpu().numpy())
    optimizer.step()
    policy_net.reset_noise()
    target_net.reset_noise()

    return loss

beta = BETA_START
frame_idx = 0
episode_rewards = []
state = env.reset()
multi_step_buffer = MultiStepBuffer(N_STEPS, GAMMA)

while frame_idx < NUM_FRAMES:
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    q_values = policy_net(state_tensor)
    action = q_values.max(1)[1].item()

    next_state, reward, done, _ = env.step(action)
    multi_step_buffer.append(state, action, reward, next_state, done)

    if len(multi_step_buffer.rewards) == N_STEPS:
        transition = multi_step_buffer.get()
        replay_buffer.push(*transition)

    state = next_state
    frame_idx += 1

    if done:
        state = env.reset()
        multi_step_buffer.reset()
        episode_rewards.append(sum(multi_step_buffer.rewards))

    if len(replay_buffer.buffer) > BATCH_SIZE:
        beta = min(1.0, beta + frame_idx * (1.0 - BETA_START) / BETA_FRAMES)
        loss = compute_td_loss(BATCH_SIZE, beta)
        writer.add_scalar('loss', loss.item(), frame_idx)

    if frame_idx % TARGET_UPDATE_FREQ == 0:
        update_target()

    if frame_idx % 10000 == 0:
        avg_reward = np.mean(episode_rewards[-100:])
        writer.add_scalar('reward_100', avg_reward, frame_idx)
        print(f"Frame: {frame_idx}, Avg Reward: {avg_reward}")

env.close()
writer.close()

torch.save(policy_net.state_dict(), "rainbow_dqn_lunar_lander.pth")

policy_net = RainbowDQN(num_inputs, num_actions)
policy_net.load_state_dict(torch.load("rainbow_dqn_lunar_lander.pth"))
policy_net.eval()

state = env.reset()
done = False
total_reward = 0

while not done:
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    q_values = policy_net(state_tensor)
    action = q_values.max(1)[1].item()
    next_state, reward, done, _ = env.step(action)
    total_reward += reward
    state = next_state
    env.render()

print(f"Total Reward: {total_reward}")
env.close()

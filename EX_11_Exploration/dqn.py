import random
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym

from exploration import DummyIntrinsicRewardModule, RNDNetwork, ICMNetwork
from helper import episode_reward_plot


class ReplayBuffer(object):
    """A replay buffer as commonly used for off-policy Q-Learning methods."""

    def __init__(self, capacity):
        """Initializes replay buffer with certain capacity."""
        self.buffer: List[Tuple[np.ndarray, int, float, np.ndarray, bool]] = [None] * capacity

        self.capacity: int = capacity
        self.size: int = 0
        self.ptr: int = 0

    def put(self, obs, action, reward, next_obs, terminal):
        """Put a tuple of (obs, action, rewards, next_obs, done) into the replay buffer.
        The max length specified by capacity should never be exceeded. 
        The oldest elements inside the replay buffer should be overwritten first.
        """
        self.buffer[self.ptr] = (obs, action, reward, next_obs, terminal)

        self.size = min(self.size + 1, self.capacity)
        self.ptr = (self.ptr + 1) % self.capacity

    def get(self, batch_size):
        """Gives batch_size samples from the replay buffer.
        Should return 5 lists of, each for every attribute stored (i.e. obs_lst, action_lst, ....)
        """
        return zip(*random.sample(self.buffer[: self.size], batch_size))

    def __len__(self):
        """Returns the number of tuples inside the replay buffer."""
        return self.size


class DQNNetwork(nn.Module):
    """The neural network used to approximate the Q-function. Should output n_actions Q-values per state."""

    def __init__(self, num_obs, num_actions):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(num_obs, 128), nn.ReLU(), nn.Linear(128, num_actions)
        )

    def forward(self, x):
        return self.layers(x)


class DQN:
    """The DQN method."""

    def __init__(
            self,
            env,
            replay_size=2000,
            batch_size=32,
            gamma=0.99,
            sync_after=5,
            lr=0.003,
            verbose=False,
            reward_module=None,
            render=False,
    ):
        if isinstance(env.action_space, gym.spaces.Box):
            raise NotImplementedError("Continuous actions not implemented!")

        # Some variables
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.n

        self.env = env
        self.replay_buffer = ReplayBuffer(replay_size)
        self.sync_after = sync_after
        self.batch_size = batch_size
        self.gamma = gamma
        self.verbose = verbose
        self.render = render

        self.dqn_net = DQNNetwork(self.obs_dim, self.act_dim)
        # Initialize DQN target network, load parameters from DQN network
        self.dqn_target_net = DQNNetwork(self.obs_dim, self.act_dim)
        self.dqn_target_net.load_state_dict(self.dqn_net.state_dict())
        # Set up optimizer, only needed for DQN network
        self.optim_dqn = optim.RMSprop(self.dqn_net.parameters(), lr=lr)

        if reward_module == "RND":
            self.intrinsic_reward_module = RNDNetwork(self.obs_dim, 128)
            self.optim_reward = optim.RMSprop(
                self.intrinsic_reward_module.parameters(), lr=lr,
            )
        elif reward_module == "ICM":
            self.intrinsic_reward_module = ICMNetwork(self.obs_dim, 256, self.act_dim)
            self.optim_reward = optim.RMSprop(
                self.intrinsic_reward_module.parameters(), lr=lr / 50.0
            )
        else:
            # This module only has a 'calculate_reward(...)' method which returns 0.0
            # Used for vanilla DQN
            self.intrinsic_reward_module = DummyIntrinsicRewardModule()

    def learn(self, time_steps):
        episode_length = 0
        intrinsic_rewards = []
        episode_lengths = []
        intrinsic_episode_rewards = []

        obs, _ = self.env.reset()
        min_episode_len = 1337
        for timestep in range(1, time_steps + 1):
            if self.render and timestep % 15 == 0:
                self.env.render()

            epsilon = epsilon_by_timestep(timestep)
            # Note: epsilon is only used for vanilla DQN
            action = self.predict(obs, epsilon)

            next_obs, extrinsic_reward, terminal, _, _ = self.env.step(action)

            intrinsic_reward = self.intrinsic_reward_module.calculate_reward(
                torch.Tensor(obs).unsqueeze(0),
                torch.Tensor(next_obs).unsqueeze(0),
                torch.Tensor([action]).unsqueeze(0),
            ).item()

            reward = extrinsic_reward + intrinsic_reward

            self.replay_buffer.put(obs, action, reward, next_obs, terminal)
            obs = next_obs

            intrinsic_episode_rewards.append(intrinsic_reward)
            episode_length += 1
            if terminal:
                obs, _ = self.env.reset()
                intrinsic_rewards.append(sum(intrinsic_episode_rewards))
                if (
                        self.verbose
                        and min_episode_len > episode_length
                        and episode_length < 200
                ):
                    min_episode_len = episode_length
                    print(f"[t={timestep}]: Solved after {min_episode_len}!")
                episode_lengths.append(episode_length)
                episode_length = 0
                intrinsic_episode_rewards = []

            if len(self.replay_buffer) > self.batch_size:
                obs_, actions, rewards, next_obs_, terminals = self.replay_buffer.get(
                    self.batch_size
                )
                obs_ = torch.stack([torch.Tensor(ob) for ob in obs_])
                next_obs_ = torch.stack(
                    [torch.Tensor(next_ob) for next_ob in next_obs_]
                )
                rewards = torch.Tensor(rewards)
                terminals = torch.Tensor(terminals)
                # Has to be torch.LongTensor in order to being able to use as index for torch.gather()
                actions = torch.LongTensor(actions)

                dqn_loss = self.compute_msbe_loss(
                    obs_, actions, rewards, next_obs_, terminals
                )
                self.optim_dqn.zero_grad()
                dqn_loss.backward()
                self.optim_dqn.step()

                if not isinstance(
                        self.intrinsic_reward_module, DummyIntrinsicRewardModule
                ):
                    intrinsic_loss = self.intrinsic_reward_module.calculate_loss(
                        obs_, next_obs_, actions
                    )
                    self.optim_reward.zero_grad()
                    intrinsic_loss.backward()
                    self.optim_reward.step()

            if timestep % self.sync_after == 0:
                self.dqn_target_net.load_state_dict(self.dqn_net.state_dict())

            if timestep % 1000 == 0 and len(episode_lengths) >= 7:
                episode_reward_plot(
                    episode_lengths,
                    intrinsic_rewards,
                    timestep,
                    window_size=7,
                    step_size=1,
                )

    def predict(self, state, epsilon=0.0):
        # We turn off epsilon greedy when using an intrinsic reward module
        if random.random() > epsilon or isinstance(
                self.intrinsic_reward_module, DummyIntrinsicRewardModule
        ):
            state = torch.FloatTensor(state).unsqueeze(0)
            q_value = self.dqn_net.forward(state)
            action = q_value.argmax().item()
        else:
            action = random.randrange(self.act_dim)
        return action

    def compute_msbe_loss(self, obs, actions, rewards, next_obs, done):
        q_values = self.dqn_net(obs)
        next_q_values = self.dqn_target_net(next_obs)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = next_q_values.max(1)[0]
        expected_q_values = rewards + self.gamma * next_q_values * (1.0 - done)
        dqn_loss = F.mse_loss(q_values, expected_q_values)

        return dqn_loss


def epsilon_by_timestep(
        timestep, epsilon_start=1.0, epsilon_final=0.01, frames_decay=10000
):
    """Linearly decays epsilon from epsilon_start to epsilon_final in frames_decay timesteps."""
    return max(
        epsilon_final,
        epsilon_start - (timestep / frames_decay) * (epsilon_start - epsilon_final),
    )


if __name__ == "__main__":
    # Switch this to 'True' for realtime rendering during training
    # Note: This might slow down training a bit
    render = False

    from env import MountainCarEnvCustomized

    env = MountainCarEnvCustomized(render_mode="human" if render else None)

    # Uncomment this to use the default MountainCar environment
    # Note: You should set `render = False`, because it limits training to 30 environment steps per second
    # env = gym.make('MountainCar-v0')

    # dqn = DQN(env, verbose=True, render=render)
    # dqn.learn(50000)

    # DQN + ICM
    # dqn = DQN(env, verbose=True, reward_module="ICM", render=render)
    # dqn.learn(50000)

    # DQN + RND
    dqn = DQN(env, verbose=True, reward_module="RND", render=render)
    dqn.learn(50000)

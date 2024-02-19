import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStack
import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from helper import visualize_agent, episode_reward_plot  # , RandomAgent


class ReplayBuffer(object):
    """A replay buffer as commonly used for off-policy Q-Learning methods."""

    def __init__(self, capacity):
        """Initializes replay buffer with certain capacity."""
        self.buffer = [None] * capacity

        self.capacity = capacity
        self.size = 0
        self.ptr = 0

    def put(self, obs, action, reward, next_obs, done):
        """Put a tuple of (obs, action, rewards, next_obs, done) into the replay buffer.
        The max length specified by capacity should never be exceeded.
        The oldest elements inside the replay buffer should be overwritten first.
        """
        if len(obs.shape) == 2:
            obs = np.expand_dims(obs, 0)
            next_obs = np.expand_dims(next_obs, 0)

        self.buffer[self.ptr] = (obs, action, reward, next_obs, done)
        self.size = min(self.size + 1, self.capacity)
        self.ptr = (self.ptr + 1) % self.capacity

    def get(self, batch_size):
        """Gives batch_size samples from the replay buffer.
        Should return 5 lists of, each for every attribute stored (i.e. obs_lst, action_lst, ....)
        """
        return zip(*random.sample(self.buffer[:self.size], batch_size))

    def __len__(self):
        """Returns the number of tuples inside the replay buffer."""
        return self.size


class DQNNetwork(nn.Module):
    """The neural network used to approximate the Q-function. Should output n_actions Q-values per state."""

    def __init__(self, num_obs, num_actions):
        super(DQNNetwork, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(num_obs, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, x):
        return self.layers(x)


class DQNNetworkVisual(nn.Module):
    """The neural network used to approximate the Q-function. Should output n_actions Q-values per state."""

    def __init__(self, c_in, num_actions):
        super(DQNNetworkVisual, self).__init__()
        # TODO: Implement here
        self.cnn1 = nn.Conv2d(c_in, 32, [8, 8], 4)
        self.cnn2 = nn.Conv2d(32, 64, [4, 4], 2)
        self.cnn3 = nn.Conv2d(64, 64, [3, 3], 1)
        self.ff1 = nn.Linear(3136, 512)
        self.ff2 = nn.Linear(512, 256)
        self.ff3 = nn.Linear(256, num_actions)

    def forward(self, x):
        import torch.nn.functional as F
        x = F.relu(self.cnn1(x))
        x = F.relu(self.cnn2(x))
        x = F.relu(self.cnn3(x))
        x = F.relu(self.ff1(x.view(x.size()[0], -1)))
        x = F.relu(self.ff2(x))
        x = self.ff3(x)
        return x


class DoubleDQN():
    """The DQN method."""

    def __init__(self, env, replay_size=100000, batch_size=32,
                 gamma=0.99, sync_after=1000, lr=0.00001, eps_decay=100000,
                 learning_start=100000, train_freq=4, frame_stack=4,
                 save_freq=1000, resume=False, vis_while_train=50000):
        """ Initializes the DQN method.

        Parameters
        ----------
        env: gym.Environment
            The gym environment the agent should learn in.
        replay_size: int
            The size of the replay buffer.
        batch_size: int
            The number of replay buffer entries an optimization step should be performed on.
        gamma: float
            The discount factor.
        sync_after: int
            Timesteps after which the target network should be synchronized with the main network.
        lr: float
            Adam optimizer learning rate.
        """

        if isinstance(env.action_space, gym.spaces.Box):
            raise NotImplementedError('Continuous actions not implemented!')
        self.env = env
        self.act_dim = env.action_space.n
        self.replay_buffer = ReplayBuffer(replay_size)
        self.sync_after = sync_after
        self.batch_size = batch_size
        self.gamma = gamma
        self.save_freq = save_freq
        self.vis_while_train = vis_while_train
        self.learning_starts = learning_start
        self.train_freq = train_freq
        self.frame_stack = frame_stack
        self.eps_decay = eps_decay
        if vis_while_train:
            self.env_test = create_atari_env(render=True)

        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda:0'

        if len(env.observation_space.shape) >= 2:
            network = DQNNetworkVisual
            in_param = env.observation_space.shape[-1] * self.frame_stack
        else:
            network = DQNNetwork
            in_param = env.observation_space.shape[0]

        self.dqn_net = network(in_param, self.act_dim).to(self.device)

        if resume:
            self.dqn_net.load_state_dict(torch.load('agent.pth'))

        self.dqn_target_net = network(in_param, self.act_dim).to(self.device)
        self.dqn_target_net.load_state_dict(self.dqn_net.state_dict())

        # Set up optimizer, only needed for DQN network
        # self.optim_dqn = optim.Adam(self.dqn_net.parameters(), lr=lr)
        self.optim_dqn = optim.RMSprop(self.dqn_net.parameters(), lr=lr)

    def learn(self, timesteps):
        """Train the agent for timesteps steps inside self.env.
        After every step taken inside the environment observations, rewards, etc. have to be saved inside the replay buffer.
        If there are enough elements already inside the replay buffer (>batch_size), compute MSBE loss and optimize DQN network.

        Parameters
        ----------
        timesteps: int
            Number of timesteps to optimize the DQN network.
        """
        all_rewards = []
        episode_rewards = []

        obs, _ = self.env.reset()
        if len(obs.shape) == 4:
            obs = np.squeeze(obs, axis=3)
        elif len(obs.shape) != 1:
            obs = np.transpose(obs, (2, 0, 1))

        for timestep in tqdm(range(1, timesteps + 1)):
            epsilon = epsilon_by_timestep(timestep, frames_decay=self.eps_decay)
            action = self.predict(obs, epsilon)
            next_obs, reward, done, _, _ = env.step(action)
            # Change from HWC to CHW format
            if len(next_obs.shape) == 4:
                next_obs = np.squeeze(next_obs, axis=3)
            elif len(next_obs.shape) != 1:
                # Change from HWC to CHW format
                next_obs = np.transpose(next_obs, (2, 0, 1))

            self.replay_buffer.put(obs, action, reward, next_obs, done)
            obs = next_obs

            episode_rewards.append(reward)

            if done:
                obs, _ = env.reset()
                if len(obs.shape) == 4:
                    obs = np.squeeze(obs, axis=3)
                elif len(obs.shape) != 1:
                    # Change from HWC to CHW format
                    obs = np.transpose(obs, (2, 0, 1))

                all_rewards.append(sum(episode_rewards))
                episode_rewards = []

            if self.batch_size < len(self.replay_buffer) and len(
                    self.replay_buffer) >= self.learning_starts and timestep % self.train_freq == 0:

                loss, max_error = self.compute_msbe_loss()

                self.optim_dqn.zero_grad()
                loss.backward()
                self.optim_dqn.step()

            if timestep % self.sync_after == 0:
                self.dqn_target_net.load_state_dict(self.dqn_net.state_dict())

            if timestep % self.save_freq == 0:
                torch.save(self.dqn_net.state_dict(), 'agent.pth')
                plt.savefig("rewards.pdf")

            if timestep > self.learning_starts:
                if timestep % 1000 == 0:
                    num_rewards = len(all_rewards)
                    sub_sample = max(1, int(100 / num_rewards))
                    episode_reward_plot(all_rewards[::sub_sample], timestep, window_size=7, step_size=1)
                if self.vis_while_train and timestep % self.vis_while_train == 0:
                    print("Visualizing...")
                    visualize_agent(self.env_test, self, num_timesteps=100)

    def predict(self, state, epsilon=0.001):
        """Predict the best action based on state. With probability epsilon take random action

        Returns
        -------
        int
            The action to be taken.
        """

        if random.random() > epsilon:
            # The unsqueezing is to get our data into batch form
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_value = self.dqn_net.forward(state)
            action = q_value.squeeze().cpu().argmax().item()
        else:
            action = random.randrange(self.act_dim)
        return action

    def compute_msbe_loss(self):
        """Compute the MSBE loss between self.dqn_net predictions and expected Q-values.

        Returns
        -------
        float
            The MSE between Q-value prediction and expected Q-values.
        """
        obs, actions, rewards, next_obs, done = self.replay_buffer.get(self.batch_size)

        obs = torch.stack([torch.Tensor(ob) for ob in obs]).to(self.device)
        next_obs = torch.stack([torch.Tensor(next_ob) for next_ob in next_obs]).to(self.device)
        rewards = torch.Tensor(rewards)
        # Will have 1.0 if done and 0.0 if not done
        done = torch.Tensor(done)

        q_values = self.dqn_net(obs).cpu()
        with torch.no_grad():
            next_q_values = self.dqn_net(next_obs).cpu()
            next_q_values_target = self.dqn_target_net(next_obs).cpu()
            max_actions = np.argmax(next_q_values, axis=1)

        actions = torch.LongTensor(actions)
        # Select Q-values of actions actually taken
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        #########################################
        ########This block has changed###########
        #########################################
        # Calculate max over next Q-values
        # The following line is the Q-value target used in the original DQN.
        # next_q_values = next_q_values.max(1)[0]
        # In DoubleDQN, we the changed target:
        next_q_values = next_q_values_target.gather(1, max_actions.unsqueeze(1)).squeeze(1)
        #########################################

        expected_q_values = rewards + self.gamma * next_q_values * (1.0 - done)

        loss = F.mse_loss(q_values, expected_q_values)
        abs_errors = torch.abs(q_values - expected_q_values)
        return loss, torch.max(abs_errors)


def epsilon_by_timestep(timestep, epsilon_start=1.0, epsilon_final=0.01, frames_decay=1000000):
    """Linearily decays epsilon from epsilon_start to epsilon_final in frames_decay timesteps"""
    return max(epsilon_final, epsilon_start - (timestep / frames_decay) * (epsilon_start - epsilon_final))


def create_atari_env(env_id="PongNoFrameskip-v4", render=False):
    config = dict(
        mode=0,
        difficulty=0,
        repeat_action_probability=0.0,
        obs_type='grayscale',
    )
    if render:
        config['render_mode'] = 'human'
    env = gym.make(env_id, **config)
    env = AtariPreprocessing(env, scale_obs=True, grayscale_newaxis=True, terminal_on_life_loss=True)
    env = FrameStack(env, 4)
    return env


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Eat. Pray. Double DQN.")
    parser.add_argument("--test", action='store_true', help="Load the agent from 'agent.pth' and visualize it.")
    args = parser.parse_args()

    if args.test:
        env = create_atari_env(render=True)
        dqn = DoubleDQN(env, resume=True)
        visualize_agent(env, dqn)
        exit()

    # Plot epsilon rate over time
    # plt.plot([epsilon_by_timestep(i) for i in range(50000)])
    # plt.show()

    #########################################
    ###############Cartpole##################
    #########################################
    # Train the DQN agent on CartPole-v1 as in the last exercise

    """
    env_id = "CartPole-v1"
    env = gym.make(env_id, render_mode="rgb_array")
    dqn = DoubleDQN(
        env,
        learning_start=0,
        train_freq=1,
        eps_decay=int(10000*0.5),
        vis_while_train=False,
        sync_after=100,
        lr=0.001
    )
    dqn.learn(50000)
    visualize_agent(env, dqn)
    """

    #########################################
    ################Atari####################
    #########################################
    # Now on to Atari

    # how it looks like with a random agent
    # env_test = create_atari_env(render=True)
    # rnd = RandomAgent(env_test)
    # visualize_agent(env_test, rnd, num_timesteps=1000)
    # env_test.close()

    time_steps = 10000000
    env = create_atari_env()
    dqn = DoubleDQN(env, eps_decay=int(time_steps * 0.1))
    dqn.learn(time_steps)

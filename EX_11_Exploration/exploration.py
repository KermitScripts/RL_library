import torch
from torch import nn as nn
from torch.nn import functional as F


class IntrinsicRewardModule(nn.Module):
    """The base class for an intrinsic reward method."""
    def calculate_reward(self, obs, next_obs, actions):
        return NotImplemented

    def calculate_loss(self, obs, next_obs, actions):
        return NotImplemented


class DummyIntrinsicRewardModule(IntrinsicRewardModule):
    """Used as a dummy for vanilla DQN."""
    def calculate_reward(self, obs, next_obs, actions):
        return torch.Tensor([0.0]).unsqueeze(0)


class RNDNetwork(IntrinsicRewardModule):
    """Implementation of Random Network Distillation (RND)"""
    def __init__(self, num_obs, num_out, alpha=0.2):
        super().__init__()

        self.target = nn.Sequential(
            nn.Linear(num_obs, 128), nn.ReLU(), nn.Linear(128, num_out), nn.ReLU(),
        )
        self.predictor = nn.Sequential(
            nn.Linear(num_obs, 128), nn.ReLU(), nn.Linear(128, num_out), nn.ReLU(),
        )
        self.alpha = alpha

    def calculate_loss(self, obs, next_obs, actions):
        rnd_target, rnd_pred = self.target(next_obs).detach(), self.predictor(next_obs)
        rnd_loss = F.mse_loss(rnd_target, rnd_pred)
        return rnd_loss

    def calculate_reward(self, obs, next_obs, actions):
        rnd_target, rnd_pred = self.target(next_obs).detach(), self.predictor(next_obs)
        reward = torch.abs(rnd_target - rnd_pred).sum()
        reward = torch.clamp(self.alpha * reward, 0.0, 1.0)
        return reward


class ICMNetwork(IntrinsicRewardModule):
    """Implementation of Intrinsic Curiosity Module (ICM)"""
    def __init__(self, num_obs, num_feature, num_act, alpha=10.0, beta=0.5):
        super().__init__()

        self.feature = nn.Sequential(nn.Linear(num_obs, num_feature), nn.ReLU(),)

        self.inverse_dynamics = nn.Sequential(
            nn.Linear(num_feature * 2, num_act), nn.ReLU(), nn.Softmax()
        )

        self.forward_dynamics = nn.Sequential(
            nn.Linear(num_feature + num_act, num_feature), nn.ReLU(),
        )

        self.alpha = alpha
        self.beta = beta
        self.num_actions = num_act
        self.num_feat = num_feature

    def calculate_loss(self, obs, next_obs, actions):
        actions = actions.unsqueeze(1)

        # Inverse dynamics loss
        obs_feat = self.feature(obs)
        next_obs_feat = self.feature(next_obs)
        features_concat = torch.cat((obs_feat, next_obs_feat), 1)
        actions_pred = self.inverse_dynamics(features_concat)
        actions_target = torch.zeros_like(actions_pred)
        for i, a in enumerate(actions):
            actions_target[i, int(a)] = 1.0
        inverse_dynamics_loss = F.cross_entropy(actions_pred, actions_target)

        # Forward dynamics loss
        features_concat = torch.cat((obs_feat, actions_pred), 1)
        next_obs_feat_pred = self.forward_dynamics(features_concat)
        forward_dynamics_loss = 0.5 * torch.pow(
            F.mse_loss(next_obs_feat_pred, next_obs_feat), 2
        )

        # Add up
        loss = (
            1.0 - self.beta
        ) * inverse_dynamics_loss + self.beta * forward_dynamics_loss
        return loss

    def calculate_reward(self, obs, next_obs, actions):
        actions_one_hot = torch.zeros((obs.size()[0], self.num_actions))
        for i, a in enumerate(actions):
            actions_one_hot[i, int(a)] = 1.0

        obs_feat = self.feature(obs)
        next_obs_feat = self.feature(next_obs)

        features_concat = torch.cat((obs_feat, actions_one_hot), 1)
        next_obs_feat_pred = self.forward_dynamics(features_concat)

        reward = self.alpha * torch.abs(next_obs_feat - next_obs_feat_pred).mean()
        return reward

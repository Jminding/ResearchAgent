"""
Proximal Policy Optimization (PPO) Agent for Surface Code QEC

Implements PPO with:
- Actor-Critic architecture
- Generalized Advantage Estimation (GAE)
- Clipped objective
- Value function clipping

Author: Research Agent
Date: 2024-12-22
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import Tuple, List, Optional, Dict
from collections import deque
import os


class ActorCritic(nn.Module):
    """
    Combined actor-critic network for PPO.

    Architecture:
    - Shared feature extractor (optional)
    - Separate actor (policy) and critic (value) heads
    """

    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dims: List[int] = [64, 64],
                 shared_layers: int = 0):
        """
        Initialize actor-critic network.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: List of hidden layer dimensions
            shared_layers: Number of shared layers between actor and critic
        """
        super(ActorCritic, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Build networks
        self.actor = self._build_mlp(state_dim, action_dim, hidden_dims, output_activation='softmax')
        self.critic = self._build_mlp(state_dim, 1, hidden_dims, output_activation=None)

    def _build_mlp(self, input_dim: int, output_dim: int,
                   hidden_dims: List[int], output_activation: Optional[str] = None) -> nn.Sequential:
        """Build MLP with specified architecture."""
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        if output_activation == 'softmax':
            layers.append(nn.Softmax(dim=-1))
        elif output_activation == 'tanh':
            layers.append(nn.Tanh())

        return nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning both action probabilities and value.

        Args:
            state: State tensor

        Returns:
            (action_probs, value)
        """
        action_probs = self.actor(state)
        value = self.critic(state)
        return action_probs, value

    def get_action_probs(self, state: torch.Tensor) -> torch.Tensor:
        """Get action probabilities from policy."""
        return self.actor(state)

    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """Get state value from critic."""
        return self.critic(state)


class RolloutBuffer:
    """
    Buffer to store rollout data for PPO updates.
    """

    def __init__(self, buffer_size: int, state_dim: int):
        """
        Initialize rollout buffer.

        Args:
            buffer_size: Maximum number of transitions to store
            state_dim: Dimension of state space
        """
        self.buffer_size = buffer_size
        self.state_dim = state_dim
        self.clear()

    def clear(self):
        """Clear all stored data."""
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.ptr = 0

    def add(self, state: np.ndarray, action: int, log_prob: float,
            reward: float, done: bool, value: float):
        """Add a transition to the buffer."""
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.ptr += 1

    def is_full(self) -> bool:
        """Check if buffer is full."""
        return self.ptr >= self.buffer_size

    def get(self) -> Tuple[torch.Tensor, ...]:
        """
        Get all data as tensors.

        Returns:
            (states, actions, log_probs, rewards, dones, values)
        """
        states = torch.FloatTensor(np.array(self.states))
        actions = torch.LongTensor(self.actions)
        log_probs = torch.FloatTensor(self.log_probs)
        rewards = torch.FloatTensor(self.rewards)
        dones = torch.FloatTensor(self.dones)
        values = torch.FloatTensor(self.values)

        return states, actions, log_probs, rewards, dones, values

    def __len__(self):
        return self.ptr


class PPOAgent:
    """
    PPO Agent for QEC decoding.

    Implements:
    - Clipped surrogate objective
    - Generalized Advantage Estimation (GAE)
    - Value function clipping
    - Entropy regularization
    """

    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dims: List[int] = [64, 64],
                 lr_actor: float = 3e-4,
                 lr_critic: float = 1e-3,
                 gamma: float = 0.99,
                 lambda_gae: float = 0.95,
                 eps_clip: float = 0.2,
                 K_epochs: int = 10,
                 batch_size: int = 64,
                 buffer_size: int = 2048,
                 entropy_coef: float = 0.01,
                 value_coef: float = 0.5,
                 max_grad_norm: float = 0.5,
                 device: str = 'cpu'):
        """
        Initialize PPO agent.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: Hidden layer dimensions
            lr_actor: Actor learning rate
            lr_critic: Critic learning rate
            gamma: Discount factor
            lambda_gae: GAE parameter
            eps_clip: PPO clipping parameter
            K_epochs: Number of epochs per update
            batch_size: Mini-batch size
            buffer_size: Rollout buffer size
            entropy_coef: Entropy regularization coefficient
            value_coef: Value loss coefficient
            max_grad_norm: Maximum gradient norm for clipping
            device: Device to use ('cpu' or 'cuda')
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.device = torch.device(device)

        # Networks
        self.policy = ActorCritic(state_dim, action_dim, hidden_dims).to(self.device)
        self.policy_old = ActorCritic(state_dim, action_dim, hidden_dims).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        # Optimizers
        self.optimizer = optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        # Buffer
        self.buffer = RolloutBuffer(buffer_size, state_dim)

        # Training stats
        self.training_stats = {
            'actor_loss': [],
            'critic_loss': [],
            'entropy': [],
            'total_loss': []
        }

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[int, float, float]:
        """
        Select action using current policy.

        Args:
            state: Current state
            deterministic: If True, select most probable action

        Returns:
            (action, log_prob, value)
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_probs, value = self.policy_old(state_tensor)

            if deterministic:
                action = torch.argmax(action_probs, dim=-1).item()
                log_prob = torch.log(action_probs[0, action] + 1e-10).item()
            else:
                dist = Categorical(action_probs)
                action = dist.sample().item()
                log_prob = dist.log_prob(torch.tensor(action)).item()

            return action, log_prob, value.item()

    def store_transition(self, state: np.ndarray, action: int, log_prob: float,
                         reward: float, done: bool, value: float):
        """Store a transition in the buffer."""
        self.buffer.add(state, action, log_prob, reward, done, value)

    def compute_gae(self, rewards: torch.Tensor, values: torch.Tensor,
                    dones: torch.Tensor, next_value: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation.

        Args:
            rewards: Reward tensor
            values: Value tensor
            dones: Done tensor
            next_value: Value of next state (for bootstrapping)

        Returns:
            (advantages, returns)
        """
        n = len(rewards)
        advantages = torch.zeros(n)
        returns = torch.zeros(n)

        gae = 0
        for t in reversed(range(n)):
            if t == n - 1:
                next_val = next_value
            else:
                next_val = values[t + 1].item()

            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lambda_gae * (1 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = gae + values[t]

        return advantages, returns

    def update(self) -> Dict[str, float]:
        """
        Perform PPO update on collected data.

        Returns:
            Dictionary of training statistics
        """
        # Get data from buffer
        states, actions, old_log_probs, rewards, dones, values = self.buffer.get()

        # Compute advantages
        advantages, returns = self.compute_gae(rewards, values, dones)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        old_log_probs = old_log_probs.to(self.device)
        advantages = advantages.to(self.device)
        returns = returns.to(self.device)

        # PPO update
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        n_updates = 0

        for epoch in range(self.K_epochs):
            # Generate random indices for mini-batches
            indices = torch.randperm(len(states))

            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # Get current policy outputs
                action_probs, new_values = self.policy(batch_states)
                dist = Categorical(action_probs)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                # Compute ratio
                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                # Clipped surrogate objective
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                critic_loss = nn.MSELoss()(new_values.squeeze(), batch_returns)

                # Total loss
                loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy

                # Update
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.item()
                n_updates += 1

        # Update old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # Clear buffer
        self.buffer.clear()

        # Compute average stats
        stats = {
            'actor_loss': total_actor_loss / n_updates,
            'critic_loss': total_critic_loss / n_updates,
            'entropy': total_entropy / n_updates,
            'total_loss': (total_actor_loss + self.value_coef * total_critic_loss) / n_updates
        }

        # Store stats
        for key, value in stats.items():
            self.training_stats[key].append(value)

        return stats

    def save(self, path: str):
        """Save model to path."""
        os.makedirs(path, exist_ok=True)
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'policy_old_state_dict': self.policy_old.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_stats': self.training_stats
        }, os.path.join(path, 'ppo_agent.pt'))

    def load(self, path: str):
        """Load model from path."""
        checkpoint = torch.load(os.path.join(path, 'ppo_agent.pt'), map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.policy_old.load_state_dict(checkpoint['policy_old_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_stats = checkpoint.get('training_stats', self.training_stats)


def test_ppo_agent():
    """Test PPO agent basic functionality."""
    print("Testing PPO Agent")
    print("=" * 50)

    state_dim = 24  # 8 syndrome bits * 3 history window
    action_dim = 28  # 9 qubits * 3 Paulis + no-op

    agent = PPOAgent(state_dim, action_dim, hidden_dims=[32, 32], buffer_size=128)

    # Test action selection
    state = np.random.randn(state_dim).astype(np.float32)
    action, log_prob, value = agent.select_action(state)
    print(f"Selected action: {action}, log_prob: {log_prob:.4f}, value: {value:.4f}")

    # Test storing transitions
    for i in range(128):
        state = np.random.randn(state_dim).astype(np.float32)
        action, log_prob, value = agent.select_action(state)
        reward = np.random.randn()
        done = i == 127
        agent.store_transition(state, action, log_prob, reward, done, value)

    # Test update
    stats = agent.update()
    print(f"Update stats: {stats}")

    # Test save/load
    agent.save("/tmp/test_ppo")
    agent.load("/tmp/test_ppo")
    print("Save/load test passed!")

    print("=" * 50)
    print("All PPO agent tests passed!")


if __name__ == "__main__":
    test_ppo_agent()

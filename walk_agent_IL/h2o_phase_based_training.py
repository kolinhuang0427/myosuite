"""
H2O Phase-based Motion Tracking Policy Training
Implements the reinforcement learning component for training policies to track retargeted humanoid motions
Based on: "H2O: Learning Human-to-Humanoid Real-Time Whole-Body Teleoperation" by Tairan He et al. (2024)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import pickle
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import sys
import os
from collections import deque
import random
from dataclasses import dataclass

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

@dataclass
class TrainingConfig:
    """Configuration for H2O training"""
    # PPO hyperparameters
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coeff: float = 0.01
    value_coeff: float = 0.5
    max_grad_norm: float = 0.5
    
    # Training parameters
    batch_size: int = 2048
    mini_batch_size: int = 256
    update_epochs: int = 10
    total_timesteps: int = 2000000
    
    # H2O specific parameters
    history_length: int = 5  # 5-step history as in paper
    termination_curriculum: bool = True
    initial_termination_threshold: float = 5.0  # meters - more lenient initially
    final_termination_threshold: float = 0.3    # meters
    reference_state_init: bool = True
    
    # Environment parameters
    max_episode_length: int = 1000
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


class ActorNetwork(nn.Module):
    """Actor network for phase-based motion tracking"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [256, 256]):
        super().__init__()
        
        self.layers = nn.ModuleList()
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        
        self.mean_head = nn.Linear(prev_dim, action_dim)
        self.log_std_head = nn.Linear(prev_dim, action_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=1.0)
            nn.init.zeros_(module.bias)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = state
        for layer in self.layers:
            x = F.tanh(layer(x))
        
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, -5, 2)  # Stabilize training
        
        return mean, log_std
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(state)
        
        if deterministic:
            return mean, torch.zeros_like(mean)
        
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        
        return action, log_prob


class CriticNetwork(nn.Module):
    """Critic network with privileged information for asymmetric training"""
    
    def __init__(self, state_dim: int, privileged_dim: int = 0, hidden_dims: List[int] = [256, 256]):
        super().__init__()
        
        input_dim = state_dim + privileged_dim
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        
        self.value_head = nn.Linear(prev_dim, 1)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=1.0)
            nn.init.zeros_(module.bias)
    
    def forward(self, state: torch.Tensor, privileged_info: Optional[torch.Tensor] = None) -> torch.Tensor:
        if privileged_info is not None:
            x = torch.cat([state, privileged_info], dim=-1)
        else:
            x = state
        
        for layer in self.layers:
            x = F.tanh(layer(x))
        
        value = self.value_head(x)
        return value


class H2OMotionTracker:
    """H2O Motion Tracking Environment Wrapper"""
    
    def __init__(self, env, motion_data: Dict, config: TrainingConfig):
        self.env = env
        self.config = config
        self.motion_data = motion_data
        
        # Extract motion information
        self.joint_names = motion_data.get('joint_names', [])
        self.robot_trajectory = motion_data['robot']  # Shape: (T, n_joints)
        self.robot_velocities = motion_data.get('robot_vel', np.zeros_like(self.robot_trajectory))
        self.time_steps = motion_data['time']
        self.dt = motion_data.get('dt', 0.002)
        self.horizon = len(self.time_steps)
        
        # State dimensions
        self.joint_dim = len(self.joint_names)
        self.history_length = config.history_length
        
        # Proprioception: q_{t-4:t}, qvel_{t-4:t}, omega_root_{t-4:t}, g_{t-4:t}, a_{t-5:t-1}
        self.proprioception_dim = (
            self.joint_dim * self.history_length +      # joint positions
            self.joint_dim * self.history_length +      # joint velocities  
            3 * self.history_length +                   # root angular velocity
            3 * self.history_length +                   # projected gravity
            self.joint_dim * self.history_length        # previous actions
        )
        
        # State: proprioception + phase variable
        self.state_dim = self.proprioception_dim + 1
        
        # Privileged information for critic (global positions, root linear velocity)
        self.privileged_dim = 3 + 3  # global position + root linear velocity
        
        # Action space (joint positions)
        self.action_dim = self.joint_dim
        
        # Initialize history buffers
        self.reset_history()
        
        # Current motion tracking
        self.current_motion_idx = 0
        self.current_phase = 0.0
        self.episode_length = 0
        
        # Termination curriculum
        self.termination_threshold = config.initial_termination_threshold
        
    def reset_history(self):
        """Reset history buffers"""
        self.joint_pos_history = deque(maxlen=self.history_length)
        self.joint_vel_history = deque(maxlen=self.history_length)
        self.root_angvel_history = deque(maxlen=self.history_length)
        self.gravity_history = deque(maxlen=self.history_length)
        self.action_history = deque(maxlen=self.history_length)
        
        # Initialize with zeros
        for _ in range(self.history_length):
            self.joint_pos_history.append(np.zeros(self.joint_dim))
            self.joint_vel_history.append(np.zeros(self.joint_dim))
            self.root_angvel_history.append(np.zeros(3))
            self.gravity_history.append(np.array([0, 0, -9.81]))
            self.action_history.append(np.zeros(self.joint_dim))
    
    def reset(self, random_init: bool = True) -> np.ndarray:
        """Reset environment with optional Reference State Initialization (RSI)"""
        self.episode_length = 0
        
        if random_init and self.config.reference_state_init:
            # Sample random phase for RSI
            self.current_phase = np.random.uniform(0.0, 1.0)
            self.current_motion_idx = int(self.current_phase * (self.horizon - 1))
        else:
            self.current_phase = 0.0
            self.current_motion_idx = 0
        
        # Initialize robot state from reference motion
        target_joint_pos = self.robot_trajectory[self.current_motion_idx]
        target_joint_vel = self.robot_velocities[self.current_motion_idx]
        
        # Reset environment
        obs = self.env.reset()
        
        # Set joint positions to reference if possible
        if hasattr(self.env.sim, 'data'):
            # Map joint positions to simulation
            for i, joint_name in enumerate(self.joint_names):
                try:
                    joint_id = self.env.sim.model.joint_name2id(joint_name)
                    if joint_id < len(self.env.sim.data.qpos):
                        self.env.sim.data.qpos[joint_id] = target_joint_pos[i]
                        self.env.sim.data.qvel[joint_id] = target_joint_vel[i]
                except Exception:
                    # Joint not found in simulation, use default
                    pass
            self.env.sim.forward()
        
        # Reset history
        self.reset_history()
        
        # Get initial state
        return self.get_state()
    
    def get_state(self) -> np.ndarray:
        """Get current state (proprioception + phase variable)"""
        # Get current robot state
        if hasattr(self.env, 'sim') and self.env.sim is not None:
            current_qpos = np.zeros(self.joint_dim)
            current_qvel = np.zeros(self.joint_dim)
            
            # Extract joint positions and velocities
            for i, joint_name in enumerate(self.joint_names):
                try:
                    joint_id = self.env.sim.model.joint_name2id(joint_name)
                    if joint_id < len(self.env.sim.data.qpos):
                        current_qpos[i] = self.env.sim.data.qpos[joint_id]
                        current_qvel[i] = self.env.sim.data.qvel[joint_id]
                except Exception:
                    # Joint not found in simulation, use previous action as proxy
                    if len(self.action_history) > 0:
                        current_qpos[i] = self.action_history[-1][i] if i < len(self.action_history[-1]) else 0.0
            
            # Get root angular velocity (simplified)
            root_angvel = np.zeros(3)  # Would need to compute from simulation
            
            # Get projected gravity
            gravity = np.array([0, 0, -9.81])  # Simplified
        else:
            # Fallback if no simulation access - use previous action as proxy for joint positions
            if len(self.action_history) > 0:
                # Use the last action as the current joint position (assumes perfect tracking)
                last_action = self.action_history[-1]
                current_qpos = last_action.copy() if len(last_action) == self.joint_dim else np.zeros(self.joint_dim)
                
                # Estimate velocity from action change
                if len(self.action_history) >= 2:
                    current_qvel = (self.action_history[-1] - self.action_history[-2]) / self.dt
                    current_qvel = current_qvel[:self.joint_dim] if len(current_qvel) >= self.joint_dim else np.zeros(self.joint_dim)
                else:
                    current_qvel = np.zeros(self.joint_dim)
            else:
                # First step - use target position as initial position if reference state init is enabled
                if self.config.reference_state_init and self.current_motion_idx < len(self.robot_trajectory):
                    current_qpos = self.robot_trajectory[self.current_motion_idx].copy()
                    current_qvel = self.robot_velocities[self.current_motion_idx].copy() if len(self.robot_velocities) > self.current_motion_idx else np.zeros(self.joint_dim)
                else:
                    current_qpos = np.zeros(self.joint_dim)
                    current_qvel = np.zeros(self.joint_dim)
            
            root_angvel = np.zeros(3)
            gravity = np.array([0, 0, -9.81])
        
        # Update history
        self.joint_pos_history.append(current_qpos)
        self.joint_vel_history.append(current_qvel)
        self.root_angvel_history.append(root_angvel)
        self.gravity_history.append(gravity)
        
        # Construct proprioception
        proprioception = np.concatenate([
            np.concatenate(self.joint_pos_history),      # joint positions
            np.concatenate(self.joint_vel_history),      # joint velocities
            np.concatenate(self.root_angvel_history),    # root angular velocity
            np.concatenate(self.gravity_history),        # projected gravity
            np.concatenate(list(self.action_history))    # previous actions
        ])
        
        # Add phase variable
        state = np.concatenate([proprioception, [self.current_phase]])
        
        return state.astype(np.float32)
    
    def get_privileged_info(self) -> np.ndarray:
        """Get privileged information for critic (global position, linear velocity)"""
        # Simplified privileged info
        global_pos = np.array([0.0, 0.0, 1.0])  # Would get from simulation
        root_linvel = np.array([0.0, 0.0, 0.0])  # Would get from simulation
        
        return np.concatenate([global_pos, root_linvel]).astype(np.float32)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Step environment with action"""
        # Update action history
        self.action_history.append(action.copy())
        
        # Apply action to environment (joint position targets)
        obs, env_reward, done, info = self.env.step(action)
        
        # Update tracking
        self.episode_length += 1
        
        # Update phase variable
        if self.horizon > 1:
            self.current_phase = min(1.0, self.current_phase + 1.0 / self.horizon)
            self.current_motion_idx = min(self.horizon - 1, int(self.current_phase * (self.horizon - 1)))
        
        # Compute H2O reward
        reward = self.compute_h2o_reward(action)
        
        # Check termination conditions
        terminated = self.check_termination()
        
        # Get next state
        next_state = self.get_state()
        
        return next_state, reward, terminated or done, info
    
    def compute_h2o_reward(self, action: np.ndarray) -> float:
        """Compute H2O reward based on Table I"""
        total_reward = 0.0
        
        # Get current and target states
        target_joint_pos = self.robot_trajectory[self.current_motion_idx]
        target_joint_vel = self.robot_velocities[self.current_motion_idx] if len(self.robot_velocities) > self.current_motion_idx else np.zeros(self.joint_dim)
        
        # Extract current joint positions from history
        current_joint_pos = np.array(list(self.joint_pos_history)[-1])
        current_joint_vel = np.array(list(self.joint_vel_history)[-1])
        
        # === PENALTY TERMS ===
        # DoF position limits
        joint_pos_violations = np.sum(np.maximum(0, np.abs(current_joint_pos) - 2.0))  # Simplified limits
        total_reward += -10.0 * joint_pos_violations
        
        # DoF velocity limits  
        joint_vel_violations = np.sum(np.maximum(0, np.abs(current_joint_vel) - 10.0))  # Simplified limits
        total_reward += -5.0 * joint_vel_violations
        
        # Torque limits (simplified as action magnitude)
        torque_violations = np.sum(np.maximum(0, np.abs(action) - 3.0))  # Simplified limits
        total_reward += -5.0 * torque_violations
        
        # === REGULARIZATION TERMS ===
        # Torques (action regularization)
        total_reward += -1e-6 * np.sum(action**2)
        
        # Action rate (smoothness)
        if len(self.action_history) >= 2:
            action_rate = np.linalg.norm(action - self.action_history[-2])
            total_reward += -0.5 * action_rate
        
        # Feet orientation and heading (simplified)
        total_reward += -2.0 * 0.1  # Simplified foot orientation penalty
        total_reward += -0.1 * 0.1  # Simplified foot heading penalty
        
        # Slippage (simplified)
        total_reward += -1.0 * 0.05  # Simplified slippage penalty
        
        # === TASK REWARD TERMS ===
        # Body position tracking
        pos_error = np.linalg.norm(current_joint_pos - target_joint_pos)
        total_reward += 1.0 * np.exp(-5.0 * pos_error)  # Exponential reward
        
        # DoF position tracking
        dof_pos_reward = np.exp(-10.0 * pos_error)
        total_reward += 0.75 * dof_pos_reward
        
        # DoF velocity tracking
        vel_error = np.linalg.norm(current_joint_vel - target_joint_vel)
        dof_vel_reward = np.exp(-1.0 * vel_error)
        total_reward += 0.5 * dof_vel_reward
        
        # Body velocity tracking (simplified)
        total_reward += 0.5 * 0.5  # Simplified velocity reward
        
        # Body rotation tracking (simplified)
        total_reward += 0.5 * 0.5  # Simplified rotation reward
        
        # Body angular velocity tracking (simplified)
        total_reward += 0.5 * 0.5  # Simplified angular velocity reward
        
        # VR 3-point tracking (if available)
        total_reward += 1.6 * 0.5  # Simplified VR tracking reward
        
        # Body position (feet) tracking (simplified)
        total_reward += 2.1 * 0.5  # Simplified feet tracking reward
        
        return total_reward
    
    def check_termination(self) -> bool:
        """Check termination conditions"""
        # Only check termination after a few steps to allow the agent to start learning
        if self.episode_length < 5:
            return False
            
        # Get tracking error
        if self.current_motion_idx < len(self.robot_trajectory):
            target_joint_pos = self.robot_trajectory[self.current_motion_idx]
            current_joint_pos = np.array(list(self.joint_pos_history)[-1])
            tracking_error = np.linalg.norm(current_joint_pos - target_joint_pos)
            
            # For the first episodes, use a more lenient threshold
            effective_threshold = self.termination_threshold
            if self.episode_length < 20:
                effective_threshold = max(self.termination_threshold, 3.0)  # At least 3.0 for early steps
            
            # Debug logging for termination
            if self.episode_length % 10 == 0:  # Log every 10 steps
                print(f"    Debug - Step {self.episode_length}: tracking_error={tracking_error:.4f}, threshold={effective_threshold:.4f}")
            
            # Check termination threshold
            if tracking_error > effective_threshold:
                print(f"    Terminating due to tracking error: {tracking_error:.4f} > {effective_threshold:.4f}")
                return True
        
        # Check episode length
        if self.episode_length >= self.config.max_episode_length:
            print(f"    Terminating due to max episode length: {self.episode_length} >= {self.config.max_episode_length}")
            return True
        
        # Check if motion is complete
        if self.current_phase >= 1.0:
            print(f"    Terminating due to motion complete: phase={self.current_phase:.4f}")
            return True
        
        return False
    
    def update_termination_curriculum(self, progress: float):
        """Update termination threshold based on training progress"""
        if self.config.termination_curriculum:
            # Use a smoother exponential decay rather than linear
            decay_factor = 1.0 - np.exp(-3.0 * progress)  # Starts slow, accelerates
            self.termination_threshold = (
                self.config.initial_termination_threshold * (1 - decay_factor) + 
                self.config.final_termination_threshold * decay_factor
            )
            # Ensure minimum threshold to prevent too early termination
            self.termination_threshold = max(self.termination_threshold, 0.5)


class H2OPPOAgent:
    """PPO Agent for H2O Phase-based Motion Tracking"""
    
    def __init__(self, config: TrainingConfig, state_dim: int, action_dim: int, privileged_dim: int = 0):
        self.config = config
        self.device = torch.device(config.device)
        
        # Networks
        self.actor = ActorNetwork(state_dim, action_dim).to(self.device)
        self.critic = CriticNetwork(state_dim, privileged_dim).to(self.device)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.lr_critic)
        
        # Training data storage
        self.reset_rollout()
    
    def reset_rollout(self):
        """Reset rollout data"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.privileged_infos = []
    
    def get_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float]:
        """Get action from policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob = self.actor.get_action(state_tensor, deterministic)
        
        return action.cpu().numpy().squeeze(), log_prob.cpu().numpy().squeeze()
    
    def get_value(self, state: np.ndarray, privileged_info: Optional[np.ndarray] = None) -> float:
        """Get value from critic"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        privileged_tensor = None
        
        if privileged_info is not None:
            privileged_tensor = torch.FloatTensor(privileged_info).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            value = self.critic(state_tensor, privileged_tensor)
        
        return value.cpu().numpy().squeeze()
    
    def store_transition(self, state: np.ndarray, action: np.ndarray, reward: float, 
                        value: float, log_prob: float, done: bool, 
                        privileged_info: Optional[np.ndarray] = None):
        """Store transition data"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        self.privileged_infos.append(privileged_info)
    
    def compute_gae(self, next_value: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Generalized Advantage Estimation"""
        rewards = np.array(self.rewards)
        values = np.array(self.values + [next_value])
        dones = np.array(self.dones)
        
        advantages = np.zeros_like(rewards)
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = values[t + 1]
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value = values[t + 1]
            
            delta = rewards[t] + self.config.gamma * next_value * next_non_terminal - values[t]
            gae = delta + self.config.gamma * self.config.gae_lambda * next_non_terminal * gae
            advantages[t] = gae
        
        returns = advantages + values[:-1]
        
        return advantages, returns
    
    def update(self, next_value: float = 0.0) -> Dict[str, float]:
        """Update networks using PPO"""
        if len(self.states) == 0:
            return {}
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(next_value)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.FloatTensor(np.array(self.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(self.log_probs)).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        
        # Privileged information for critic
        privileged_infos = None
        if self.privileged_infos[0] is not None:
            privileged_infos = torch.FloatTensor(np.array(self.privileged_infos)).to(self.device)
        
        # Training metrics
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy_loss = 0
        
        # Multiple epochs of optimization
        for _ in range(self.config.update_epochs):
            # Mini-batch training
            batch_size = len(states)
            mini_batch_size = min(self.config.mini_batch_size, batch_size)
            
            for start_idx in range(0, batch_size, mini_batch_size):
                end_idx = min(start_idx + mini_batch_size, batch_size)
                
                batch_states = states[start_idx:end_idx]
                batch_actions = actions[start_idx:end_idx]
                batch_old_log_probs = old_log_probs[start_idx:end_idx]
                batch_returns = returns_tensor[start_idx:end_idx]
                batch_advantages = advantages_tensor[start_idx:end_idx]
                
                batch_privileged = None
                if privileged_infos is not None:
                    batch_privileged = privileged_infos[start_idx:end_idx]
                
                # Actor update
                mean, log_std = self.actor(batch_states)
                std = torch.exp(log_std)
                dist = Normal(mean, std)
                
                new_log_probs = dist.log_prob(batch_actions).sum(-1)
                entropy = dist.entropy().sum(-1)
                
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 
                                  1 + self.config.clip_epsilon) * batch_advantages
                
                actor_loss = -torch.min(surr1, surr2).mean()
                entropy_loss = -self.config.entropy_coeff * entropy.mean()
                
                total_actor_loss = actor_loss + entropy_loss
                
                self.actor_optimizer.zero_grad()
                total_actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
                self.actor_optimizer.step()
                
                # Critic update
                values_pred = self.critic(batch_states, batch_privileged).squeeze()
                critic_loss = F.mse_loss(values_pred, batch_returns)
                
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
                self.critic_optimizer.step()
                
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy_loss += entropy_loss.item()
        
        # Reset rollout data
        self.reset_rollout()
        
        return {
            'actor_loss': total_actor_loss / self.config.update_epochs,
            'critic_loss': total_critic_loss / self.config.update_epochs,
            'entropy_loss': total_entropy_loss / self.config.update_epochs
        }


def train_h2o_agent(motion_data_path: str, env, config: TrainingConfig = None) -> H2OPPOAgent:
    """Train H2O agent using phase-based motion tracking"""
    
    if config is None:
        config = TrainingConfig()
    
    print("=== H2O Phase-based Motion Tracking Training ===")
    print(f"Loading motion data from {motion_data_path}")
    
    # Load motion data
    if motion_data_path.endswith('.npz'):
        motion_data = dict(np.load(motion_data_path, allow_pickle=True))
    else:
        with open(motion_data_path, 'rb') as f:
            motion_data = pickle.load(f)
    
    print(f"Motion data loaded: {motion_data['robot'].shape[0]} timesteps, {motion_data['robot'].shape[1]} joints")
    
    # Create motion tracker
    tracker = H2OMotionTracker(env, motion_data, config)
    
    # Create agent
    agent = H2OPPOAgent(
        config=config,
        state_dim=tracker.state_dim,
        action_dim=tracker.action_dim,
        privileged_dim=tracker.privileged_dim
    )
    
    print(f"Agent created:")
    print(f"  State dim: {tracker.state_dim}")
    print(f"  Action dim: {tracker.action_dim}")
    print(f"  Privileged dim: {tracker.privileged_dim}")
    print(f"  Device: {config.device}")
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    training_progress = []
    
    total_timesteps = 0
    episode_count = 0
    
    print("\nStarting training...")
    
    while total_timesteps < config.total_timesteps:
        # Reset environment
        state = tracker.reset(random_init=config.reference_state_init)
        episode_reward = 0
        episode_length = 0
        
        # Episode rollout
        while episode_length < config.max_episode_length:
            # Get privileged info for critic
            privileged_info = tracker.get_privileged_info()
            
            # Get action and value
            action, log_prob = agent.get_action(state, deterministic=False)
            value = agent.get_value(state, privileged_info)
            
            # Step environment
            next_state, reward, done, info = tracker.step(action)
            
            # Store transition
            agent.store_transition(state, action, reward, value, log_prob, done, privileged_info)
            
            # Update tracking
            state = next_state
            episode_reward += reward
            episode_length += 1
            total_timesteps += 1
            
            # Check if rollout is full or episode ended
            if len(agent.states) >= config.batch_size or done:
                # Get final value for GAE
                final_privileged_info = tracker.get_privileged_info()
                next_value = agent.get_value(state, final_privileged_info) if not done else 0.0
                
                # Update networks
                losses = agent.update(next_value)
                
                # Log metrics
                if episode_count % 100 == 0 and losses:
                    print(f"Episode {episode_count}, Timesteps {total_timesteps}")
                    print(f"  Reward: {episode_reward:.2f}, Length: {episode_length}")
                    print(f"  Actor Loss: {losses.get('actor_loss', 0):.4f}")
                    print(f"  Critic Loss: {losses.get('critic_loss', 0):.4f}")
                    print(f"  Termination Threshold: {tracker.termination_threshold:.3f}")
                
                break
            
            if done:
                break
        
        # Update termination curriculum
        training_progress_pct = total_timesteps / config.total_timesteps
        tracker.update_termination_curriculum(training_progress_pct)
        
        # Record metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        training_progress.append(total_timesteps)
        
        episode_count += 1
        
        # Enhanced logging every 10 episodes for debugging
        if episode_count % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_length = np.mean(episode_lengths[-10:])
            print(f"Episode {episode_count}, Timesteps {total_timesteps}")
            print(f"  Recent Avg Reward: {avg_reward:.2f}, Recent Avg Length: {avg_length:.1f}")
            print(f"  Termination Threshold: {tracker.termination_threshold:.3f}")
            
            # Debug tracking error
            if len(tracker.joint_pos_history) > 0 and tracker.current_motion_idx < len(tracker.robot_trajectory):
                current_pos = np.array(list(tracker.joint_pos_history)[-1])
                target_pos = tracker.robot_trajectory[tracker.current_motion_idx]
                tracking_error = np.linalg.norm(current_pos - target_pos)
                print(f"  Current Tracking Error: {tracking_error:.4f}")
                print(f"  Current Phase: {tracker.current_phase:.3f}, Motion Index: {tracker.current_motion_idx}")
        
        # Save intermediate results
        if episode_count % 1000 == 0:
            save_path = f'h2o_agent_checkpoint_{episode_count}.pt'
            torch.save({
                'actor_state_dict': agent.actor.state_dict(),
                'critic_state_dict': agent.critic.state_dict(),
                'actor_optimizer_state_dict': agent.actor_optimizer.state_dict(),
                'critic_optimizer_state_dict': agent.critic_optimizer.state_dict(),
                'episode_rewards': episode_rewards,
                'episode_lengths': episode_lengths,
                'training_progress': training_progress,
                'config': config
            }, save_path)
            print(f"Checkpoint saved: {save_path}")
    
    print("\n=== Training Completed ===")
    print(f"Total episodes: {episode_count}")
    print(f"Total timesteps: {total_timesteps}")
    print(f"Average reward (last 100): {np.mean(episode_rewards[-100:]):.2f}")
    
    # Save final model
    final_save_path = 'h2o_agent_final.pt'
    torch.save({
        'actor_state_dict': agent.actor.state_dict(),
        'critic_state_dict': agent.critic.state_dict(),
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'training_progress': training_progress,
        'config': config
    }, final_save_path)
    print(f"Final model saved: {final_save_path}")
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(training_progress, episode_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Timesteps')
    plt.ylabel('Reward')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(training_progress, episode_lengths)
    plt.title('Episode Lengths')
    plt.xlabel('Timesteps')
    plt.ylabel('Length')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('h2o_training_curves.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return agent


def main():
    """Main function to run H2O phase-based training"""
    
    # Create a simple dummy environment for testing
    class DummyEnv:
        def __init__(self):
            self.action_space = type('Space', (), {'shape': (23,)})()
            self.observation_space = type('Space', (), {'shape': (50,)})()
            
        def reset(self):
            return np.random.randn(50)
            
        def step(self, action):
            obs = np.random.randn(50)
            reward = 0.0
            done = np.random.random() < 0.01
            info = {}
            return obs, reward, done, info
    
    # Configuration
    config = TrainingConfig(
        total_timesteps=50000,  # Shorter for testing
        batch_size=1024,
        max_episode_length=500
    )
    
    # Create environment
    env = DummyEnv()
    
    # Run training
    agent = train_h2o_agent(
        motion_data_path='h2o_retargeted_myosuite_data_fixed.npz',
        env=env,
        config=config
    )
    
    print("H2O training completed successfully!")


if __name__ == '__main__':
    main() 
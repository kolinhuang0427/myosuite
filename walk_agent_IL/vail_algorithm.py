import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions import Normal, kl_divergence
import os
import pickle
from collections import deque
import pandas as pd
import gymnasium as gym


class JointTorqueEnvWrapper(gym.Wrapper):
    """
    Wrapper that converts muscle activation environment to joint torque control.
    Maps joint torque actions to the corresponding joint control in the environment.
    """
    
    def __init__(self, env):
        super().__init__(env)
        
        # Define the joint names we want to control with torques
        # Based on the myo_data.pkl structure and available joints
        self.joint_names = [
            'hip_flexion_l', 'hip_flexion_r',
            'hip_adduction_l', 'hip_adduction_r', 
            'hip_rotation_l', 'hip_rotation_r',
            'knee_angle_l', 'knee_angle_r',
            'ankle_angle_l', 'ankle_angle_r',
            'subtalar_angle_l', 'subtalar_angle_r',
            'mtp_angle_l', 'mtp_angle_r'
        ]
        
        self.n_joints = len(self.joint_names)
        
        # Create new action space for joint torques
        # Reasonable torque limits for human joints
        torque_limits = {
            'hip_flexion': 150.0,     # Nm
            'hip_adduction': 120.0,   # Nm
            'hip_rotation': 80.0,     # Nm
            'knee_angle': 120.0,      # Nm
            'ankle_angle': 100.0,     # Nm
            'subtalar_angle': 50.0,   # Nm
            'mtp_angle': 30.0         # Nm
        }
        
        # Set action limits based on joint type
        action_low = []
        action_high = []
        
        for joint_name in self.joint_names:
            joint_type = '_'.join(joint_name.split('_')[:-1])  # Remove _l or _r suffix
            limit = torque_limits.get(joint_type, 50.0)  # Default limit
            action_low.append(-limit)
            action_high.append(limit)
        
        self.action_space = gym.spaces.Box(
            low=np.array(action_low, dtype=np.float32),
            high=np.array(action_high, dtype=np.float32),
            dtype=np.float32
        )
        
        # Keep original observation space - we don't modify observations
        self.observation_space = env.observation_space
        
        # Get joint indices for the environment
        self.joint_indices = []
        for joint_name in self.joint_names:
            try:
                joint_id = self.env.sim.model.joint_name2id(joint_name)
                joint_dof = self.env.sim.model.jnt_dofadr[joint_id]
                self.joint_indices.append(joint_dof)
            except:
                print(f"Warning: Joint {joint_name} not found in model")
                self.joint_indices.append(None)
        
        print(f"JointTorqueEnvWrapper initialized with {self.n_joints} joints")
        print(f"Action space: {self.action_space}")
        print(f"Observation space: {self.observation_space}")
    
    def step(self, action):
        """Convert joint torque action to muscle activations and step environment"""
        # Ensure action is the right size
        if len(action) != self.n_joints:
            raise ValueError(f"Expected action size {self.n_joints}, got {len(action)}")
        
        # For now, we'll use a simple approach: set muscle activations to zero
        # and apply torques directly. In a more sophisticated version, you could
        # use inverse dynamics to compute muscle activations that produce the desired torques.
        
        # Set muscle activations to zero (bypass muscle control)
        muscle_action = np.zeros(self.env.unwrapped.action_space.shape[0])
        
        # Apply joint torques as external forces
        # This is a simplified approach - the torques are applied directly to the joints
        qfrc = np.zeros(self.env.sim.model.nv)
        for i, (joint_idx, torque) in enumerate(zip(self.joint_indices, action)):
            if joint_idx is not None and joint_idx < len(qfrc):
                qfrc[joint_idx] = torque
        
        # Apply the torques to the system
        self.env.sim.data.qfrc_applied[:] = qfrc
        
        # Step the environment with zero muscle activations
        return self.env.step(muscle_action)
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class VAILPolicyNetwork(nn.Module):
    """Policy network for VAIL algorithm"""
    
    def __init__(self, obs_dim, action_dim, hidden_sizes=[256, 256], activation=nn.Tanh):
        super(VAILPolicyNetwork, self).__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Build policy network
        layers = []
        input_dim = obs_dim
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(activation())
            input_dim = hidden_size
        
        self.shared_net = nn.Sequential(*layers)
        
        # Mean and log_std for policy distribution
        self.mean_head = nn.Linear(input_dim, action_dim)
        self.log_std_head = nn.Linear(input_dim, action_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0.0)
    
    def forward(self, obs):
        """Forward pass through policy network"""
        shared_features = self.shared_net(obs)
        mean = self.mean_head(shared_features)
        log_std = self.log_std_head(shared_features)
        log_std = torch.clamp(log_std, min=-20, max=2)  # Stabilize log_std
        
        return mean, log_std
    
    def get_action(self, obs, deterministic=False):
        """Get action from policy"""
        with torch.no_grad():
            mean, log_std = self.forward(obs)
            
            if deterministic:
                return mean
            else:
                std = torch.exp(log_std)
                dist = Normal(mean, std)
                action = dist.sample()
                return action
    
    def evaluate_actions(self, obs, actions):
        """Evaluate log probability of actions"""
        mean, log_std = self.forward(obs)
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return log_prob, entropy


class VAILDiscriminator(nn.Module):
    """Discriminator network for VAIL algorithm"""
    
    def __init__(self, obs_dim, action_dim, hidden_sizes=[256, 256], activation=nn.Tanh):
        super(VAILDiscriminator, self).__init__()
        
        input_dim = obs_dim + action_dim
        
        layers = []
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(activation())
            input_dim = hidden_size
        
        # Output single logit for binary classification
        layers.append(nn.Linear(input_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0.0)
    
    def forward(self, obs, actions):
        """Forward pass through discriminator"""
        x = torch.cat([obs, actions], dim=-1)
        logits = self.network(x)
        return logits
    
    def get_reward(self, obs, actions):
        """Get reward from discriminator for policy training"""
        with torch.no_grad():
            logits = self.forward(obs, actions)
            # Use negative log probability of being expert
            # This encourages policy to maximize probability of being expert
            reward = -F.logsigmoid(logits).squeeze(-1)
            return reward


class VAILValueNetwork(nn.Module):
    """Value network for VAIL algorithm"""
    
    def __init__(self, obs_dim, hidden_sizes=[256, 256], activation=nn.Tanh):
        super(VAILValueNetwork, self).__init__()
        
        layers = []
        input_dim = obs_dim
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(activation())
            input_dim = hidden_size
        
        layers.append(nn.Linear(input_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0.0)
    
    def forward(self, obs):
        """Forward pass through value network"""
        return self.network(obs).squeeze(-1)


class VAILAlgorithm:
    """VAIL (Variational Adversarial Imitation Learning) Algorithm for Joint Torque Control"""
    
    def __init__(
        self,
        env,
        expert_data_path,
        policy_lr=3e-4,
        discriminator_lr=3e-4,
        value_lr=3e-4,
        gamma=0.99,
        lambda_gae=0.95,
        clip_epsilon=0.2,
        entropy_coeff=0.01,
        vf_coeff=0.5,
        max_grad_norm=0.5,
        n_policy_epochs=10,
        n_discriminator_epochs=5,
        batch_size=64,
        buffer_size=2048,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        use_joint_control=True
    ):
        # Wrap environment for joint torque control if requested
        if use_joint_control:
            self.env = JointTorqueEnvWrapper(env)
            print("Using joint torque control")
        else:
            self.env = env
            print("Using original muscle activation control")
        
        self.device = device
        
        # Get environment dimensions
        self.obs_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        
        print(f"Observation dimension: {self.obs_dim}")
        print(f"Action dimension: {self.action_dim}")
        
        # Hyperparameters
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.clip_epsilon = clip_epsilon
        self.entropy_coeff = entropy_coeff
        self.vf_coeff = vf_coeff
        self.max_grad_norm = max_grad_norm
        self.n_policy_epochs = n_policy_epochs
        self.n_discriminator_epochs = n_discriminator_epochs
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        
        # Initialize networks
        self.policy = VAILPolicyNetwork(self.obs_dim, self.action_dim).to(device)
        self.discriminator = VAILDiscriminator(self.obs_dim, self.action_dim).to(device)
        self.value_net = VAILValueNetwork(self.obs_dim).to(device)
        
        # Initialize optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=policy_lr)
        self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=discriminator_lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=value_lr)
        
        # Load expert data
        self.expert_data = self._load_expert_data(expert_data_path, use_joint_control)
        
        # Experience buffer
        self.buffer = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': [],
            'next_observations': []
        }
        
    def _load_expert_data(self, expert_data_path, use_joint_control=True):
        """Load expert demonstrations from myo_data.pkl or traditional format"""
        print(f"Loading expert data from {expert_data_path}")
        
        if expert_data_path.endswith('myo_data.pkl'):
            return self._load_myo_data(expert_data_path, use_joint_control)
        elif expert_data_path.endswith('.pickle') or expert_data_path.endswith('.pkl'):
            with open(expert_data_path, 'rb') as f:
                expert_data = pickle.load(f)
        else:
            raise ValueError(f"Unsupported expert data format: {expert_data_path}")
        
        # Extract observations and actions for traditional format
        expert_obs = []
        expert_actions = []
        
        for traj in expert_data:
            if isinstance(traj, dict):
                obs = traj.get('observations', [])
                actions = traj.get('actions', [])
            else:
                obs = traj[0] if len(traj) > 0 else []
                actions = traj[1] if len(traj) > 1 else []
            
            if len(obs) > 0 and len(actions) > 0:
                expert_obs.extend(obs)
                expert_actions.extend(actions)
        
        expert_obs = torch.FloatTensor(np.array(expert_obs)).to(self.device)
        expert_actions = torch.FloatTensor(np.array(expert_actions)).to(self.device)
        
        print(f"Loaded {len(expert_obs)} expert transitions")
        
        return {'observations': expert_obs, 'actions': expert_actions}
    
    def _load_myo_data(self, data_path, use_joint_control=True):
        """Load and process expert data from myo_data.pkl"""
        print(f"Loading myo_data from {data_path}")
        
        with open(data_path, 'rb') as f:
            myo_data = pickle.load(f)
        
        # Extract joint positions and velocities
        qpos_df = myo_data['qpos']
        qvel_df = myo_data.get('qvel', pd.DataFrame())
        
        print(f"Available joint positions: {list(qpos_df.columns)}")
        if not qvel_df.empty:
            print(f"Available joint velocities: {list(qvel_df.columns)}")
        
        # Define joint mapping for our control
        if use_joint_control and hasattr(self.env, 'joint_names'):
            joint_names = self.env.joint_names
        else:
            # Default joint names from myo_data
            joint_names = [
                'hip_flexion_l', 'hip_flexion_r',
                'hip_adduction_l', 'hip_adduction_r',
                'hip_rotation_l', 'hip_rotation_r',
                'knee_angle_l', 'knee_angle_r',
                'ankle_angle_l', 'ankle_angle_r',
                'subtalar_angle_l', 'subtalar_angle_r',
                'mtp_angle_l', 'mtp_angle_r'
            ]
        
        # Extract joint data that exists in the dataset
        available_joints = []
        joint_positions = []
        joint_velocities = []
        
        for joint in joint_names:
            if joint in qpos_df.columns:
                available_joints.append(joint)
                joint_positions.append(qpos_df[joint].values)
                
                if joint in qvel_df.columns:
                    joint_velocities.append(qvel_df[joint].values)
                else:
                    # Compute velocities from positions if not available
                    pos_data = qpos_df[joint].values
                    vel_data = np.gradient(pos_data)  # Simple numerical derivative
                    joint_velocities.append(vel_data)
        
        print(f"Using {len(available_joints)} joints: {available_joints}")
        
        if len(available_joints) == 0:
            raise ValueError("No matching joints found in expert data")
        
        # Stack joint data
        joint_positions = np.column_stack(joint_positions)
        joint_velocities = np.column_stack(joint_velocities)
        
        # Create fake observations that match the environment's observation space
        # We'll use random observations for now, but in practice you'd want to 
        # either record the actual environment observations or use a more sophisticated mapping
        n_samples = len(joint_positions) - 1
        
        # Create synthetic observations that match environment obs space
        fake_obs = []
        for i in range(n_samples):
            # Create a fake observation vector that matches the environment's observation space
            obs = np.random.randn(self.obs_dim) * 0.1  # Small random values
            # Optionally, you could put the joint information in the first few dimensions
            n_joints_to_use = min(len(available_joints), self.obs_dim // 2)
            if n_joints_to_use > 0:
                obs[:n_joints_to_use] = joint_positions[i][:n_joints_to_use]
                if self.obs_dim > n_joints_to_use:
                    obs[n_joints_to_use:2*n_joints_to_use] = joint_velocities[i][:n_joints_to_use]
            fake_obs.append(obs)
        
        # Create expert actions (joint torques)
        expert_actions = []
        for i in range(n_samples):
            # Action: compute desired torques from position changes
            pos_diff = joint_positions[i + 1] - joint_positions[i]
            vel_diff = joint_velocities[i + 1] - joint_velocities[i]
            
            # Simple PD-like torque calculation
            kp, kd = 100.0, 10.0  # Gain parameters
            torques = kp * pos_diff + kd * vel_diff
            
            # Resize torques to match action dimension
            if len(torques) > self.action_dim:
                torques = torques[:self.action_dim]
            elif len(torques) < self.action_dim:
                # Pad with zeros
                padded_torques = np.zeros(self.action_dim)
                padded_torques[:len(torques)] = torques
                torques = padded_torques
            
            # Clip torques to reasonable limits
            torques = np.clip(torques, -200.0, 200.0)
            expert_actions.append(torques)
        
        expert_obs = torch.FloatTensor(np.array(fake_obs)).to(self.device)
        expert_actions = torch.FloatTensor(np.array(expert_actions)).to(self.device)
        
        print(f"Created {len(expert_obs)} expert transitions from myo_data")
        print(f"Observation shape: {expert_obs.shape}")
        print(f"Action shape: {expert_actions.shape}")
        
        return {'observations': expert_obs, 'actions': expert_actions}
    
    def collect_trajectory(self, max_steps=1000):
        """Collect a trajectory from the environment"""
        reset_result = self.env.reset()
        if isinstance(reset_result, tuple):
            obs, info = reset_result
        else:
            obs = reset_result
            
        done = False
        step_count = 0
        
        trajectory = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': [],
            'next_observations': []
        }
        
        while not done and step_count < max_steps:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            
            # Get action from policy
            action = self.policy.get_action(obs_tensor).squeeze(0)
            log_prob, _ = self.policy.evaluate_actions(obs_tensor, action.unsqueeze(0))
            value = self.value_net(obs_tensor)
            
            # Convert action to numpy for environment
            action_np = action.cpu().numpy()
            
            # Step environment
            step_result = self.env.step(action_np)
            if len(step_result) == 5:  # new gym API with truncated
                next_obs, env_reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:  # old gym API
                next_obs, env_reward, done, info = step_result
            
            # Get discriminator reward
            disc_reward = self.discriminator.get_reward(obs_tensor, action.unsqueeze(0)).item()
            
            # Store transition
            trajectory['observations'].append(obs)
            trajectory['actions'].append(action_np)
            trajectory['rewards'].append(disc_reward)  # Use discriminator reward
            trajectory['values'].append(value.item())
            trajectory['log_probs'].append(log_prob.item())
            trajectory['dones'].append(done)
            trajectory['next_observations'].append(next_obs)
            
            obs = next_obs
            step_count += 1
        
        return trajectory
    
    def compute_gae(self, rewards, values, dones, next_values):
        """Compute Generalized Advantage Estimation"""
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_values
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lambda_gae * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        returns = [adv + val for adv, val in zip(advantages, values)]
        
        return advantages, returns
    
    def update_discriminator(self):
        """Update discriminator network"""
        # Sample expert data
        expert_batch_size = min(self.batch_size, len(self.expert_data['observations']))
        expert_indices = np.random.choice(len(self.expert_data['observations']), expert_batch_size, replace=False)
        expert_obs = self.expert_data['observations'][expert_indices]
        expert_actions = self.expert_data['actions'][expert_indices]
        
        # Sample policy data
        policy_batch_size = min(self.batch_size, len(self.buffer['observations']))
        policy_indices = np.random.choice(len(self.buffer['observations']), policy_batch_size, replace=False)
        
        policy_obs = torch.FloatTensor([self.buffer['observations'][i] for i in policy_indices]).to(self.device)
        policy_actions = torch.FloatTensor([self.buffer['actions'][i] for i in policy_indices]).to(self.device)
        
        # Forward pass
        expert_logits = self.discriminator(expert_obs, expert_actions)
        policy_logits = self.discriminator(policy_obs, policy_actions)
        
        # Discriminator loss (binary cross entropy)
        expert_loss = F.binary_cross_entropy_with_logits(expert_logits, torch.ones_like(expert_logits))
        policy_loss = F.binary_cross_entropy_with_logits(policy_logits, torch.zeros_like(policy_logits))
        
        discriminator_loss = expert_loss + policy_loss
        
        # Update discriminator
        self.discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.max_grad_norm)
        self.discriminator_optimizer.step()
        
        return discriminator_loss.item()
    
    def update_policy(self):
        """Update policy and value networks using PPO"""
        # Convert buffer to tensors
        observations = torch.FloatTensor(self.buffer['observations']).to(self.device)
        actions = torch.FloatTensor(self.buffer['actions']).to(self.device)
        old_log_probs = torch.FloatTensor(self.buffer['log_probs']).to(self.device)
        values = torch.FloatTensor(self.buffer['values']).to(self.device)
        rewards = self.buffer['rewards']
        dones = self.buffer['dones']
        
        # Compute GAE
        next_value = self.value_net(torch.FloatTensor(self.buffer['next_observations'][-1]).unsqueeze(0).to(self.device)).item()
        advantages, returns = self.compute_gae(rewards, self.buffer['values'], dones, next_value)
        
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Update policy for multiple epochs
        for _ in range(self.n_policy_epochs):
            # Get current policy outputs
            new_log_probs, entropy = self.policy.evaluate_actions(observations, actions)
            new_values = self.value_net(observations)
            
            # PPO ratio
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # PPO clipped objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(new_values, returns)
            
            # Entropy bonus
            entropy_loss = -self.entropy_coeff * entropy.mean()
            
            # Total loss
            total_loss = policy_loss + self.vf_coeff * value_loss + entropy_loss
            
            # Update networks
            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
            self.policy_optimizer.step()
            self.value_optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item()
        }
    
    def train(self, total_timesteps, log_interval=50, save_interval=1000, save_dir='./vail_checkpoints'):
        """Train VAIL agent"""
        os.makedirs(save_dir, exist_ok=True)
        
        timestep = 0
        episode = 0
        
        print(f"Starting VAIL training for {total_timesteps} timesteps")
        
        while timestep < total_timesteps:
            # Collect trajectory
            trajectory = self.collect_trajectory(max_steps=self.buffer_size)
            
            # Add to buffer
            for key in self.buffer:
                self.buffer[key].extend(trajectory[key])
            
            timestep += len(trajectory['observations'])
            episode += 1
            
            # Limit buffer size
            if len(self.buffer['observations']) > self.buffer_size:
                excess = len(self.buffer['observations']) - self.buffer_size
                for key in self.buffer:
                    self.buffer[key] = self.buffer[key][excess:]
            
            # Update networks if we have enough data
            if len(self.buffer['observations']) >= self.batch_size:
                # Update discriminator
                for _ in range(self.n_discriminator_epochs):
                    disc_loss = self.update_discriminator()
                
                # Update policy
                policy_losses = self.update_policy()
                
                # Clear buffer
                for key in self.buffer:
                    self.buffer[key] = []
            
            # Logging
            if episode % log_interval == 0:
                episode_reward = sum(trajectory['rewards'])
                print(f"Episode {episode}, Timestep {timestep}, Reward: {episode_reward:.2f}")
            
            # Save checkpoint
            if episode % save_interval == 0:
                checkpoint_path = os.path.join(save_dir, f'checkpoint_episode_{episode}.pt')
                self.save_checkpoint(checkpoint_path)
        
        # Save final model
        final_path = os.path.join(save_dir, 'final_model.pt')
        self.save_checkpoint(final_path)
        print(f"Training completed! Final model saved to {final_path}")
    
    def save_checkpoint(self, path):
        """Save model checkpoint"""
        checkpoint = {
            'policy_state_dict': self.policy.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'value_net_state_dict': self.value_net.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'discriminator_optimizer_state_dict': self.discriminator_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.value_net.load_state_dict(checkpoint['value_net_state_dict'])
        
        if 'policy_optimizer_state_dict' in checkpoint:
            self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        if 'discriminator_optimizer_state_dict' in checkpoint:
            self.discriminator_optimizer.load_state_dict(checkpoint['discriminator_optimizer_state_dict'])
        if 'value_optimizer_state_dict' in checkpoint:
            self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        
        print(f"Checkpoint loaded from {path}")
    
    def evaluate(self, n_episodes=10, render=False):
        """Evaluate the trained agent"""
        total_rewards = []
        
        for episode in range(n_episodes):
            reset_result = self.env.reset()
            if isinstance(reset_result, tuple):
                obs, info = reset_result
            else:
                obs = reset_result
            
            done = False
            episode_reward = 0
            step_count = 0
            max_steps = 1000
            
            while not done and step_count < max_steps:
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                action = self.policy.get_action(obs_tensor, deterministic=True)
                action_np = action.squeeze(0).cpu().numpy()
                
                step_result = self.env.step(action_np)
                if len(step_result) == 5:  # new gym API with truncated
                    obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:  # old gym API
                    obs, reward, done, info = step_result
                
                episode_reward += reward
                step_count += 1
                
                if render:
                    try:
                        self.env.render()
                    except:
                        pass
            
            total_rewards.append(episode_reward)
        
        mean_reward = np.mean(total_rewards)
        std_reward = np.std(total_rewards)
        
        print(f"Evaluation over {n_episodes} episodes:")
        print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
        
        return mean_reward, std_reward 
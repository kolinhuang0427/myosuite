import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import pickle
from collections import deque
import pandas as pd
import gymnasium as gym
from typing import Optional, Tuple, Dict, List
import math


class JointTorqueEnvWrapper(gym.Wrapper):
    """
    Wrapper that converts muscle activation environment to joint torque control.
    Maps joint torque actions to the corresponding joint control in the environment.
    """
    
    def __init__(self, env):
        super().__init__(env)
        
        # Define the joint names we want to control with torques
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
        torque_limits = {
            'hip_flexion': 150.0,     # Nm
            'hip_adduction': 120.0,   # Nm
            'hip_rotation': 80.0,     # Nm
            'knee_angle': 120.0,      # Nm
            'ankle_angle': 100.0,     # Nm
            'subtalar_angle': 50.0,   # Nm
            'mtp_angle': 30.0         # Nm
        }
        
        action_low = []
        action_high = []
        
        for joint_name in self.joint_names:
            joint_type = '_'.join(joint_name.split('_')[:-1])
            limit = torque_limits.get(joint_type, 50.0)
            action_low.append(-limit)
            action_high.append(limit)
        
        self.action_space = gym.spaces.Box(
            low=np.array(action_low, dtype=np.float32),
            high=np.array(action_high, dtype=np.float32),
            dtype=np.float32
        )
        
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
    
    def step(self, action):
        if len(action) != self.n_joints:
            raise ValueError(f"Expected action size {self.n_joints}, got {len(action)}")
        
        muscle_action = np.zeros(self.env.unwrapped.action_space.shape[0])
        
        qfrc = np.zeros(self.env.sim.model.nv)
        for i, (joint_idx, torque) in enumerate(zip(self.joint_indices, action)):
            if joint_idx is not None and joint_idx < len(qfrc):
                qfrc[joint_idx] = torque
        
        self.env.sim.data.qfrc_applied[:] = qfrc
        return self.env.step(muscle_action)
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for timestep encoding in diffusion."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class DiffusionMLP(nn.Module):
    """MLP backbone for diffusion policy with time and state conditioning."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        time_dim: int = 32,
        hidden_dims: List[int] = [256, 256, 256],
        activation: nn.Module = nn.Mish
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.time_dim = time_dim
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            activation(),
            nn.Linear(time_dim * 2, time_dim),
        )
        
        # Input projection
        input_dim = state_dim + action_dim + time_dim
        
        # Build the main network
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                activation(),
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.net = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor, time: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the diffusion MLP.
        
        Args:
            x: Noisy action tensor [batch_size, action_dim]
            time: Timestep tensor [batch_size]
            state: State conditioning tensor [batch_size, state_dim]
        
        Returns:
            Predicted noise [batch_size, action_dim]
        """
        # Embed time
        time_emb = self.time_mlp(time)
        
        # Concatenate inputs
        net_input = torch.cat([x, time_emb, state], dim=-1)
        
        # Forward pass
        return self.net(net_input)


class DDPMScheduler:
    """DDPM noise scheduler for diffusion training and sampling."""
    
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear"
    ):
        self.num_train_timesteps = num_train_timesteps
        
        if beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        elif beta_schedule == "cosine":
            # Cosine schedule as in improved DDPM
            s = 0.008
            steps = num_train_timesteps + 1
            x = torch.linspace(0, num_train_timesteps, steps, dtype=torch.float32)
            alphas_cumprod = torch.cos(((x / num_train_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clip(betas, 0, 0.999)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        
        # Track current device
        self.device = 'cpu'
    
    def to(self, device):
        """Move all tensors to the specified device."""
        self.device = device
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        return self
    
    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """Add noise to samples according to the noise schedule."""
        # Ensure scheduler tensors are on the same device as inputs
        if self.device != original_samples.device:
            self.to(original_samples.device)
            
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps].flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
        
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps].flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples
    
    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        generator=None
    ) -> torch.Tensor:
        """Perform one denoising step."""
        # Ensure scheduler tensors are on the same device as inputs
        if self.device != model_output.device:
            self.to(model_output.device)
            
        t = timestep
        
        # 1. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod_prev[t] if t > 0 else torch.tensor(1.0, device=self.device)
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        
        # 2. compute predicted original sample from predicted noise
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        
        # 3. Clip predicted original sample
        pred_original_sample = torch.clamp(pred_original_sample, -1, 1)
        
        # 4. compute coefficients for pred_original_sample x_0 and current sample x_t
        pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * self.betas[t]) / beta_prod_t
        current_sample_coeff = self.alphas[t] ** (0.5) * beta_prod_t_prev / beta_prod_t
        
        # 5. compute predicted previous sample μ_t
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample
        
        # 6. add noise
        variance = 0
        if t > 0:
            device = model_output.device
            noise = torch.randn(model_output.shape, generator=generator, device=device, dtype=model_output.dtype)
            variance = (self.posterior_variance[t] ** 0.5) * noise
        
        pred_prev_sample = pred_prev_sample + variance
        
        return pred_prev_sample


class DiffusionPolicy(nn.Module):
    """Diffusion-based policy for imitation learning."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        num_train_timesteps: int = 100,
        num_inference_steps: int = 20,
        action_scale: float = 1.0,
        **model_kwargs
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_scale = action_scale
        self.num_inference_steps = num_inference_steps
        
        # Diffusion model
        self.model = DiffusionMLP(state_dim, action_dim, **model_kwargs)
        
        # Noise scheduler
        self.scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps)
        
    def forward(self, states: torch.Tensor, actions: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Forward pass during training - predict noise."""
        return self.model(actions, timesteps, states)
    
    def get_action(self, states: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Generate actions using DDPM sampling."""
        batch_size = states.shape[0]
        device = states.device
        
        # Start from random noise
        actions = torch.randn(batch_size, self.action_dim, device=device) * self.action_scale
        
        # Set timesteps
        timesteps = list(range(len(self.scheduler.alphas_cumprod)))[::-1]
        step_size = len(timesteps) // self.num_inference_steps
        timesteps = timesteps[::step_size]
        
        # Denoising loop
        for i, t in enumerate(timesteps):
            timestep = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            with torch.no_grad():
                # Predict noise
                noise_pred = self.model(actions, timestep, states)
                
                # Perform denoising step
                actions = self.scheduler.step(noise_pred, t, actions)
        
        return actions
    
    def compute_loss(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Compute diffusion training loss."""
        batch_size = states.shape[0]
        device = states.device
        
        # Sample random timesteps
        timesteps = torch.randint(
            0, len(self.scheduler.alphas_cumprod), (batch_size,), device=device
        ).long()
        
        # Sample noise
        noise = torch.randn_like(actions)
        
        # Add noise to actions
        noisy_actions = self.scheduler.add_noise(actions, noise, timesteps)
        
        # Predict noise
        noise_pred = self.forward(states, noisy_actions, timesteps)
        
        # Compute loss
        loss = F.mse_loss(noise_pred, noise)
        
        return loss


class DiffusionILAlgorithm:
    """Diffusion-based Imitation Learning Algorithm."""
    
    def __init__(
        self,
        env,
        expert_data_path: str,
        policy_lr: float = 1e-4,
        batch_size: int = 256,
        buffer_size: int = 10000,
        num_train_timesteps: int = 100,
        num_inference_steps: int = 20,
        action_scale: float = 1.0,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        use_joint_control: bool = True,
        **policy_kwargs
    ):
        self.device = device
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        
        # Wrap environment for joint control if requested
        if use_joint_control:
            self.env = JointTorqueEnvWrapper(env)
        else:
            self.env = env
        
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        
        print(f"Environment dimensions: state={self.state_dim}, action={self.action_dim}")
        
        # Initialize diffusion policy
        self.policy = DiffusionPolicy(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            num_train_timesteps=num_train_timesteps,
            num_inference_steps=num_inference_steps,
            action_scale=action_scale,
            **policy_kwargs
        ).to(device)
        
        # Optimizer
        self.optimizer = optim.AdamW(self.policy.parameters(), lr=policy_lr)
        
        # Load expert data
        self.expert_states, self.expert_actions = self._load_expert_data(expert_data_path, use_joint_control)
        
        # Data buffer for training
        self.data_buffer = deque(maxlen=buffer_size)
        
        # Training statistics
        self.training_stats = {
            'losses': [],
            'timesteps': [],
            'eval_rewards': []
        }
        
        print(f"DiffusionIL initialized with {len(self.expert_states)} expert samples")
    
    def _load_expert_data(self, expert_data_path: str, use_joint_control: bool = True):
        """Load expert demonstration data."""
        if expert_data_path.endswith('.pickle') or expert_data_path.endswith('.pkl'):
            return self._load_myo_data(expert_data_path, use_joint_control)
        else:
            with open(expert_data_path, 'rb') as f:
                expert_data = pickle.load(f)
            
            states = []
            actions = []
            
            for trajectory in expert_data:
                states.extend(trajectory['observations'])
                actions.extend(trajectory['actions'])
            
            states = torch.FloatTensor(np.array(states)).to(self.device)
            actions = torch.FloatTensor(np.array(actions)).to(self.device)
            
            return states, actions
    
    def _load_myo_data(self, data_path: str, use_joint_control: bool = True):
        """Load MyoSuite demonstration data from pickle file."""
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"Loaded data keys: {data.keys()}")
        
        # Extract observations and actions based on data structure
        if 'observations' in data and 'actions' in data:
            observations = data['observations']
            actions = data['actions']
        elif 'qpos' in data and 'qvel' in data:
            # Handle case where we only have joint positions and velocities
            qpos = data['qpos']
            qvel = data['qvel']
            qvel = np.zeros_like(qpos)
            
            print(f"Found qpos: {qpos.shape}, qvel: {qvel.shape}")
            
            if use_joint_control:
                # For joint control, use qpos as actions (joint torques)
                # and create synthetic observations by concatenating qpos and qvel
                actions = qpos  # Use joint positions as target joint torques
                observations = np.concatenate([qpos, qvel], axis=1)  # Combine qpos and qvel as observations
                
                # Trim actions to match our joint control action space (14 joints)
                if actions.shape[1] > self.action_dim:
                    actions = actions[:, :self.action_dim]
                elif actions.shape[1] < self.action_dim:
                    # Pad with zeros if needed
                    padding = np.zeros((actions.shape[0], self.action_dim - actions.shape[1]))
                    actions = np.concatenate([actions, padding], axis=1)
                
                # Create synthetic full observations by padding to match environment obs space
                obs_dim = self.env.observation_space.shape[0]
                if observations.shape[1] < obs_dim:
                    # Pad observations to match the full observation space
                    padding = np.zeros((observations.shape[0], obs_dim - observations.shape[1]))
                    observations = np.concatenate([observations, padding], axis=1)
                elif observations.shape[1] > obs_dim:
                    # Trim if too large
                    observations = observations[:, :obs_dim]
                    
                print(f"Using qpos/qvel data: actions from qpos, observations from qpos+qvel+padding")
            else:
                # For muscle control, this data format is not suitable
                raise ValueError("qpos/qvel data is not suitable for muscle control. Use joint control instead.")
        else:
            # Handle different data formats
            observations = []
            actions = []
            
            for key, value in data.items():
                if isinstance(value, dict) and 'obs' in value:
                    observations.extend(value['obs'])
                    if use_joint_control and 'qpos' in value:
                        # Use joint positions/velocities as actions for joint control
                        joint_actions = value['qpos'][:, :self.action_dim]
                        actions.extend(joint_actions)
                    elif 'acts' in value:
                        actions.extend(value['acts'])
                    elif 'actions' in value:
                        actions.extend(value['actions'])
        
        # Convert to tensors
        observations = np.array(observations)
        actions = np.array(actions)
        
        # Ensure we have valid data
        if len(observations) == 0 or len(actions) == 0:
            raise ValueError(f"No valid expert data found. Observations: {len(observations)}, Actions: {len(actions)}")
        
        # Ensure consistent dimensions
        if len(actions.shape) == 1:
            actions = actions.reshape(-1, 1)
        if len(observations.shape) == 1:
            observations = observations.reshape(-1, 1)
        
        print(f"Processed data shapes: obs={observations.shape}, acts={actions.shape}")
        
        states = torch.FloatTensor(observations).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        
        return states, actions
    
    def sample_expert_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample a batch from expert demonstrations."""
        indices = torch.randint(0, len(self.expert_states), (self.batch_size,))
        states = self.expert_states[indices]
        actions = self.expert_actions[indices]
        return states, actions
    
    def collect_trajectory(self, max_steps: int = 1000) -> Dict:
        """Collect a trajectory using current policy."""
        trajectory = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'dones': []
        }
        
        reset_result = self.env.reset()
        if isinstance(reset_result, tuple):
            obs, _ = reset_result
        else:
            obs = reset_result
        
        total_reward = 0
        
        for step in range(max_steps):
            # Get action from policy
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action_tensor = self.policy.get_action(obs_tensor)
                action = action_tensor.cpu().numpy().flatten()
            
            # Clip action to environment bounds
            action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
            
            # Execute action
            step_result = self.env.step(action)
            if len(step_result) == 5:  # new gym API
                next_obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:  # old gym API
                next_obs, reward, done, info = step_result
            
            # Store transition
            trajectory['observations'].append(obs.copy())
            trajectory['actions'].append(action.copy())
            trajectory['rewards'].append(reward)
            trajectory['dones'].append(done)
            
            total_reward += reward
            obs = next_obs
            
            if done:
                break
        
        trajectory['total_reward'] = total_reward
        return trajectory
    
    def train_step(self) -> float:
        """Perform one training step."""
        # Sample expert batch
        states, actions = self.sample_expert_batch()
        
        # Compute loss
        loss = self.policy.compute_loss(states, actions)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train(
        self,
        total_timesteps: int,
        log_interval: int = 100,
        eval_interval: int = 1000,
        save_interval: int = 5000,
        save_dir: str = './diffusion_checkpoints'
    ):
        """Train the diffusion policy."""
        os.makedirs(save_dir, exist_ok=True)
        
        print("Starting Diffusion IL training...")
        print(f"Total timesteps: {total_timesteps}")
        print(f"Batch size: {self.batch_size}")
        print(f"Device: {self.device}")
        
        for timestep in range(total_timesteps):
            # Training step
            loss = self.train_step()
            self.training_stats['losses'].append(loss)
            self.training_stats['timesteps'].append(timestep)
            
            # Logging
            if timestep % log_interval == 0:
                avg_loss = np.mean(self.training_stats['losses'][-log_interval:])
                print(f"Timestep {timestep}: Loss = {avg_loss:.6f}")
            
            # Evaluation
            if timestep % eval_interval == 0 and timestep > 0:
                mean_reward, std_reward = self.evaluate(n_episodes=5)
                self.training_stats['eval_rewards'].append((timestep, mean_reward, std_reward))
                print(f"Evaluation at timestep {timestep}: {mean_reward:.2f} ± {std_reward:.2f}")
            
            # Save checkpoint
            if timestep % save_interval == 0 and timestep > 0:
                checkpoint_path = os.path.join(save_dir, f'checkpoint_{timestep}.pt')
                self.save_checkpoint(checkpoint_path)
                print(f"Saved checkpoint: {checkpoint_path}")
        
        # Save final model
        final_path = os.path.join(save_dir, 'final_model.pt')
        self.save_checkpoint(final_path)
        print(f"Training completed! Final model saved: {final_path}")
    
    def evaluate(self, n_episodes: int = 10, render: bool = False) -> Tuple[float, float]:
        """Evaluate the current policy."""
        self.policy.eval()
        
        episode_rewards = []
        
        for episode in range(n_episodes):
            trajectory = self.collect_trajectory()
            episode_rewards.append(trajectory['total_reward'])
        
        self.policy.train()
        
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        
        return mean_reward, std_reward
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        checkpoint = {
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_stats': self.training_stats,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'training_stats' in checkpoint:
            self.training_stats = checkpoint['training_stats']
        print(f"Loaded checkpoint from {path}") 
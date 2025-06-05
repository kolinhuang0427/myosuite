import torch
import numpy as np
from typing import Union, Dict, Any
import os
import sys

# Add the current directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from diffusion_algorithm import DiffusionILAlgorithm, JointTorqueEnvWrapper


class DiffusionPolicyWrapper:
    """
    Wrapper for trained Diffusion IL policy that provides a clean interface
    for loading and using the policy in different contexts.
    """
    
    def __init__(
        self,
        model_path: str,
        env=None,
        device: str = 'auto',
        num_inference_steps: int = 20,
        deterministic: bool = False
    ):
        """
        Initialize the Diffusion Policy Wrapper.
        
        Args:
            model_path: Path to the trained diffusion model checkpoint
            env: Environment instance (optional, for creating wrapped env)
            device: Device to run inference on ('auto', 'cpu', 'cuda')
            num_inference_steps: Number of diffusion inference steps
            deterministic: Whether to use deterministic sampling (not applicable to diffusion)
        """
        self.model_path = model_path
        self.num_inference_steps = num_inference_steps
        self.deterministic = deterministic
        
        # Set device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Load model
        self._load_model(env)
        
        print(f"DiffusionPolicyWrapper initialized with:")
        print(f"  Model path: {model_path}")
        print(f"  Device: {self.device}")
        print(f"  Inference steps: {num_inference_steps}")
        print(f"  State dim: {self.state_dim}")
        print(f"  Action dim: {self.action_dim}")
    
    def _load_model(self, env=None):
        """Load the trained diffusion model."""
        import os
        import tempfile
        import pickle
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model checkpoint not found at {self.model_path}")
        
        # Load checkpoint to get model architecture info
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        
        self.state_dim = checkpoint['state_dim']
        self.action_dim = checkpoint['action_dim']
        
        # Create dummy environment if not provided (for getting dimensions)
        if env is None:
            from myosuite.utils import gym
            import myosuite
            try:
                dummy_env = gym.make('myoLegWalk-v0')
                if hasattr(dummy_env, 'sim'):  # Check if it's a mujoco env
                    env = JointTorqueEnvWrapper(dummy_env)
                else:
                    env = dummy_env
            except:
                # Create minimal dummy env if MyoSuite env creation fails
                class DummyEnv:
                    def __init__(self, state_dim, action_dim):
                        self.observation_space = type('MockSpace', (), {'shape': (state_dim,)})()
                        self.action_space = type('MockSpace', (), {
                            'shape': (action_dim,),
                            'low': np.full(action_dim, -1.0),
                            'high': np.full(action_dim, 1.0)
                        })()
                
                env = DummyEnv(self.state_dim, self.action_dim)
        
        # Create a temporary dummy expert data file for algorithm initialization
        dummy_data = {
            'qpos': np.random.randn(10, self.action_dim),
            'qvel': np.random.randn(10, self.action_dim)
        }
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            pickle.dump(dummy_data, f)
            dummy_data_path = f.name
        
        try:
            # Create algorithm instance with loaded dimensions
            self.algorithm = DiffusionILAlgorithm(
                env=env,
                expert_data_path=dummy_data_path,  # Use dummy data
                device=self.device,
                use_joint_control=True,
                num_inference_steps=self.num_inference_steps
            )
            
            # Load the trained weights
            self.algorithm.load_checkpoint(self.model_path)
            
            # Set to evaluation mode
            self.algorithm.policy.eval()
            
            # Store the policy for direct access
            self.policy = self.algorithm.policy
        finally:
            # Clean up temporary file
            os.unlink(dummy_data_path)
    
    def predict(
        self,
        observation: Union[np.ndarray, torch.Tensor],
        deterministic: bool = None
    ) -> np.ndarray:
        """
        Predict action for given observation.
        
        Args:
            observation: Input observation
            deterministic: Whether to use deterministic sampling (ignored for diffusion)
        
        Returns:
            Predicted action as numpy array
        """
        # Handle deterministic flag (diffusion is inherently stochastic)
        if deterministic is None:
            deterministic = self.deterministic
        
        # Convert observation to tensor
        if isinstance(observation, np.ndarray):
            obs_tensor = torch.FloatTensor(observation).to(self.device)
        else:
            obs_tensor = observation.to(self.device)
        
        # Ensure batch dimension
        if len(obs_tensor.shape) == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
            single_obs = True
        else:
            single_obs = False
        
        # Get action from policy
        with torch.no_grad():
            action_tensor = self.policy.get_action(obs_tensor, deterministic=deterministic)
        
        # Convert to numpy
        action = action_tensor.cpu().numpy()
        
        # Remove batch dimension if input was single observation
        if single_obs:
            action = action.squeeze(0)
        
        return action
    
    def get_action(
        self,
        observation: Union[np.ndarray, torch.Tensor],
        deterministic: bool = None
    ) -> np.ndarray:
        """
        Alias for predict method to match common RL interface.
        """
        return self.predict(observation, deterministic)
    
    def evaluate_on_episodes(
        self,
        env,
        n_episodes: int = 10,
        max_steps: int = 1000,
        render: bool = False,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate the policy on multiple episodes.
        
        Args:
            env: Environment to evaluate on
            n_episodes: Number of episodes to run
            max_steps: Maximum steps per episode
            render: Whether to render episodes
            verbose: Whether to print progress
        
        Returns:
            Dictionary with evaluation statistics
        """
        episode_rewards = []
        episode_lengths = []
        success_count = 0
        
        for episode in range(n_episodes):
            if verbose:
                print(f"Episode {episode + 1}/{n_episodes}", end="")
            
            # Reset environment
            reset_result = env.reset()
            if isinstance(reset_result, tuple):
                obs, _ = reset_result
            else:
                obs = reset_result
            
            episode_reward = 0
            episode_length = 0
            
            for step in range(max_steps):
                # Get action
                action = self.predict(obs, deterministic=False)
                
                # Clip action to environment bounds if needed
                if hasattr(env, 'action_space'):
                    action = np.clip(action, env.action_space.low, env.action_space.high)
                
                # Step environment
                step_result = env.step(action)
                if len(step_result) == 5:  # new gym API
                    obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:  # old gym API
                    obs, reward, done, info = step_result
                
                episode_reward += reward
                episode_length += 1
                
                if render:
                    env.render()
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # Check for success (this depends on your environment)
            if episode_reward > 0:  # Simple success criterion
                success_count += 1
            
            if verbose:
                print(f" - Reward: {episode_reward:.2f}, Length: {episode_length}")
        
        # Compute statistics
        stats = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths),
            'success_rate': success_count / n_episodes,
            'n_episodes': n_episodes
        }
        
        if verbose:
            print(f"\nEvaluation Results:")
            print(f"  Mean Reward: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
            print(f"  Mean Length: {stats['mean_length']:.1f} ± {stats['std_length']:.1f}")
            print(f"  Success Rate: {stats['success_rate']:.2f}")
            print(f"  Reward Range: [{stats['min_reward']:.2f}, {stats['max_reward']:.2f}]")
        
        return stats
    
    def save_video_demo(
        self,
        env,
        video_path: str,
        n_episodes: int = 1,
        max_steps: int = 1000,
        fps: int = 30
    ):
        """
        Save video demonstration of the policy.
        
        Args:
            env: Environment to record
            video_path: Path to save video
            n_episodes: Number of episodes to record
            max_steps: Maximum steps per episode
            fps: Video frame rate
        """
        try:
            import cv2
            import imageio
        except ImportError:
            print("OpenCV and imageio required for video recording")
            return
        
        frames = []
        
        for episode in range(n_episodes):
            print(f"Recording episode {episode + 1}/{n_episodes}")
            
            # Reset environment
            reset_result = env.reset()
            if isinstance(reset_result, tuple):
                obs, _ = reset_result
            else:
                obs = reset_result
            
            for step in range(max_steps):
                # Render and capture frame
                if hasattr(env, 'render'):
                    try:
                        # Try newer gymnasium style first
                        frame = env.render()
                        if frame is not None:
                            frames.append(frame)
                    except Exception as e:
                        # If that fails, try without capturing frames
                        print(f"Warning: Could not capture frame: {e}")
                
                # Get action and step
                action = self.predict(obs, deterministic=False)
                
                if hasattr(env, 'action_space'):
                    action = np.clip(action, env.action_space.low, env.action_space.high)
                
                step_result = env.step(action)
                if len(step_result) == 5:
                    obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:
                    obs, reward, done, info = step_result
                
                if done:
                    break
        
        # Save video
        if frames:
            imageio.mimsave(video_path, frames, fps=fps)
            print(f"Video saved to {video_path}")
        else:
            print("No frames captured for video")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            'model_path': self.model_path,
            'device': self.device,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'num_inference_steps': self.num_inference_steps,
            'model_type': 'Diffusion IL',
            'policy_params': sum(p.numel() for p in self.policy.parameters()),
            'trainable_params': sum(p.numel() for p in self.policy.parameters() if p.requires_grad)
        }


def load_diffusion_policy(
    model_path: str,
    env=None,
    device: str = 'auto',
    num_inference_steps: int = 20
) -> DiffusionPolicyWrapper:
    """
    Convenience function to load a trained diffusion policy.
    
    Args:
        model_path: Path to the trained model checkpoint
        env: Environment instance (optional)
        device: Device to use for inference
        num_inference_steps: Number of diffusion inference steps
    
    Returns:
        DiffusionPolicyWrapper instance
    """
    return DiffusionPolicyWrapper(
        model_path=model_path,
        env=env,
        device=device,
        num_inference_steps=num_inference_steps
    )


class DiffusionPolicyForExamineEnv:
    """
    Diffusion policy wrapper compatible with MyoSuite's examine_env utility.
    This class provides the interface expected by examine_env for policy evaluation.
    """
    
    def __init__(self, env, seed=None, model_path=None, env_name='myoLegWalk-v0'):
        """
        Initialize the diffusion policy for examine_env.
        
        Args:
            env: Environment instance (passed by examine_env)
            seed: Random seed (passed by examine_env)
            model_path: Path to the trained diffusion model
            env_name: Environment name
        """
        # Handle different initialization patterns
        if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'spec'):
            # Environment object passed (from examine_env)
            env_name = env.unwrapped.spec.id
            self.examine_env = env  # Store the environment from examine_env
            temp_env = env
            close_env = False
        elif isinstance(env, str):
            # Environment name passed
            env_name = env
            from myosuite.utils import gym
            temp_env = gym.make(env_name)
            self.examine_env = temp_env
            close_env = True
        else:
            # Default case
            from myosuite.utils import gym
            temp_env = gym.make(env_name)
            self.examine_env = temp_env
            close_env = True
        
        if model_path is None:
            # Get the current directory and default model path
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir, 'diffusion_checkpoints', 'final_model.pt')
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_path = model_path
        self.env_name = env_name
        
        print(f"Loading Diffusion IL policy from {model_path}")
        print(f"Environment: {env_name}")
        print(f"Using device: {self.device}")
        
        # Create temporary dummy expert data if needed
        expert_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'myo_data.pkl')
        if not os.path.exists(expert_data_path):
            print(f"Warning: Expert data not found at {expert_data_path}")
            print("Creating dummy expert data for model loading...")
            # Create dummy data in correct format
            dummy_data = {
                'qpos': np.random.randn(10, 14),
                'qvel': np.random.randn(10, 14)
            }
            os.makedirs(os.path.dirname(expert_data_path), exist_ok=True)
            import pickle
            with open(expert_data_path, 'wb') as f:
                pickle.dump(dummy_data, f)
        
        # Create diffusion algorithm with joint control
        self.diffusion_agent = DiffusionILAlgorithm(
            env=temp_env,
            expert_data_path=expert_data_path,
            device=self.device,
            use_joint_control=True  # Use joint control like in training
        )
        
        # Load trained model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        self.diffusion_agent.load_checkpoint(model_path)
        print(f"Diffusion IL policy loaded successfully!")
        print(f"Policy expects {self.diffusion_agent.env.action_space.shape[0]} joint torques")
        print(f"Original env expects {temp_env.action_space.shape[0]} muscle activations")
        
        if close_env:
            temp_env.close()
    
    def predict(self, obs, deterministic=False):
        """
        Predict action for given observation
        
        Args:
            obs: Observation from environment
            deterministic: Whether to use deterministic policy (ignored for diffusion)
        
        Returns:
            action: Muscle activation action (zeros, torques applied separately)
            None: No additional info (for compatibility)
        """
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            # Get joint torque action from policy
            joint_action = self.diffusion_agent.policy.get_action(obs_tensor)
            joint_action_np = joint_action.squeeze(0).cpu().numpy()
        
        # Apply joint torques to the environment (similar to JointTorqueEnvWrapper)
        if hasattr(self.examine_env, 'sim'):
            qfrc = np.zeros(self.examine_env.sim.model.nv)
            
            # Get joint indices from the diffusion agent's wrapper
            if hasattr(self.diffusion_agent.env, 'joint_indices'):
                joint_indices = self.diffusion_agent.env.joint_indices
                for i, (joint_idx, torque) in enumerate(zip(joint_indices, joint_action_np)):
                    if joint_idx is not None and joint_idx < len(qfrc):
                        qfrc[joint_idx] = torque
            
            # Apply the torques to the system
            self.examine_env.sim.data.qfrc_applied[:] = qfrc
        
        # Return zero muscle activations (since we're using joint torques)
        muscle_action = np.zeros(self.examine_env.action_space.shape[0])
        
        return muscle_action, None
    
    def get_action(self, obs):
        """
        Get action for MyoSuite examine_env compatibility
        
        Args:
            obs: Observation from environment
        
        Returns:
            tuple: (exploration_action, {'evaluation': deterministic_action})
        """
        # Get action (diffusion is inherently stochastic)
        action, _ = self.predict(obs, deterministic=False)
        
        # Return same action for both exploration and evaluation
        return action, {'evaluation': action} 
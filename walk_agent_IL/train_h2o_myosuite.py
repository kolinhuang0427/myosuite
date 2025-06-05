"""
Training Script for H2O Phase-based Motion Tracking with MyoSuite
Integrates H2O retargeting data with MyoSuite walking environment for policy training
"""

import argparse
import numpy as np
import torch
import os
import sys
from typing import Dict, Any

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from h2o_phase_based_training import (
    TrainingConfig, 
    train_h2o_agent, 
    H2OMotionTracker,
    H2OPPOAgent
)

# Import MyoSuite
try:
    import myosuite
    from myosuite.utils import gym
    MYOSUITE_AVAILABLE = True
except ImportError:
    print("Warning: MyoSuite not available. Using dummy environment.")
    MYOSUITE_AVAILABLE = False


class MyoSuiteH2OWrapper:
    """Wrapper to integrate H2O training with MyoSuite walking environment"""
    
    def __init__(self, env_name: str = 'myoLegWalk-v0'):
        self.env_name = env_name
        
        if MYOSUITE_AVAILABLE:
            try:
                self.env = gym.make(env_name)
                print(f"Created MyoSuite environment: {env_name}")
            except Exception as e:
                print(f"Failed to create MyoSuite environment {env_name}: {e}")
                self.env = self._create_dummy_env()
        else:
            self.env = self._create_dummy_env()
        
        # Extract environment information
        self.muscle_action_dim = self.env.action_space.shape[0]
        self.obs_dim = self.env.observation_space.shape[0]
        
        # H2O uses joint positions, MyoSuite uses muscle activations
        # We'll map from 14 joint positions to muscle activations
        self.joint_action_dim = 14  # H2O joint targets
        
        print(f"Environment: {env_name}")
        print(f"  Muscle action space: {self.muscle_action_dim}")
        print(f"  Joint action space (H2O): {self.joint_action_dim}")
        print(f"  Observation space: {self.obs_dim}")
        
        # Simple joint to muscle mapping (this is a simplified approach)
        # In practice, this would require inverse dynamics or learned mapping
        self.joint_to_muscle_mapping = self._create_joint_to_muscle_mapping()
    
    def _create_joint_to_muscle_mapping(self):
        """Create a simple mapping from joint positions to muscle activations"""
        # This is a very simplified linear mapping
        # In practice, you'd want a more sophisticated mapping based on muscle moment arms
        mapping = np.random.randn(self.muscle_action_dim, self.joint_action_dim) * 0.1
        
        # Add identity-like structure where possible
        min_dim = min(self.muscle_action_dim, self.joint_action_dim)
        identity_part = np.zeros((self.muscle_action_dim, self.joint_action_dim))
        identity_part[:min_dim, :min_dim] = np.eye(min_dim) * 0.5
        mapping += identity_part
        
        return mapping
    
    def _joint_to_muscle_action(self, joint_action: np.ndarray) -> np.ndarray:
        """Convert joint position targets to muscle activations"""
        # Ensure input is correct size
        if len(joint_action) != self.joint_action_dim:
            joint_action = np.pad(joint_action, (0, max(0, self.joint_action_dim - len(joint_action))))[:self.joint_action_dim]
        
        # Apply mapping
        muscle_action = np.dot(self.joint_to_muscle_mapping, joint_action)
        
        # Constrain to [0, 1] for muscle activations
        muscle_action = np.clip(muscle_action + 0.1, 0.0, 1.0)  # Add baseline activation
        
        return muscle_action
    
    def _create_dummy_env(self):
        """Create dummy environment for testing when MyoSuite is not available"""
        class DummyEnv:
            def __init__(self):
                # MyoLeg has approximately 80 muscle actuators
                self.action_space = type('Space', (), {'shape': (80,)})()
                # Typical observation space
                self.observation_space = type('Space', (), {'shape': (80,)})()
                self.sim = None
                
            def reset(self):
                obs = np.random.randn(80) * 0.1
                info = {}
                return obs, info
                
            def step(self, action):
                obs = np.random.randn(80) * 0.1
                reward = np.random.normal(0, 0.1)
                terminated = np.random.random() < 0.01
                truncated = False
                info = {}
                return obs, reward, terminated, truncated, info
        
        print("Using dummy environment (MyoSuite not available)")
        return DummyEnv()
    
    def reset(self):
        result = self.env.reset()
        # Handle both old (obs only) and new (obs, info) gymnasium interfaces
        if isinstance(result, tuple):
            return result[0]  # Return just the observation
        else:
            return result
    
    def step(self, action):
        # Convert joint action to muscle action
        muscle_action = self._joint_to_muscle_action(action)
        result = self.env.step(muscle_action)
        
        # Handle both old (4) and new (5) gymnasium interfaces
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            # Override termination - let H2O logic handle it
            done = False  # Always return False, let H2O tracker decide termination
            return obs, reward, done, info
        else:
            obs, reward, done, info = result
            # Override termination - let H2O logic handle it  
            done = False  # Always return False, let H2O tracker decide termination
            return obs, reward, done, info
    
    @property
    def action_space(self):
        # Return the joint action space that H2O expects
        return type('Space', (), {'shape': (self.joint_action_dim,)})()
    
    @property
    def action_dim(self):
        # Return joint action dimension for H2O
        return self.joint_action_dim
    
    @property
    def observation_space(self):
        return self.env.observation_space
    
    @property
    def sim(self):
        return getattr(self.env, 'sim', None)


def create_h2o_config(args) -> TrainingConfig:
    """Create H2O training configuration from arguments"""
    return TrainingConfig(
        # PPO hyperparameters
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        entropy_coeff=args.entropy_coeff,
        value_coeff=args.value_coeff,
        max_grad_norm=args.max_grad_norm,
        
        # Training parameters
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        update_epochs=args.update_epochs,
        total_timesteps=args.total_timesteps,
        
        # H2O specific parameters
        history_length=args.history_length,
        termination_curriculum=getattr(args, 'termination_curriculum', False),
        initial_termination_threshold=args.initial_termination_threshold,
        final_termination_threshold=args.final_termination_threshold,
        reference_state_init=getattr(args, 'reference_state_init', False),
        
        # Environment parameters
        max_episode_length=args.max_episode_length,
        device=args.device
    )


def evaluate_agent(agent: H2OPPOAgent, tracker: H2OMotionTracker, 
                  num_episodes: int = 10) -> Dict[str, float]:
    """Evaluate trained agent"""
    print(f"\nEvaluating agent over {num_episodes} episodes...")
    
    episode_rewards = []
    episode_lengths = []
    tracking_errors = []
    
    for ep in range(num_episodes):
        state = tracker.reset(random_init=True)
        episode_reward = 0
        episode_length = 0
        episode_tracking_errors = []
        
        while episode_length < tracker.config.max_episode_length:
            # Get deterministic action
            action, _ = agent.get_action(state, deterministic=True)
            
            # Step environment
            next_state, reward, done, info = tracker.step(action)
            
            # Track error
            if tracker.current_motion_idx < len(tracker.robot_trajectory):
                target_pos = tracker.robot_trajectory[tracker.current_motion_idx]
                current_pos = np.array(list(tracker.joint_pos_history)[-1])
                error = np.linalg.norm(current_pos - target_pos)
                episode_tracking_errors.append(error)
            
            state = next_state
            episode_reward += reward
            episode_length += 1
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        tracking_errors.extend(episode_tracking_errors)
        
        print(f"  Episode {ep+1}: Reward={episode_reward:.2f}, Length={episode_length}, "
              f"Avg Tracking Error={np.mean(episode_tracking_errors):.4f}")
    
    results = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
        'mean_tracking_error': np.mean(tracking_errors),
        'std_tracking_error': np.std(tracking_errors)
    }
    
    print(f"\nEvaluation Results:")
    print(f"  Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"  Mean Length: {results['mean_length']:.1f} ± {results['std_length']:.1f}")
    print(f"  Mean Tracking Error: {results['mean_tracking_error']:.4f} ± {results['std_tracking_error']:.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Train H2O agent with MyoSuite')
    
    # Environment arguments
    parser.add_argument('--env_name', type=str, default='myoLegWalk-v0',
                       help='MyoSuite environment name')
    parser.add_argument('--motion_data', type=str, 
                       default='data/h2o_retargeted_myosuite_data_fixed.npz',
                       help='Path to retargeted motion data')
    
    # Training arguments
    parser.add_argument('--total_timesteps', type=int, default=2000000,
                       help='Total training timesteps')
    parser.add_argument('--batch_size', type=int, default=2048,
                       help='Batch size for training')
    parser.add_argument('--mini_batch_size', type=int, default=256,
                       help='Mini-batch size for optimization')
    parser.add_argument('--update_epochs', type=int, default=10,
                       help='Number of optimization epochs per update')
    parser.add_argument('--max_episode_length', type=int, default=1000,
                       help='Maximum episode length')
    
    # PPO hyperparameters
    parser.add_argument('--lr_actor', type=float, default=3e-4,
                       help='Actor learning rate')
    parser.add_argument('--lr_critic', type=float, default=3e-4,
                       help='Critic learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor')
    parser.add_argument('--gae_lambda', type=float, default=0.95,
                       help='GAE lambda parameter')
    parser.add_argument('--clip_epsilon', type=float, default=0.2,
                       help='PPO clipping parameter')
    parser.add_argument('--entropy_coeff', type=float, default=0.01,
                       help='Entropy coefficient')
    parser.add_argument('--value_coeff', type=float, default=0.5,
                       help='Value function coefficient')
    parser.add_argument('--max_grad_norm', type=float, default=0.5,
                       help='Maximum gradient norm')
    
    # H2O specific arguments
    parser.add_argument('--history_length', type=int, default=5,
                       help='History length for proprioception')
    parser.add_argument('--termination_curriculum', action='store_true',
                       help='Use termination curriculum')
    parser.add_argument('--initial_termination_threshold', type=float, default=5.0,
                       help='Initial termination threshold (meters)')
    parser.add_argument('--final_termination_threshold', type=float, default=0.3,
                       help='Final termination threshold (meters)')
    parser.add_argument('--reference_state_init', action='store_true',
                       help='Use reference state initialization')
    
    # System arguments
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--save_dir', type=str, default='./h2o_results',
                       help='Directory to save results')
    
    # Evaluation arguments
    parser.add_argument('--eval_episodes', type=int, default=10,
                       help='Number of episodes for evaluation')
    parser.add_argument('--eval_only', action='store_true',
                       help='Only evaluate existing model')
    parser.add_argument('--model_path', type=str, default='h2o_agent_final.pt',
                       help='Path to model for evaluation')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("=== H2O Training with MyoSuite ===")
    print(f"Arguments: {vars(args)}")
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Check if motion data exists
    if not os.path.exists(args.motion_data):
        print(f"Error: Motion data not found at {args.motion_data}")
        print("Please run h2o_retargeting_fixed.py first to generate the retargeted data")
        return
    
    # Create environment
    env_wrapper = MyoSuiteH2OWrapper(args.env_name)
    
    # Create configuration
    config = create_h2o_config(args)
    
    if args.eval_only:
        # Evaluation only mode
        if not os.path.exists(args.model_path):
            print(f"Error: Model not found at {args.model_path}")
            return
        
        print(f"Loading model from {args.model_path}")
        
        # Load motion data
        if args.motion_data.endswith('.npz'):
            motion_data = dict(np.load(args.motion_data, allow_pickle=True))
        else:
            import pickle
            with open(args.motion_data, 'rb') as f:
                motion_data = pickle.load(f)
        
        # Create tracker
        tracker = H2OMotionTracker(env_wrapper, motion_data, config)
        
        # Create and load agent
        agent = H2OPPOAgent(
            config=config,
            state_dim=tracker.state_dim,
            action_dim=tracker.action_dim,
            privileged_dim=tracker.privileged_dim
        )
        
        # Load model weights
        checkpoint = torch.load(args.model_path, map_location=config.device)
        agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        agent.critic.load_state_dict(checkpoint['critic_state_dict'])
        
        # Evaluate
        results = evaluate_agent(agent, tracker, args.eval_episodes)
        
        # Save evaluation results
        eval_save_path = os.path.join(args.save_dir, 'evaluation_results.npz')
        np.savez(eval_save_path, **results)
        print(f"Evaluation results saved to {eval_save_path}")
        
    else:
        # Training mode
        print("Starting H2O training...")
        
        # Train agent
        agent = train_h2o_agent(
            motion_data_path=args.motion_data,
            env=env_wrapper,
            config=config
        )
        
        # Move saved files to save directory
        import shutil
        import glob
        
        # Move all generated files to save directory
        for pattern in ['h2o_agent_*.pt', 'h2o_training_curves.png']:
            for file in glob.glob(pattern):
                dest = os.path.join(args.save_dir, file)
                shutil.move(file, dest)
                print(f"Moved {file} to {dest}")
        
        # Evaluate final agent
        if args.motion_data.endswith('.npz'):
            motion_data = dict(np.load(args.motion_data, allow_pickle=True))
        else:
            import pickle
            with open(args.motion_data, 'rb') as f:
                motion_data = pickle.load(f)
        
        tracker = H2OMotionTracker(env_wrapper, motion_data, config)
        results = evaluate_agent(agent, tracker, args.eval_episodes)
        
        # Save evaluation results
        eval_save_path = os.path.join(args.save_dir, 'final_evaluation_results.npz')
        np.savez(eval_save_path, **results)
        print(f"Final evaluation results saved to {eval_save_path}")
    
    print("H2O training/evaluation completed!")


if __name__ == '__main__':
    main() 
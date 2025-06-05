#!/usr/bin/env python3
"""
H2O Policy wrapper for use with MyoSuite's examine_env utility

Usage:
    python -m myosuite.utils.examine_env --env_name myoLegWalk-v0 --policy_path walk_agent_IL.h2o_policy_wrapper.H2OPolicyWrapper --num_episodes 3
"""

import os
import sys
import torch
import numpy as np

# Add the current directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from h2o_phase_based_training import TrainingConfig, H2OMotionTracker, H2OPPOAgent
from train_h2o_myosuite import MyoSuiteH2OWrapper


class H2OPolicyWrapper:
    """
    H2O Policy wrapper compatible with MyoSuite's examine_env utility
    """
    
    def __init__(self, env=None, seed=None, model_path=None, env_name='myoLegWalk-v0'):
        # Handle different initialization patterns
        if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'spec'):
            # Environment object passed (from examine_env)
            env_name = env.unwrapped.spec.id
            self.examine_env = env  # Store the environment from examine_env
            temp_env = env
            close_env = False
            print(f"🤖 Loading H2O policy from: {env}")
        elif isinstance(env, str):
            # Environment name passed
            env_name = env
            from myosuite.utils import gym
            temp_env = gym.make(env_name)
            self.examine_env = temp_env
            close_env = True
            print(f"🤖 Loading H2O policy for env: {env_name}")
        else:
            # Default case
            from myosuite.utils import gym
            temp_env = gym.make(env_name)
            self.examine_env = temp_env
            close_env = True
            print(f"🤖 Loading H2O policy for default env: {env_name}")
        
        if model_path is None:
            model_path = os.path.join(current_dir, 'h2o_results_long', 'h2o_agent_final.pt')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_path = model_path
        self.env_name = env_name
        
        print(f"🎯 Loading H2O model from: {model_path}")
        print(f"🌍 Environment: {env_name}")
        print(f"🔧 Using device: {self.device}")
        
        # Load motion data
        motion_data_path = os.path.join(current_dir, 'data', 'h2o_retargeted_myosuite_data_fixed.npz')
        if not os.path.exists(motion_data_path):
            raise FileNotFoundError(f"Motion data not found at {motion_data_path}")
        
        print(f"📊 Loading motion data from: {motion_data_path}")
        motion_data = dict(np.load(motion_data_path, allow_pickle=True))
        
        # Create MyoSuite wrapper for H2O
        self.env_wrapper = MyoSuiteH2OWrapper(env_name)
        
        # Create configuration (use defaults)
        self.config = TrainingConfig(device=self.device)
        
        # Create motion tracker
        self.motion_tracker = H2OMotionTracker(self.env_wrapper, motion_data, self.config)
        
        # Create agent
        self.h2o_agent = H2OPPOAgent(
            config=self.config,
            state_dim=self.motion_tracker.state_dim,
            action_dim=self.motion_tracker.action_dim,
            privileged_dim=self.motion_tracker.privileged_dim
        )
        
        # Load trained model
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.h2o_agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.h2o_agent.critic.load_state_dict(checkpoint['critic_state_dict'])
        
        print(f"✅ H2O policy loaded successfully!")
        print(f"📐 State dim: {self.motion_tracker.state_dim}")
        print(f"🎬 Action dim: {self.motion_tracker.action_dim}")
        print(f"🎭 Motion trajectory: {motion_data['robot'].shape}")
        
        # Initialize episode state
        self.current_state = None
        self.episode_step = 0
        self.reset_episode()
        
        if close_env:
            temp_env.close()
    
    def reset_episode(self):
        """Reset for a new episode"""
        # Reset motion tracker
        self.current_state = self.motion_tracker.reset(random_init=True)
        self.episode_step = 0
        print(f"🔄 Episode reset - Initial phase: {self.motion_tracker.current_phase:.3f}")
    
    def predict(self, obs, deterministic=True):
        """
        Predict action for given observation
        
        Args:
            obs: Observation from environment
            deterministic: Whether to use deterministic policy (default: True)
        
        Returns:
            action: Muscle activation action
            None: No additional info (for compatibility)
        """
        # Update tracker state based on environment observation
        # For now, we'll use the tracker's internal state progression
        
        # Get joint action from H2O policy
        h2o_action, log_prob = self.h2o_agent.get_action(self.current_state, deterministic=deterministic)
        
        # Convert joint action to muscle action
        muscle_action = self.env_wrapper._joint_to_muscle_action(h2o_action)
        
        # Update tracker state (simulate stepping)
        next_state, reward, done, info = self.motion_tracker.step(h2o_action)
        self.current_state = next_state
        self.episode_step += 1
        
        # Check if we need to reset (episode ended)
        if done or self.episode_step >= self.config.max_episode_length:
            print(f"🏁 Episode ended at step {self.episode_step} (done: {done})")
            self.reset_episode()
        
        return muscle_action, None
    
    def get_action(self, obs):
        """
        Get action for MyoSuite examine_env compatibility
        
        Args:
            obs: Observation from environment
        
        Returns:
            tuple: (exploration_action, {'evaluation': deterministic_action})
        """
        # Get deterministic action for evaluation
        eval_action, _ = self.predict(obs, deterministic=True)
        
        # For H2O, we typically want deterministic behavior for demos
        exp_action = eval_action.copy()
        
        return exp_action, {'evaluation': eval_action}


# For direct usage with examine_env
H2OPolicy = H2OPolicyWrapper


def main():
    """Test the H2O policy wrapper"""
    import myosuite
    from myosuite.utils import gym
    
    # Create environment
    env = gym.make('myoLegWalk-v0')
    
    # Create policy wrapper
    policy = H2OPolicyWrapper(env=env)
    
    # Test a few steps
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]  # Handle new gymnasium interface
        
    for step in range(100):
        action, info = policy.get_action(obs)
        result = env.step(action)
        
        # Handle both old (4) and new (5) gymnasium interfaces
        if len(result) == 5:
            obs, reward, terminated, truncated, env_info = result
            done = terminated or truncated
        else:
            obs, reward, done, env_info = result
        
        if step % 20 == 0:
            print(f"Step {step}: reward={reward:.3f}, done={done}")
        
        if done:
            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]  # Handle new gymnasium interface
            print(f"Episode ended at step {step}, resetting...")
    
    env.close()
    print("H2O policy test completed!")


if __name__ == '__main__':
    main() 
#!/usr/bin/env python3
"""
VAIL Policy wrapper for use with MyoSuite's examine_env utility

Usage:
    python -m myosuite.utils.examine_env --env_name myoLegWalk-v0 --policy_path walk_agent_IL.vail_policy_wrapper.VAILPolicyWrapper --num_episodes 3
"""

import os
import sys
import torch
import numpy as np

# Add the current directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from vail_algorithm import VAILAlgorithm, JointTorqueEnvWrapper


class ExamineEnvWrapper:
    """
    Wrapper that makes the joint torque environment compatible with examine_env
    by intercepting the policy calls and handling the action space conversion
    """
    
    def __init__(self, original_env, vail_agent):
        self.original_env = original_env
        self.vail_agent = vail_agent
        self.joint_env = vail_agent.env  # This is the wrapped environment with joint control
        
        # Get joint information from the wrapper
        if hasattr(self.joint_env, 'joint_indices'):
            self.joint_indices = self.joint_env.joint_indices
            self.n_joints = self.joint_env.n_joints
        else:
            # Fallback if wrapper doesn't have joint info
            self.joint_indices = []
            self.n_joints = 14  # Default number of joints
        
    def apply_joint_torques(self, joint_torques):
        """Apply joint torques to the original environment"""
        # Apply joint torques as external forces
        # This mimics the JointTorqueEnvWrapper's step method
        qfrc = np.zeros(self.original_env.sim.model.nv)
        
        for i, (joint_idx, torque) in enumerate(zip(self.joint_indices, joint_torques)):
            if joint_idx is not None and joint_idx < len(qfrc):
                qfrc[joint_idx] = torque
        
        # Apply the torques to the system
        self.original_env.sim.data.qfrc_applied[:] = qfrc


class VAILPolicyWrapper:
    """
    VAIL Policy wrapper compatible with MyoSuite's examine_env utility
    """
    
    def __init__(self, env=None, seed=None, model_path=None, env_name='myoLegWalk-v0'):
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
            model_path = os.path.join(current_dir, 'vail_checkpoints', 'final_model.pt')
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_path = model_path
        self.env_name = env_name
        
        print(f"Loading VAIL policy from {model_path}")
        print(f"Environment: {env_name}")
        print(f"Using device: {self.device}")
        
        # Create VAIL agent with joint control (same as training)
        expert_data_path = os.path.join(current_dir, 'expert_data', f'{env_name}_expert_data.pickle')
        if not os.path.exists(expert_data_path):
            print(f"Warning: Expert data not found at {expert_data_path}")
            print("Creating dummy expert data for model loading...")
            # Create a temporary wrapped env to get the right action space
            temp_wrapped = JointTorqueEnvWrapper(temp_env)
            dummy_data = [{'observations': [np.zeros(temp_env.observation_space.shape)], 
                          'actions': [np.zeros(temp_wrapped.action_space.shape)]}]
            os.makedirs(os.path.dirname(expert_data_path), exist_ok=True)
            import pickle
            with open(expert_data_path, 'wb') as f:
                pickle.dump(dummy_data, f)
            del temp_wrapped
        
        self.vail_agent = VAILAlgorithm(
            env=temp_env,
            expert_data_path=expert_data_path,
            device=self.device,
            use_joint_control=True  # Explicitly use joint control
        )
        
        # Create wrapper to handle joint torque application
        self.env_wrapper = ExamineEnvWrapper(self.examine_env, self.vail_agent)
        
        # Load trained model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        self.vail_agent.load_checkpoint(model_path)
        print(f"VAIL policy loaded successfully!")
        print(f"Policy expects {self.vail_agent.env.action_space.shape[0]} joint torques")
        print(f"Original env expects {temp_env.action_space.shape[0]} muscle activations")
        
        if close_env:
            temp_env.close()
    
    def predict(self, obs, deterministic=True):
        """
        Predict action for given observation
        
        Args:
            obs: Observation from environment
            deterministic: Whether to use deterministic policy (default: True)
        
        Returns:
            action: Muscle activation action (zeros, torques applied separately)
            None: No additional info (for compatibility)
        """
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            # Get joint torque action from policy
            joint_action = self.vail_agent.policy.get_action(obs_tensor, deterministic=deterministic)
            joint_action_np = joint_action.squeeze(0).cpu().numpy()
        
        # Apply joint torques to the environment
        self.env_wrapper.apply_joint_torques(joint_action_np)
        
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
        # Get deterministic action for evaluation
        eval_action, _ = self.predict(obs, deterministic=True)
        
        # Get stochastic action for exploration
        exp_action, _ = self.predict(obs, deterministic=False)
        
        return exp_action, {'evaluation': eval_action}


# Create a default instance for examine_env to use
def get_policy():
    """Factory function to create policy instance"""
    return VAILPolicyWrapper()


# For direct usage with examine_env
VAILPolicy = VAILPolicyWrapper 
import os
import sys
import argparse
import torch
import numpy as np
from myosuite.utils import gym
import myosuite
from vail_algorithm import VAILAlgorithm


def create_environment(env_name='myoLegWalk-v0', **env_kwargs):
    """Create and configure the MyoSuite environment"""
    try:
        env = gym.make(env_name, **env_kwargs)
        print(f"Successfully created environment: {env_name}")
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")
        
        # Test the environment to make sure it works
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            obs, info = reset_result
        else:
            obs = reset_result
        
        action = env.action_space.sample()
        step_result = env.step(action)
        if len(step_result) == 5:  # new gym API with truncated
            next_obs, reward, terminated, truncated, info = step_result
        else:  # old gym API
            next_obs, reward, done, info = step_result
        
        print(f"Environment test successful")
        return env
    except Exception as e:
        print(f"Failed to create environment {env_name}: {e}")
        print("Trying alternative environment names...")
        
        # Try alternative environment names
        alternatives = [
            'myoLegWalk-v0',
            'myoChaseTagP1-v0',
            'myoChaseTagP2-v0',
            'myoRunTrack-v0'
        ]
        
        for alt_env in alternatives:
            try:
                env = gym.make(alt_env, **env_kwargs)
                print(f"Successfully created alternative environment: {alt_env}")
                print(f"Observation space: {env.observation_space}")
                print(f"Action space: {env.action_space}")
                
                # Test the environment
                reset_result = env.reset()
                if isinstance(reset_result, tuple):
                    obs, info = reset_result
                else:
                    obs = reset_result
                
                action = env.action_space.sample()
                step_result = env.step(action)
                if len(step_result) == 5:  # new gym API with truncated
                    next_obs, reward, terminated, truncated, info = step_result
                else:  # old gym API
                    next_obs, reward, done, info = step_result
                
                print(f"Alternative environment test successful")
                return env
            except Exception as alt_e:
                print(f"Failed to create {alt_env}: {alt_e}")
        
        raise ValueError("Could not create any suitable walking environment")


def generate_expert_data(env_name, expert_data_path, n_trajectories=20, max_steps=1000):
    """Generate expert demonstration data if it doesn't exist"""
    if os.path.exists(expert_data_path):
        print(f"Expert data already exists at {expert_data_path}")
        return
    
    print(f"Generating expert data for {env_name}...")
    
    # Create expert data collector
    collector = ExpertDataCollector(env_name)
    
    # Collect expert trajectories using the simple controller
    trajectories = collector.collect_expert_trajectories_with_controller(
        n_trajectories=n_trajectories,
        max_steps_per_traj=max_steps,
        save_path=expert_data_path
    )
    
    print(f"Expert data generation completed and saved to {expert_data_path}")


def train_vail_agent(args):
    """Train VAIL agent for walking with joint torque control"""
    print("=" * 60)
    print("VAIL Training for MyoSuite Walking Task (Joint Torque Control)")
    print("=" * 60)
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create environment
    env = create_environment(args.env_name)
    
    # Use myo_data.pkl as expert data
    expert_data_path = args.expert_data_path
    if not os.path.exists(expert_data_path):
        raise FileNotFoundError(f"Expert data not found at {expert_data_path}")
    
    print(f"Using expert data from: {expert_data_path}")
    
    # Initialize VAIL algorithm with joint control
    vail_agent = VAILAlgorithm(
        env=env,
        expert_data_path=expert_data_path,
        policy_lr=args.policy_lr,
        discriminator_lr=args.discriminator_lr,
        value_lr=args.value_lr,
        gamma=args.gamma,
        lambda_gae=args.lambda_gae,
        clip_epsilon=args.clip_epsilon,
        entropy_coeff=args.entropy_coeff,
        vf_coeff=args.vf_coeff,
        max_grad_norm=args.max_grad_norm,
        n_policy_epochs=args.n_policy_epochs,
        n_discriminator_epochs=args.n_discriminator_epochs,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        device=args.device,
        use_joint_control=True  # Enable joint torque control
    )
    
    print(f"VAIL agent initialized with:")
    print(f"  Device: {args.device}")
    print(f"  Policy LR: {args.policy_lr}")
    print(f"  Discriminator LR: {args.discriminator_lr}")
    print(f"  Value LR: {args.value_lr}")
    print(f"  Buffer size: {args.buffer_size}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Joint control: Enabled")
    
    # Train the agent
    print("\nStarting VAIL training...")
    vail_agent.train(
        total_timesteps=args.total_timesteps,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        save_dir=args.checkpoint_dir
    )
    
    # Evaluate the trained agent
    print("\nEvaluating trained agent...")
    mean_reward, std_reward = vail_agent.evaluate(
        n_episodes=args.eval_episodes,
        render=args.render_eval
    )
    
    # Save final model
    final_model_path = os.path.join(args.checkpoint_dir, 'final_model.pt')
    vail_agent.save_checkpoint(final_model_path)
    
    print(f"\nTraining completed!")
    print(f"Final evaluation: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Final model saved to: {final_model_path}")


def evaluate_trained_agent(args):
    """Evaluate a pre-trained VAIL agent"""
    print("=" * 60)
    print("VAIL Agent Evaluation (Joint Torque Control)")
    print("=" * 60)
    
    # Create environment
    env = create_environment(args.env_name)
    
    # Create VAIL agent (structure only)
    expert_data_path = args.expert_data_path
    vail_agent = VAILAlgorithm(
        env=env,
        expert_data_path=expert_data_path,
        device=args.device,
        use_joint_control=True
    )
    
    # Load trained model
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found at {args.model_path}")
    
    vail_agent.load_checkpoint(args.model_path)
    
    # Evaluate
    print(f"Evaluating model from {args.model_path}")
    mean_reward, std_reward = vail_agent.evaluate(
        n_episodes=args.eval_episodes,
        render=args.render_eval
    )
    
    print(f"Evaluation results:")
    print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")


def main():
    parser = argparse.ArgumentParser(description='Train VAIL agent for walking with joint torque control')
    
    # Training arguments
    parser.add_argument('--mode', type=str, choices=['train', 'eval'], default='train',
                       help='Training or evaluation mode')
    parser.add_argument('--env_name', type=str, default='myoLegWalk-v0',
                       help='Environment name')
    parser.add_argument('--expert_data_path', type=str, 
                       default='./data/myo_data.pkl',
                       help='Path to expert data (myo_data.pkl)')
    
    # Model arguments
    parser.add_argument('--policy_lr', type=float, default=3e-4,
                       help='Policy learning rate')
    parser.add_argument('--discriminator_lr', type=float, default=3e-4,
                       help='Discriminator learning rate')
    parser.add_argument('--value_lr', type=float, default=3e-4,
                       help='Value function learning rate')
    
    # PPO/GAE arguments
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor')
    parser.add_argument('--lambda_gae', type=float, default=0.95,
                       help='GAE lambda parameter')
    parser.add_argument('--clip_epsilon', type=float, default=0.2,
                       help='PPO clipping parameter')
    parser.add_argument('--entropy_coeff', type=float, default=0.01,
                       help='Entropy coefficient')
    parser.add_argument('--vf_coeff', type=float, default=0.5,
                       help='Value function coefficient')
    parser.add_argument('--max_grad_norm', type=float, default=0.5,
                       help='Maximum gradient norm')
    
    # Training parameters
    parser.add_argument('--total_timesteps', type=int, default=2000000,
                       help='Total training timesteps')
    parser.add_argument('--buffer_size', type=int, default=2048,
                       help='Experience buffer size')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for updates')
    parser.add_argument('--n_policy_epochs', type=int, default=10,
                       help='Number of policy update epochs')
    parser.add_argument('--n_discriminator_epochs', type=int, default=5,
                       help='Number of discriminator update epochs')
    
    # Logging and saving
    parser.add_argument('--log_interval', type=int, default=50,
                       help='Log interval (episodes)')
    parser.add_argument('--save_interval', type=int, default=1000,
                       help='Save interval (episodes)')
    parser.add_argument('--checkpoint_dir', type=str, default='./vail_checkpoints',
                       help='Directory to save checkpoints')
    
    # Evaluation arguments
    parser.add_argument('--model_path', type=str, 
                       default='./vail_checkpoints/final_model.pt',
                       help='Path to trained model for evaluation')
    parser.add_argument('--eval_episodes', type=int, default=10,
                       help='Number of episodes for evaluation')
    parser.add_argument('--render_eval', action='store_true',
                       help='Render during evaluation')
    
    # System arguments
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {args.device}")
    
    # Run training or evaluation
    if args.mode == 'train':
        train_vail_agent(args)
    else:
        evaluate_trained_agent(args)


if __name__ == "__main__":
    main() 
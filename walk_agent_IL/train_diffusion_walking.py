import os
import sys
import argparse
import torch
import numpy as np
from myosuite.utils import gym
import myosuite
from diffusion_algorithm import DiffusionILAlgorithm


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


def train_diffusion_agent(args):
    """Train Diffusion IL agent for walking with joint torque control"""
    print("=" * 60)
    print("Diffusion IL Training for MyoSuite Walking Task (Joint Torque Control)")
    print("=" * 60)
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create environment
    env = create_environment(args.env_name)
    
    # Check expert data
    expert_data_path = args.expert_data_path
    if not os.path.exists(expert_data_path):
        raise FileNotFoundError(f"Expert data not found at {expert_data_path}")
    
    print(f"Using expert data from: {expert_data_path}")
    
    # Initialize Diffusion IL algorithm with joint control
    diffusion_agent = DiffusionILAlgorithm(
        env=env,
        expert_data_path=expert_data_path,
        policy_lr=args.policy_lr,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        num_train_timesteps=args.num_train_timesteps,
        num_inference_steps=args.num_inference_steps,
        action_scale=args.action_scale,
        device=args.device,
        use_joint_control=True,  # Enable joint torque control
        hidden_dims=args.hidden_dims,
        time_dim=args.time_dim
    )
    
    print(f"Diffusion IL agent initialized with:")
    print(f"  Device: {args.device}")
    print(f"  Policy LR: {args.policy_lr}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Buffer size: {args.buffer_size}")
    print(f"  Num train timesteps: {args.num_train_timesteps}")
    print(f"  Num inference steps: {args.num_inference_steps}")
    print(f"  Action scale: {args.action_scale}")
    print(f"  Joint control: Enabled")
    
    # Train the agent
    print("\nStarting Diffusion IL training...")
    diffusion_agent.train(
        total_timesteps=args.total_timesteps,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        save_dir=args.checkpoint_dir
    )
    
    # Evaluate the trained agent
    print("\nEvaluating trained agent...")
    mean_reward, std_reward = diffusion_agent.evaluate(
        n_episodes=args.eval_episodes,
        render=args.render_eval
    )
    
    # Save final model
    final_model_path = os.path.join(args.checkpoint_dir, 'final_model.pt')
    diffusion_agent.save_checkpoint(final_model_path)
    
    print(f"\nTraining completed!")
    print(f"Final evaluation: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Final model saved to: {final_model_path}")


def evaluate_trained_agent(args):
    """Evaluate a pre-trained Diffusion IL agent"""
    print("=" * 60)
    print("Diffusion IL Agent Evaluation (Joint Torque Control)")
    print("=" * 60)
    
    # Create environment
    env = create_environment(args.env_name)
    
    # Create Diffusion IL agent (structure only)
    expert_data_path = args.expert_data_path
    diffusion_agent = DiffusionILAlgorithm(
        env=env,
        expert_data_path=expert_data_path,
        device=args.device,
        use_joint_control=True,
        hidden_dims=args.hidden_dims,
        time_dim=args.time_dim,
        num_inference_steps=args.num_inference_steps
    )
    
    # Load trained model
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found at {args.model_path}")
    
    diffusion_agent.load_checkpoint(args.model_path)
    
    # Evaluate
    print(f"Evaluating model from {args.model_path}")
    mean_reward, std_reward = diffusion_agent.evaluate(
        n_episodes=args.eval_episodes,
        render=args.render_eval
    )
    
    print(f"Evaluation results: {mean_reward:.2f} ± {std_reward:.2f}")


def demo_diffusion_agent(args):
    """Demo a pre-trained Diffusion IL agent with rendering"""
    print("=" * 60)
    print("Diffusion IL Agent Demo (Joint Torque Control)")
    print("=" * 60)
    
    # Create environment
    env = create_environment(args.env_name)
    
    # Create Diffusion IL agent
    expert_data_path = args.expert_data_path
    diffusion_agent = DiffusionILAlgorithm(
        env=env,
        expert_data_path=expert_data_path,
        device=args.device,
        use_joint_control=True,
        hidden_dims=args.hidden_dims,
        time_dim=args.time_dim,
        num_inference_steps=args.num_inference_steps
    )
    
    # Load trained model
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found at {args.model_path}")
    
    diffusion_agent.load_checkpoint(args.model_path)
    
    # Run demo episodes
    print(f"Running {args.demo_episodes} demo episodes...")
    
    for episode in range(args.demo_episodes):
        print(f"\nEpisode {episode + 1}/{args.demo_episodes}")
        
        trajectory = diffusion_agent.collect_trajectory(max_steps=args.max_episode_steps)
        
        print(f"Episode reward: {trajectory['total_reward']:.2f}")
        print(f"Episode length: {len(trajectory['observations'])}")
        
        # Optionally save video or render
        if args.render_eval:
            print("Rendering episode...")
            # Note: Rendering setup depends on your environment configuration
    
    print("Demo completed!")


def main():
    """Main function to parse arguments and run training or evaluation"""
    parser = argparse.ArgumentParser(description='Diffusion IL for MyoSuite Walking')
    
    # Mode selection
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'eval', 'demo'],
                       help='Mode: train, eval, or demo')
    
    # Environment settings
    parser.add_argument('--env_name', type=str, default='myoLegWalk-v0',
                       help='Environment name')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # Data settings
    parser.add_argument('--expert_data_path', type=str, 
                       default='./data/h2o_retargeted_myo_data_fixed.pkl',
                       help='Path to expert demonstration data')
    
    # Model settings
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[256, 256, 256],
                       help='Hidden layer dimensions for diffusion model')
    parser.add_argument('--time_dim', type=int, default=32,
                       help='Time embedding dimension')
    parser.add_argument('--num_train_timesteps', type=int, default=200,
                       help='Number of diffusion training timesteps')
    parser.add_argument('--num_inference_steps', type=int, default=50,
                       help='Number of diffusion inference steps')
    parser.add_argument('--action_scale', type=float, default=3.0,
                       help='Action scale for noise initialization')
    
    # Training hyperparameters
    parser.add_argument('--policy_lr', type=float, default=3e-4,
                       help='Policy learning rate')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for training')
    parser.add_argument('--buffer_size', type=int, default=10000,
                       help='Buffer size for data storage')
    parser.add_argument('--total_timesteps', type=int, default=50000,
                       help='Total training timesteps')
    
    # Logging and evaluation
    parser.add_argument('--log_interval', type=int, default=200,
                       help='Logging interval')
    parser.add_argument('--eval_interval', type=int, default=2000,
                       help='Evaluation interval')
    parser.add_argument('--save_interval', type=int, default=50000,
                       help='Model saving interval')
    parser.add_argument('--eval_episodes', type=int, default=10,
                       help='Number of episodes for evaluation')
    parser.add_argument('--render_eval', action='store_true',
                       help='Render during evaluation')
    
    # Demo settings
    parser.add_argument('--demo_episodes', type=int, default=5,
                       help='Number of demo episodes')
    parser.add_argument('--max_episode_steps', type=int, default=1000,
                       help='Maximum steps per episode')
    
    # File paths
    parser.add_argument('--checkpoint_dir', type=str, default='./diffusion_checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--model_path', type=str, default='./diffusion_checkpoints/final_model.pt',
                       help='Path to trained model for evaluation/demo')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use for training')
    
    args = parser.parse_args()
    
    # Print configuration
    print("Configuration:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print()
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Run based on mode
    if args.mode == 'train':
        train_diffusion_agent(args)
    elif args.mode == 'eval':
        evaluate_trained_agent(args)
    elif args.mode == 'demo':
        demo_diffusion_agent(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Simple demo script for the trained VAIL walking agent
"""

import os
import sys
import subprocess
import argparse

def run_vail_demo(model_path, env_name, episodes=3, headless=False):
    """Run VAIL agent demonstration"""
    
    print("=" * 60)
    print("🚶 VAIL Walking Agent Demonstration")
    print("=" * 60)
    
    print(f"Model: {model_path}")
    print(f"Environment: {env_name}")
    print(f"Episodes: {episodes}")
    print(f"Rendering: {'offscreen' if headless else 'onscreen'}")
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return False
    
    print("\n🔄 Running MyoSuite visualization...")
    
    # Get the parent directory to run from myosuite root
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    
    # Construct the command
    cmd = [
        "python", "-m", "myosuite.utils.examine_env",
        "--env_name", env_name,
        "--policy_path", "walk_agent_IL.vail_policy_wrapper.VAILPolicyWrapper",
        "--num_episodes", str(episodes),
        "--render", "offscreen" if headless else "onscreen"
    ]
    
    try:
        # Run the visualization
        result = subprocess.run(cmd, cwd=parent_dir, capture_output=True, text=True)
        
        print("\n📊 Results:")
        if "Average success over rollouts:" in result.stdout:
            # Extract success rate
            lines = result.stdout.split('\n')
            for line in lines:
                if "Average success over rollouts:" in line:
                    success_rate = line.split(':')[1].strip()
                    print(f"✅ Success Rate: {success_rate}")
                    break
        
        if result.returncode == 0:
            print("✅ Demonstration completed successfully!")
            return True
        else:
            print("⚠️  Some issues occurred during visualization")
            if result.stderr:
                print(f"Errors: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error running demonstration: {e}")
        return False


def show_training_summary():
    """Show a summary of the training results"""
    
    print("\n" + "=" * 60)
    print("🎯 VAIL Training Summary")
    print("=" * 60)
    
    # Check for checkpoints
    checkpoint_dir = "./vail_checkpoints"
    if os.path.exists(checkpoint_dir):
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
        print(f"📁 Saved Models: {len(checkpoints)} checkpoints")
        for checkpoint in sorted(checkpoints):
            size_mb = os.path.getsize(os.path.join(checkpoint_dir, checkpoint)) / (1024*1024)
            print(f"   • {checkpoint} ({size_mb:.1f} MB)")
    
    # Check for expert data
    expert_dir = "./expert_data"
    if os.path.exists(expert_dir):
        expert_files = [f for f in os.listdir(expert_dir) if f.endswith('.pickle')]
        print(f"📊 Expert Data: {len(expert_files)} files")
        for expert_file in expert_files:
            size_mb = os.path.getsize(os.path.join(expert_dir, expert_file)) / (1024*1024)
            print(f"   • {expert_file} ({size_mb:.1f} MB)")
    
    print("\n🧠 Algorithm: VAIL (Variational Adversarial Imitation Learning)")
    print("🏃 Task: Biped Walking in MyoSuite")
    print("🎮 Environment: myoLegWalk-v0 (403 obs, 80 actions)")
    print("🔬 Framework: PyTorch + MyoSuite")


def main():
    parser = argparse.ArgumentParser(description='VAIL Walking Agent Demo')
    
    parser.add_argument('--model_path', type=str, 
                       default='./vail_checkpoints/final_model.pt',
                       help='Path to trained model')
    parser.add_argument('--env_name', type=str, 
                       default='myoLegWalk-v0',
                       help='Environment name')
    parser.add_argument('--episodes', type=int, default=3,
                       help='Number of episodes to demonstrate')
    parser.add_argument('--summary_only', action='store_true',
                       help='Show training summary only (no visualization)')
    parser.add_argument('--headless', action='store_true',
                       help='Run without onscreen rendering')
    
    args = parser.parse_args()
    
    # Change to the script directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Show training summary
    show_training_summary()
    
    if not args.summary_only:
        print(f"\n🎬 Starting demonstration...")
        success = run_vail_demo(args.model_path, args.env_name, args.episodes, args.headless)
        
        if success:
            print("\n🎉 Demo completed successfully!")
            print("\nTo run again:")
            print(f"  python demo_vail_agent.py --episodes {args.episodes} --headless")
        else:
            print("\n❌ Demo encountered issues")
    else:
        print("\n✅ Summary complete (use --help for demo options)")


if __name__ == "__main__":
    main() 
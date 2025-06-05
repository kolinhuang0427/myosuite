#!/usr/bin/env python3
"""
Simple video creation script for Diffusion IL walking agent
"""

import os
import sys
import subprocess

def create_diffusion_video(model_path=None, output_dir='./videos', episodes=1):
    """Create video of Diffusion IL agent using examine_env"""
    
    if model_path is None:
        model_path = './diffusion_checkpoints/final_model.pt'
    
    print("🎬 Creating Diffusion IL Walking Video")
    print(f"Model: {model_path}")
    print(f"Episodes: {episodes}")
    print(f"Output: {output_dir}")
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return False
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get parent directory for running examine_env
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    
    # Set environment for software rendering (more stable)
    env = os.environ.copy()
    env['MUJOCO_GL'] = 'osmesa'
    env['PYOPENGL_PLATFORM'] = 'osmesa'
    
    # Run examine_env with video recording
    cmd = [
        'python', '-m', 'myosuite.utils.examine_env',
        '--env_name', 'myoLegWalk-v0',
        '--policy_path', 'walk_agent_IL.diffusion_policy_wrapper.DiffusionPolicyForExamineEnv',
        '--num_episodes', str(episodes),
        '--render', 'offscreen',
        '--output_dir', output_dir,
        '--output_name', 'diffusion_walking'
    ]
    
    print(f"🎮 Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, cwd=parent_dir, env=env, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Video created successfully!")
            
            # List created videos
            import glob
            videos = glob.glob(os.path.join(output_dir, "*.mp4"))
            if videos:
                print(f"📹 Videos created:")
                for video in videos:
                    size_mb = os.path.getsize(video) / (1024 * 1024)
                    print(f"   📄 {video} ({size_mb:.1f} MB)")
            
            return True
        else:
            print("❌ Video creation failed")
            if result.stderr:
                print(f"Error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create Diffusion IL walking video')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to model (default: ./diffusion_checkpoints/final_model.pt)')
    parser.add_argument('--output', type=str, default='./videos',
                       help='Output directory (default: ./videos)')
    parser.add_argument('--episodes', type=int, default=1,
                       help='Number of episodes (default: 1)')
    
    args = parser.parse_args()
    
    success = create_diffusion_video(
        model_path=args.model,
        output_dir=args.output,
        episodes=args.episodes
    )
    
    if success:
        print("\n🎉 Done! Check the videos directory.")
    else:
        print("\n❌ Failed to create video.") 
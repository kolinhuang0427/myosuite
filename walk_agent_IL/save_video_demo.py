#!/usr/bin/env python3
"""
Video demo script for VAIL walking agent using MyoSuite's examine_env with video output
"""

import os
import sys
import subprocess
import argparse
import tempfile
import shutil
import glob

def try_alternative_rendering(model_path, env_name, episodes, output_dir, temp_dir, parent_dir):
    """Try alternative rendering methods if EGL fails"""
    
    print("🔄 Trying software rendering fallback...")
    
    # Try with software rendering
    env_software = os.environ.copy()
    env_software['MUJOCO_GL'] = 'osmesa'  # Software rendering
    env_software['PYOPENGL_PLATFORM'] = 'osmesa'
    
    cmd = [
        "python", "-m", "myosuite.utils.examine_env",
        "--env_name", env_name,
        "--policy_path", "walk_agent_IL.vail_policy_wrapper.VAILPolicyWrapper",
        "--num_episodes", str(episodes),
        "--render", "offscreen",
        "--output_dir", temp_dir,
        "--output_name", "vail_demo_software"
    ]
    
    print(f"🎮 Fallback Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, cwd=parent_dir, capture_output=True, text=True, env=env_software)
    
    return result

def save_vail_video(model_path, env_name, episodes, output_dir='./videos'):
    """Save video of VAIL agent using MyoSuite's examine_env"""
    
    print("=" * 60)
    print("🎬 VAIL Walking Agent - Video Recording")
    print("=" * 60)
    
    print(f"Model: {model_path}")
    print(f"Environment: {env_name}")
    print(f"Episodes: {episodes}")
    print(f"Output Directory: {output_dir}")
    
    # Debug display information
    print("\n🖥️  Display Environment Debug:")
    print(f"DISPLAY: {os.environ.get('DISPLAY', 'Not set')}")
    print(f"MUJOCO_GL: {os.environ.get('MUJOCO_GL', 'Not set')}")
    print(f"PYOPENGL_PLATFORM: {os.environ.get('PYOPENGL_PLATFORM', 'Not set')}")
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return False
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        print("\n🔄 Running MyoSuite visualization with video recording...")
        
        # Get the parent directory to run from myosuite root
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        
        # Set environment variables for headless rendering
        env = os.environ.copy()
        env['MUJOCO_GL'] = 'egl'  # Use EGL for headless rendering
        env['PYOPENGL_PLATFORM'] = 'egl'
        
        # Create temporary directory for examine_env output
        with tempfile.TemporaryDirectory() as temp_dir:
            # Construct the command for examine_env with video output
            cmd = [
                "python", "-m", "myosuite.utils.examine_env",
                "--env_name", env_name,
                "--policy_path", "walk_agent_IL.vail_policy_wrapper.VAILPolicyWrapper",
                "--num_episodes", str(episodes),
                "--render", "onscreen",
                "--output_dir", temp_dir,
                "--output_name", "vail_demo"
            ]
            
            print(f"🎮 Command: {' '.join(cmd)}")
            
            # Run the visualization
            result = subprocess.run(cmd, cwd=parent_dir, capture_output=True, text=True, env=env)
            
            print("\n📊 Results:")
            
            # Initialize video_files list
            video_files = []
            
            if result.returncode == 0:
                print("✅ Recording completed successfully!")
                
                # Look for generated video files
                video_files = glob.glob(os.path.join(temp_dir, "*.mp4"))
                if video_files:
                    print(f"📹 Found {len(video_files)} video files")
                    
                    # Copy videos to output directory
                    for i, video_file in enumerate(video_files):
                        output_file = os.path.join(output_dir, f"vail_demo_episode_{i+1}.mp4")
                        shutil.copy2(video_file, output_file)
                        
                        file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
                        print(f"   📹 Saved: {output_file} ({file_size:.1f} MB)")
                    
                else:
                    print("⚠️  No video files generated")
                    
            else:
                print("⚠️  Issues occurred during recording")
                if result.stderr:
                    print(f"Errors: {result.stderr}")
                if result.stdout:
                    print(f"Output: {result.stdout}")
                
                # Try fallback if GLX error detected
                if "GLX" in result.stderr or "GLXBadContext" in result.stderr:
                    print("\n🔄 GLX error detected, trying software rendering fallback...")
                    result = try_alternative_rendering(model_path, env_name, episodes, output_dir, temp_dir, parent_dir)
                    
                    if result.returncode == 0:
                        print("✅ Software rendering completed successfully!")
                        
                        # Look for generated video files from fallback
                        video_files = glob.glob(os.path.join(temp_dir, "*.mp4"))
                        if video_files:
                            print(f"📹 Found {len(video_files)} video files")
                            
                            # Copy videos to output directory
                            for i, video_file in enumerate(video_files):
                                output_file = os.path.join(output_dir, f"vail_demo_episode_{i+1}.mp4")
                                shutil.copy2(video_file, output_file)
                                
                                file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
                                print(f"   📹 Saved: {output_file} ({file_size:.1f} MB)")
                        else:
                            print("⚠️  No video files generated with software rendering")
                    else:
                        print("❌ Software rendering also failed")
                        if result.stderr:
                            print(f"Fallback Errors: {result.stderr}")
                        if result.stdout:
                            print(f"Fallback Output: {result.stdout}")
            
            # Even if no videos, check for other output files
            all_files = glob.glob(os.path.join(temp_dir, "*"))
            if all_files:
                print(f"\n📁 Files generated in temp directory:")
                for file_path in all_files:
                    file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                    print(f"   📄 {os.path.basename(file_path)} ({file_size:.1f} MB)")
            
            return len(video_files) > 0
            
    except Exception as e:
        print(f"❌ Error during video recording: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='VAIL Video Demo using MyoSuite examine_env')
    
    parser.add_argument('--model_path', type=str, 
                       default='./vail_checkpoints/final_model.pt',
                       help='Path to trained model')
    parser.add_argument('--env_name', type=str, 
                       default='myoLegWalk-v0',
                       help='Environment name')
    parser.add_argument('--episodes', type=int, default=1,
                       help='Number of episodes to record')
    parser.add_argument('--output_dir', type=str, default='./videos',
                       help='Directory to save videos')
    
    args = parser.parse_args()
    
    # Change to the script directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    print("🚀 Starting VAIL Video Recording...")
    success = save_vail_video(
        args.model_path, 
        args.env_name, 
        args.episodes,
        args.output_dir
    )
    
    if success:
        print("\n🎉 Video recording completed successfully!")
        print(f"\nVideos saved in: {args.output_dir}")
    else:
        print("\n❌ Video recording encountered issues")


if __name__ == "__main__":
    main() 
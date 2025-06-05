#!/usr/bin/env python3
"""
Visualize sample gait cycle data on MyoSkeleton
Creates a video of the sample gait motion on the MyoLeg model
"""

import os
import sys
import numpy as np
import pandas as pd
from myosuite.utils import gym
import cv2
import argparse
import mujoco

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from IPython.display import HTML
from base64 import b64encode
 
# def show_video(video_path, video_width = 400):
   
#   video_file = open(video_path, "r+b").read()
 
#   video_url = f"data:video/mp4;base64,{b64encode(video_file).decode()}"
#   return HTML(f"""<video autoplay width={video_width} controls><source src="{video_url}"></video>""")

# env = gym.make('myoElbowPose1D6MRandom-v0')
# print('List of cameras available', [env.sim.model.camera(i).name for i in range(env.sim.model.ncam)])
# env.reset()
# frames = []
# for _ in range(100):
#     frame = env.sim.renderer.render_offscreen(
#                         width=400,
#                         height=400,
#                         camera_id=0)
#     frames.append(frame)
#     env.step(env.action_space.sample()) # take a random action
# env.close()

# os.makedirs('videos', exist_ok=True)
# # make a local copy
# # Save as MP4 using OpenCV
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter('videos/temp.mp4', fourcc, 30.0, (400, 400))
# for frame in frames:
#     # Convert RGB to BGR for OpenCV
#     frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
#     out.write(frame_bgr)
# out.release()

# # show in the notebook
# show_video('videos/temp.mp4')

#end of sample code

class SampleGaitPlayer:
    """
    Plays sample gait cycle data on the MyoSkeleton model
    """
    
    def __init__(self, gait_data_path, env_name='myoLegWalk-v0'):
        """
        Initialize the gait player
        
        Args:
            gait_data_path: Path to the sample gait cycle CSV file
            env_name: MyoSuite environment name
        """
        self.env_name = env_name
        self.gait_data_path = gait_data_path
        
        # Load and process gait data
        self.load_gait_data()
        
        # Initialize environment
        self.setup_environment()
        
        # Map CSV columns to model joint names
        self.setup_joint_mapping()
        
    def load_gait_data(self):
        """Load gait cycle data from CSV file"""
        print(f"📊 Loading gait data from {self.gait_data_path}")
        
        # Load CSV data
        self.gait_df = pd.read_csv(self.gait_data_path)
        print(f"✅ Loaded {len(self.gait_df)} frames of gait data")
        
        # Print available columns
        print(f"📋 Available data columns: {list(self.gait_df.columns)}")
        
    def setup_environment(self):
        """Setup MyoSuite environment"""
        print(f"🏃 Setting up {self.env_name} environment")
        
        self.env = gym.make(self.env_name)
        self.env.reset()
        
        # Get model and extract joint information properly
        self.model = self.env.unwrapped.sim.model
        self.data = self.env.unwrapped.sim.data
        
        # Get joint names using the correct MyoSuite approach
        self.model_joint_names = []
        for i in range(self.model.njnt):
            try:
                # Use the mujoco naming system properly
                joint_id = i
                # Get the joint name using MuJoCo's naming convention
                joint_addr = self.model.name_jntadr[joint_id]
                joint_name = self.model.names[joint_addr:joint_addr+50].decode('utf-8').split('\x00')[0]
                self.model_joint_names.append(joint_name)
            except:
                self.model_joint_names.append(f"joint_{i}")
        
        print(f"🦴 Model joints ({len(self.model_joint_names)}): {self.model_joint_names}")
        
        # Get DOF information
        self.nq = self.model.nq  # Number of position coordinates
        self.nv = self.model.nv  # Number of velocity coordinates
        print(f"📐 Model DOF: nq={self.nq}, nv={self.nv}")
        
    def setup_joint_mapping(self):
        """Map CSV columns to model joint indices"""
        print("🔗 Setting up joint mapping")
        
        # Direct mapping from CSV columns to MyoSuite joint names
        # Based on the CSV file and MyoSuite model structure
        self.joint_mapping = {
            # Left leg
            'hip_flexion_l': 'hip_flexion_l',
            'hip_adduction_l': 'hip_adduction_l', 
            'hip_rotation_l': 'hip_rotation_l',
            'knee_angle_l': 'knee_angle_l',
            'ankle_angle_l': 'ankle_angle_l',
            # Right leg
            'hip_flexion_r': 'hip_flexion_r',
            'hip_adduction_r': 'hip_adduction_r',
            'hip_rotation_r': 'hip_rotation_r',
            # For right leg, CSV has 'osl_knee_angle_r' and 'osl_ankle_angle_r'
            # but model might have 'knee_angle_r' and 'ankle_angle_r'
        }
        
        # Find valid mappings using joint_name2id
        self.valid_joints = {}
        
        # Check each CSV column
        for csv_col in self.gait_df.columns:
            if csv_col in self.joint_mapping:
                model_joint = self.joint_mapping[csv_col]
                try:
                    joint_id = self.model.joint_name2id(model_joint)
                    qpos_addr = self.model.jnt_qposadr[joint_id]
                    self.valid_joints[csv_col] = qpos_addr
                    print(f"  ✅ {csv_col} -> {model_joint} (qpos index {qpos_addr})")
                except:
                    print(f"  ❌ {csv_col} -> {model_joint} (not found)")
            else:
                # Try to map directly if the CSV column name matches a joint name
                try:
                    joint_id = self.model.joint_name2id(csv_col)
                    qpos_addr = self.model.jnt_qposadr[joint_id]
                    self.valid_joints[csv_col] = qpos_addr
                    print(f"  ✅ {csv_col} -> {csv_col} (qpos index {qpos_addr})")
                except:
                    # Special handling for OSL joints
                    if csv_col == 'osl_knee_angle_r':
                        try:
                            joint_id = self.model.joint_name2id('knee_angle_r')
                            qpos_addr = self.model.jnt_qposadr[joint_id]
                            self.valid_joints[csv_col] = qpos_addr
                            print(f"  ✅ {csv_col} -> knee_angle_r (qpos index {qpos_addr})")
                        except:
                            print(f"  ❌ {csv_col} -> knee_angle_r (not found)")
                    elif csv_col == 'osl_ankle_angle_r':
                        try:
                            joint_id = self.model.joint_name2id('ankle_angle_r')
                            qpos_addr = self.model.jnt_qposadr[joint_id]
                            self.valid_joints[csv_col] = qpos_addr
                            print(f"  ✅ {csv_col} -> ankle_angle_r (qpos index {qpos_addr})")
                        except:
                            print(f"  ❌ {csv_col} -> ankle_angle_r (not found)")
        
        # Check for root joint (free joint for whole body position/orientation)
        self.root_joint_idx = None
        try:
            self.root_joint_idx = 0  # Root joint is typically at index 0
            print(f"  🏠 Root joint at qpos index 0-6 (3D pos + quaternion)")
        except:
            print("  ❌ No root joint found")
        
        # Check for pelvis orientation data
        self.has_pelvis_orientation = all(col in self.gait_df.columns for col in 
                                        ['pelvis_euler_roll', 'pelvis_euler_pitch', 'pelvis_euler_yaw'])
        if self.has_pelvis_orientation:
            print("  🧭 Found pelvis orientation data")
            
        # Check for pelvis velocity data
        self.has_pelvis_velocity = all(col in self.gait_df.columns for col in
                                     ['pelvis_vel_X', 'pelvis_vel_Y', 'pelvis_vel_Z'])
        if self.has_pelvis_velocity:
            print("  🏃 Found pelvis velocity data")
        
        print(f"🎯 Using {len(self.valid_joints)} joints for visualization")
        
    def apply_motion_frame(self, row):
        """Apply a single frame of motion data to the model"""
        # Get current state
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        
        # Set root position and orientation (first 7 elements of qpos)
        if self.root_joint_idx is not None:
            # Calculate proper pelvis height based on foot positions
            if 'l_foot_relative_Z' in row and 'r_foot_relative_Z' in row:
                l_foot_z = row['l_foot_relative_Z']
                r_foot_z = row['r_foot_relative_Z']
                
                # Find the lowest foot position (most negative Z)
                lowest_foot_z = min(l_foot_z, r_foot_z)
                
                # Calculate pelvis height so lowest foot is at ground level (Z=0)
                pelvis_height = -lowest_foot_z + 0.05  # Add small offset
            else:
                pelvis_height = 0.92  # Default standing height
            
            # Set position (first 3 elements)
            qpos[0] = 0.0  # X position
            qpos[1] = 0.0  # Y position 
            qpos[2] = pelvis_height  # Z position
            
            # Set orientation (quaternion, elements 3-6)
            if self.has_pelvis_orientation:
                # Convert Euler angles to quaternion
                roll = row['pelvis_euler_roll']
                pitch = row['pelvis_euler_pitch'] 
                yaw = row['pelvis_euler_yaw']
                
                # Convert to quaternion (w, x, y, z)
                cy = np.cos(yaw * 0.5)
                sy = np.sin(yaw * 0.5)
                cp = np.cos(pitch * 0.5)
                sp = np.sin(pitch * 0.5)
                cr = np.cos(roll * 0.5)
                sr = np.sin(roll * 0.5)
                
                w = cr * cp * cy + sr * sp * sy
                x = sr * cp * cy - cr * sp * sy
                y = cr * sp * cy + sr * cp * sy
                z = cr * cp * sy - sr * sp * cy
                
                # Set quaternion (w, x, y, z format)
                qpos[3] = w
                qpos[4] = x
                qpos[5] = y
                qpos[6] = z
            else:
                # Default upright orientation (w=1, x=y=z=0 but adjusted for MyoSuite)
                qpos[3] = 0.707388  # w
                qpos[4] = 0.0       # x
                qpos[5] = 0.0       # y
                qpos[6] = -0.706825 # z (MyoSuite default orientation)
        
        # Set joint angles from gait data
        for csv_col, qpos_idx in self.valid_joints.items():
            if csv_col in row:
                qpos[qpos_idx] = row[csv_col]
        
        # Set velocities if available
        if self.has_pelvis_velocity:
            qvel[0] = row['pelvis_vel_X']  # X velocity
            qvel[1] = row['pelvis_vel_Y']  # Y velocity
            qvel[2] = row['pelvis_vel_Z']  # Z velocity
        
        # Apply the state using MyoSuite/dm_control approach
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        
        # Forward dynamics using the MyoSuite environment's approach
        self.env.unwrapped.sim.forward()

    def create_video(self, output_file, width=640, height=480, fps=30, loops=1, 
                    playback_speed=1.0):
        """
        Create video of gait motion
        
        Args:
            output_file: Output video filename
            width, height: Video dimensions
            fps: Frames per second
            loops: Number of times to loop the gait cycle
            playback_speed: Speed multiplier
        """
        print(f"🎬 Creating video: {output_file}")
        print(f"   📐 Resolution: {width}x{height}")
        print(f"   🕐 FPS: {fps}, Loops: {loops}, Speed: {playback_speed}x")
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            output_file,
            fourcc,
            fps,
            (width, height)
        )
        
        # Calculate timing
        total_frames = len(self.gait_df) * loops
        dt = 1.0 / (fps * playback_speed)  # Time step in simulation
        
        print(f"🎞️  Rendering {total_frames} frames...")
        
        try:
            for loop in range(loops):
                for frame_idx, (_, row) in enumerate(self.gait_df.iterrows()):
                    # Apply motion frame
                    self.apply_motion_frame(row)
                    
                    # Render frame using the correct renderer
                    frame = self.env.unwrapped.sim.renderer.render_offscreen(
                        width=width, height=height, camera_id=0
                    )
                    
                    # Convert RGB to BGR for OpenCV
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    out.write(frame_bgr)
                    
                    if (frame_idx + 1) % 50 == 0:
                        print(f"   ⏳ Progress: {frame_idx + 1 + loop * len(self.gait_df)}/{total_frames} frames")
            
            print("✅ Video rendering complete!")
            
        finally:
            out.release()
            
        print(f"💾 Video saved: {output_file}")
        
    def preview_motion(self, duration=10.0, playback_speed=1.0):
        """
        Preview the motion onscreen for a limited duration
        
        Args:
            duration: Duration to preview in seconds
            playback_speed: Speed multiplier
        """
        print(f"👀 Previewing motion for {duration} seconds at {playback_speed}x speed")
        
        # Calculate timing
        gait_cycle_time = len(self.gait_df) * 0.01  # Assume 100 Hz data
        dt = gait_cycle_time / len(self.gait_df) / playback_speed
        max_frames = int(duration / dt)
        
        print(f"🔄 Gait cycle time: {gait_cycle_time:.2f}s")
        print(f"⏱️  Frame dt: {dt:.4f}s")
        
        # Enable onscreen rendering
        try:
            # Try to enable onscreen rendering
            self.env.render()
        except:
            print("⚠️  Onscreen rendering not available, using offscreen rendering")
        
        # Replay motion
        frame_count = 0
        while frame_count < max_frames:
            # Loop through gait cycle
            for frame_idx, (_, row) in enumerate(self.gait_df.iterrows()):
                if frame_count >= max_frames:
                    break
                    
                # Apply motion frame
                self.apply_motion_frame(row)
                
                # Try to render onscreen, fallback to offscreen
                try:
                    self.env.render()
                except:
                    # Fallback to offscreen rendering and display info
                    if frame_count % 30 == 0:  # Print every 30 frames (~1 second)
                        print(f"   ⏳ Preview progress: {frame_count}/{max_frames} frames")
                
                frame_count += 1
                
                # Control playback speed
                import time
                time.sleep(dt)
        
        print("✅ Preview complete!")
        
    def close(self):
        """Clean up resources"""
        if hasattr(self, 'env'):
            self.env.close()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Visualize Sample Gait Cycle Data")
    
    parser.add_argument('--gait_data', type=str, 
                       default='../myosuite/envs/myo/assets/leg/sample_gait_cycle.csv',
                       help='Path to sample gait cycle CSV file')
    parser.add_argument('--output', type=str, 
                       default='./videos/sample_gait_video.mp4',
                       help='Output video filename')
    parser.add_argument('--preview', action='store_true',
                       help='Show preview instead of creating video')
    parser.add_argument('--preview_duration', type=float, default=8.0,
                       help='Duration for preview in seconds')
    parser.add_argument('--width', type=int, default=640,
                       help='Video width')
    parser.add_argument('--height', type=int, default=480,
                       help='Video height')
    parser.add_argument('--fps', type=int, default=30,
                       help='Video FPS')
    parser.add_argument('--loops', type=int, default=3,
                       help='Number of gait cycles to loop')
    parser.add_argument('--speed', type=float, default=1.0,
                       help='Playback speed multiplier')
    
    args = parser.parse_args()
    
    print("🎭 Sample Gait Cycle Visualization")
    print("=" * 50)
    
    try:
        # Create gait player
        player = SampleGaitPlayer(
            gait_data_path=args.gait_data,
            env_name='myoLegWalk-v0'
        )
        
        if args.preview:
            # Show preview
            player.preview_motion(
                duration=args.preview_duration,
                playback_speed=args.speed
            )
        else:
            # Create video
            player.create_video(
                output_file=args.output,
                width=args.width,
                height=args.height,
                fps=args.fps,
                loops=args.loops,
                playback_speed=args.speed
            )
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    finally:
        if 'player' in locals():
            player.close()
    
    print("🎉 Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main()) 
#!/usr/bin/env python3
"""
Visualize retargeted motion data on MyoSkeleton
Creates a video of the retargeted walking motion on the MyoLeg model
"""

import os
import sys
import numpy as np
import pickle
from myosuite.utils import gym
import cv2
import argparse

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)


class RetargetedMotionPlayer:
    """
    Plays retargeted motion data on the MyoSkeleton model
    """
    
    def __init__(self, motion_data_path, env_name='myoLegWalk-v0'):
        """
        Initialize the motion player
        
        Args:
            motion_data_path: Path to the retargeted motion data
            env_name: MyoSuite environment name
        """
        self.env_name = env_name
        self.motion_data_path = motion_data_path
        
        # Load motion data
        self.load_motion_data()
        
        # Initialize environment
        self.setup_environment()
        
        # Map motion data joints to model joint indices
        self.setup_joint_mapping()
        
    def load_motion_data(self):
        """Load retargeted motion data"""
        print(f"📊 Loading motion data from {self.motion_data_path}")
        
        if self.motion_data_path.endswith('.npz'):
            self.motion_data = dict(np.load(self.motion_data_path, allow_pickle=True))
        elif self.motion_data_path.endswith('.pkl'):
            with open(self.motion_data_path, 'rb') as f:
                self.motion_data = pickle.load(f)
        else:
            raise ValueError("Motion data must be .npz or .pkl file")
        
        print(f"✅ Loaded motion data")
        print(f"🎭 Motion trajectory shape: {self.motion_data['robot'].shape}")
        print(f"⏱️  Duration: {self.motion_data['horizon'] * self.motion_data['dt']:.2f} seconds")
        print(f"🎯 Joint names: {self.motion_data['joint_names']}")
        
    def setup_environment(self):
        """Setup MyoSuite environment"""
        print(f"🏃 Setting up {self.env_name} environment")
        
        self.env = gym.make(self.env_name)
        self.env.reset()
        
        # Get model and extract joint information properly
        self.model = self.env.unwrapped.sim.model
        self.data = self.env.unwrapped.sim.data
        
        print(f"🌍 Created environment: {self.env_name}")
        print(f"🦴 Environment has {self.model.nq} position DOFs and {self.model.nv} velocity DOFs")
        
        # Get DOF information
        self.nq = self.model.nq  # Number of position coordinates
        self.nv = self.model.nv  # Number of velocity coordinates
        print(f"📐 Model DOF: nq={self.nq}, nv={self.nv}")
        
    def setup_joint_mapping(self):
        """Map motion data joints to model joint indices using proper MyoSuite approach"""
        print("🔗 Setting up joint mapping")
        
        motion_joint_names = self.motion_data['joint_names']
        
        # Find valid mappings using joint_name2id
        self.valid_joints = {}
        
        for motion_idx, joint_name in enumerate(motion_joint_names):
            try:
                # Use MyoSuite's joint_name2id to get proper joint ID
                joint_id = self.model.joint_name2id(joint_name)
                qpos_addr = self.model.jnt_qposadr[joint_id]
                self.valid_joints[motion_idx] = qpos_addr
                print(f"  ✅ {joint_name} (motion idx {motion_idx}) -> qpos index {qpos_addr}")
            except:
                print(f"  ❌ {joint_name} (motion idx {motion_idx}) -> not found in model")
        
        # Check for root joint (free joint for whole body position/orientation)
        self.has_root_joint = False
        if 'root' in motion_joint_names:
            try:
                root_joint_id = self.model.joint_name2id('root')
                self.root_qpos_start = self.model.jnt_qposadr[root_joint_id]
                self.has_root_joint = True
                print(f"  🏠 Root joint found at qpos indices {self.root_qpos_start}-{self.root_qpos_start+6}")
            except:
                print("  ❌ Root joint mentioned in data but not found in model")
        
        print(f"🎯 Using {len(self.valid_joints)} joints for visualization")
        
    def apply_motion_frame(self, motion_qpos):
        """Apply a single frame of motion data to the model"""
        # Get current state
        qpos = self.data.qpos.copy()
        
        # Set joint angles from motion data
        for motion_idx, qpos_idx in self.valid_joints.items():
            if motion_idx < len(motion_qpos):
                qpos[qpos_idx] = motion_qpos[motion_idx]
        
        # Handle root joint if present (typically first 7 elements: 3D pos + quaternion)
        if self.has_root_joint and len(motion_qpos) >= 7:
            # Set root position (first 3 elements)
            qpos[0:3] = motion_qpos[0:3]
            # Set root orientation (quaternion, elements 3-6)
            qpos[3:7] = motion_qpos[3:7]
        
        # Apply the state using MyoSuite/dm_control approach
        self.data.qpos[:] = qpos
        
        # Forward dynamics using the MyoSuite environment's approach
        self.env.unwrapped.sim.forward()
    
    def create_video(self, output_path='retargeted_motion.mp4', camera_name=None, 
                    frame_size=(640, 480), playback_speed=1.0, loop_count=3):
        """
        Create video of the retargeted motion
        
        Args:
            output_path: Output video file path
            camera_name: Camera to use for rendering (None for default)
            frame_size: Video frame size (width, height)
            playback_speed: Speed multiplier for playback
            loop_count: Number of times to loop the motion
        """
        print(f"🎬 Creating video: {output_path}")
        print(f"📷 Camera: {camera_name if camera_name else 'default'}")
        print(f"🖼️  Frame size: {frame_size}")
        print(f"⚡ Playback speed: {playback_speed}x")
        print(f"🔄 Loops: {loop_count}")
        
        # Get motion trajectory
        trajectory = self.motion_data['robot']
        dt = self.motion_data['dt'] / playback_speed
        fps = int(1.0 / dt)
        total_frames = len(trajectory) * loop_count
        
        print(f"🎞️  Total frames to render: {total_frames}")
        print(f"🎬 Video FPS: {fps}")
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            frame_size
        )
        
        try:
            # Render each frame
            for loop in range(loop_count):
                print(f"🔄 Rendering loop {loop + 1}/{loop_count}")
                
                for frame_idx, motion_qpos in enumerate(trajectory):
                    # Apply motion frame to model
                    self.apply_motion_frame(motion_qpos)
                    
                    # Render frame using the correct renderer
                    frame = self.env.unwrapped.sim.renderer.render_offscreen(
                        width=frame_size[0],
                        height=frame_size[1],
                        camera_id=camera_name
                    )
                    
                    # Convert RGB to BGR for OpenCV
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    out.write(frame_bgr)
                    
                    if frame_idx % 50 == 0:
                        print(f"  ⏳ Progress: {frame_idx + 1 + loop * len(trajectory)}/{total_frames} frames")
                        
            print("✅ Video rendering complete!")
            
        finally:
            out.release()
        
        print(f"💾 Video saved: {output_path}")
        print(f"📊 Video stats:")
        print(f"   - Duration: {total_frames / fps:.2f} seconds")
        print(f"   - Frame rate: {fps} fps")
        print(f"   - Resolution: {frame_size[0]}x{frame_size[1]}")
        
    def preview_motion(self, duration=5.0, playback_speed=1.0):
        """
        Preview the motion onscreen for a limited duration
        
        Args:
            duration: Duration to preview in seconds
            playback_speed: Speed multiplier
        """
        print(f"👀 Previewing motion for {duration} seconds at {playback_speed}x speed")
        
        # Get motion trajectory
        trajectory = self.motion_data['robot']
        dt = self.motion_data['dt'] / playback_speed
        max_frames = int(duration / dt)
        
        print(f"🔄 Motion cycle time: {len(trajectory) * dt:.2f}s")
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
            # Loop through motion cycle
            for frame_idx, motion_qpos in enumerate(trajectory):
                if frame_count >= max_frames:
                    break
                    
                # Apply motion frame
                self.apply_motion_frame(motion_qpos)
                
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
        """Close the environment"""
        self.env.close()


def main():
    """Main function to visualize retargeted motion"""
    parser = argparse.ArgumentParser(description='Visualize retargeted motion on MyoSkeleton')
    parser.add_argument('--motion_data', '-m', type=str, 
                       default='h2o_retargeted_myosuite_data_fixed.npz',
                       help='Path to retargeted motion data file')
    parser.add_argument('--output', '-o', type=str, 
                       default='../videos/retargeted_h2o_motion.mp4',
                       help='Output video file path')
    parser.add_argument('--env', '-e', type=str, 
                       default='myoLegWalk-v0',
                       help='MyoSuite environment name')
    parser.add_argument('--camera', '-c', type=str, 
                       default=None,
                       help='Camera name for rendering')
    parser.add_argument('--width', '-w', type=int, 
                       default=640,
                       help='Video width')
    parser.add_argument('--height', '-ht', type=int, 
                       default=480,
                       help='Video height')
    parser.add_argument('--speed', '-s', type=float, 
                       default=1.0,
                       help='Playback speed multiplier')
    parser.add_argument('--loops', '-l', type=int, 
                       default=1,
                       help='Number of motion loops')
    parser.add_argument('--preview', '-p', action='store_true',
                       help='Preview motion onscreen instead of creating video')
    parser.add_argument('--preview_duration', '-d', type=float, 
                       default=5.0,
                       help='Preview duration in seconds')
    
    args = parser.parse_args()
    
    # Check if motion data file exists
    motion_data_path = os.path.join(current_dir, args.motion_data)
    if not os.path.exists(motion_data_path):
        print(f"❌ Motion data file not found: {motion_data_path}")
        print("💡 Make sure to run the H2O retargeting script first!")
        return
    
    # Create motion player
    player = RetargetedMotionPlayer(motion_data_path, args.env)
    
    try:
        if args.preview:
            # Preview motion onscreen
            player.preview_motion(args.preview_duration, args.speed)
        else:
            # Create video
            output_path = os.path.join(current_dir, args.output)
            player.create_video(
                output_path=output_path,
                camera_name=args.camera,
                frame_size=(args.width, args.height),
                playback_speed=args.speed,
                loop_count=args.loops
            )
    finally:
        player.close()


if __name__ == '__main__':
    main() 
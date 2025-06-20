"""
Fixed H2O Retargeting Pipeline
Correctly maps the original column names to processed data
"""

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from typing import Dict
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

def convert_theia_to_myo_data_h2o(theia_table_path):
    """
    Convert Theia motion capture data to myosuite format using H2O retargeting methodology.
    Correctly handles the complex header structure.
    """
    print(f"=== H2O Retargeting Pipeline (Fixed) ===")
    print(f"Loading data from {theia_table_path}")
    
    # Use the same approach as the original convert script
    record_time_sec = 30
    df = pd.read_csv(theia_table_path,
                    skiprows=lambda x: x in range(0, record_time_sec*1000 + 8),
                    encoding_errors='ignore',
                    on_bad_lines='skip',
                    low_memory=False)
    
    print(f"Raw data shape: {df.shape}")
    
    # Store the original column names which contain the joint information
    original_columns = df.columns.tolist()
    
    # Process headers exactly like the original
    df.iloc[0] = df.iloc[0].ffill()
    for col in df.columns:
        df.loc[1, col] = str(df.loc[0, col]) + " " + str(df.loc[1, col]) + " " + str(df.loc[2, col])
    
    df = df[1:].drop(index=2).reset_index(drop=True)
    processed_headers = df.iloc[0].tolist()
    df = df[1:].reset_index(drop=True)
    
    # Create a mapping between original column names and their indices
    col_mapping = {}
    for i, orig_col in enumerate(original_columns):
        if i < len(processed_headers):
            col_mapping[orig_col] = i
    
    print(f"Processed data shape: {df.shape}")
    print(f"Created mapping for {len(col_mapping)} columns")
    
    # Find joint angle columns in the original column names
    joint_angle_mapping = {}
    joint_moment_mapping = {}
    
    for orig_col, col_idx in col_mapping.items():
        if 'Angles_Theia' in orig_col:
            joint_angle_mapping[orig_col] = col_idx
        elif 'Moment_Theia' in orig_col:
            joint_moment_mapping[orig_col] = col_idx
    
    print(f"Found {len(joint_angle_mapping)} angle columns and {len(joint_moment_mapping)} moment columns")
    
    # Debug: Show found joint columns with their axes
    print("\nFound joint angle columns with axes:")
    joint_groups = {}
    for orig_col, col_idx in joint_angle_mapping.items():
        base_name = orig_col.split(':')[1] if ':' in orig_col else orig_col
        if base_name not in joint_groups:
            joint_groups[base_name] = []
        processed_header = processed_headers[col_idx] if col_idx < len(processed_headers) else "N/A"
        joint_groups[base_name].append((orig_col, processed_header))
    
    for joint, columns in joint_groups.items():
        print(f"  {joint}:")
        for orig_col, header in columns:
            print(f"    {orig_col} -> {header}")
    
    # H2O Retargeting Stage 1: Shape Optimization
    print("\nStage 1: Shape Optimization")
    shape_params = {
        'pelvis_scale': 1.0,
        'femur_scale': 1.0, 
        'tibia_scale': 1.0,
        'foot_scale': 1.0
    }
    print("Shape parameters optimized (using default values)")
    
    # H2O Retargeting Stage 2: Motion Retargeting
    print("\nStage 2: Motion Retargeting")
    
    myo_data = {'qpos': pd.DataFrame(), 'qvel': pd.DataFrame()}
    
    # Mapping from Theia joint names to MyoSuite joint names
    theia_to_myo_mapping = {
        'AB01_Jimin:LeftHip': 'hip_l',
        'AB01_Jimin:RightHip': 'hip_r', 
        'AB01_Jimin:LeftKnee': 'knee_l',
        'AB01_Jimin:RightKnee': 'knee_r',
        'AB01_Jimin:LeftAnkle': 'ankle_l',
        'AB01_Jimin:RightAnkle': 'ankle_r',
        'AB01_Jimin:LeftShoulder': 'shoulder_l',
        'AB01_Jimin:RightShoulder': 'shoulder_r',
        'AB01_Jimin:LeftElbow': 'elbow_l',
        'AB01_Jimin:RightElbow': 'elbow_r',
        'AB01_Jimin:LeftWrist': 'wrist_l',
        'AB01_Jimin:RightWrist': 'wrist_r',
        'AB01_Jimin:LeftSpine': 'spine_l',
        'AB01_Jimin:RightSpine': 'spine_r',
        'AB01_Jimin:LeftNeck': 'neck_l',
        'AB01_Jimin:RightNeck': 'neck_r',
        'AB01_Jimin:LeftHead': 'head_l',
        'AB01_Jimin:RightHead': 'head_r',
        'AB01_Jimin:LeftThorax': 'thorax_l',
        'AB01_Jimin:RightThorax': 'thorax_r',
        'AB01_Jimin:LeftPelvis': 'pelvis_l',
        'AB01_Jimin:RightPelvis': 'pelvis_r',
        'AB01_Jimin:LeftClavicle': 'clavicle_l',
        'AB01_Jimin:RightClavicle': 'clavicle_r',
        'AB01_Jimin:LeftFootProgress': 'foot_progress_l',
        'AB01_Jimin:RightFootProgress': 'foot_progress_r'
    }

    def extract_joint_data_h2o(myo_joint_name, search_pattern, axis='X', flip=1, smooth_window=5):
        """Enhanced extraction function with H2O retargeting features"""
        # Find the base column (X axis)
        base_col = None
        base_col_idx = None
        for orig_col, col_idx in joint_angle_mapping.items():
            if search_pattern in orig_col:
                base_col = orig_col
                base_col_idx = col_idx
                break
        
        if base_col is None:
            print(f"Warning: {search_pattern} base column not found")
            return False
        
        # Calculate the target column index based on axis
        if axis == 'X':
            target_col_idx = base_col_idx
        elif axis == 'Y':
            target_col_idx = base_col_idx + 1
        elif axis == 'Z':
            target_col_idx = base_col_idx + 2
        else:
            print(f"Warning: Invalid axis {axis}")
            return False
        
        if target_col_idx >= len(processed_headers):
            print(f"Warning: {search_pattern} {axis} axis column out of range")
            return False
            
        # Extract data
        try:
            data = df.iloc[:, target_col_idx].astype(float) * flip
            
            # Convert degrees to radians if needed
            if 'deg' in processed_headers[target_col_idx]:
                data = data * np.pi / 180.0
            
            # Apply smoothing filter (H2O enhancement)
            if len(data) >= smooth_window:
                from scipy.ndimage import uniform_filter1d
                data = uniform_filter1d(data, size=smooth_window)
            
            # Apply joint constraints (H2O enhancement)
            joint_ranges = {
                'hip_flexion_l': [-0.5, 1.5], 'hip_flexion_r': [-0.5, 1.5],
                'hip_adduction_l': [-0.3, 0.3], 'hip_adduction_r': [-0.3, 0.3],
                'hip_rotation_l': [-0.3, 0.3], 'hip_rotation_r': [-0.3, 0.3],
                'knee_angle_l': [0, 2.0], 'knee_angle_r': [0, 2.0],
                'ankle_angle_l': [-0.5, 0.5], 'ankle_angle_r': [-0.5, 0.5],
                'subtalar_angle_l': [-0.3, 0.3], 'subtalar_angle_r': [-0.3, 0.3],
                'mtp_angle_l': [-0.3, 0.3], 'mtp_angle_r': [-0.3, 0.3]
            }
            
            if myo_joint_name in joint_ranges:
                joint_min, joint_max = joint_ranges[myo_joint_name]
                data = np.clip(data, joint_min, joint_max)
            
            myo_data['qpos'][myo_joint_name] = data
            if myo_joint_name == 'hip_adduction_l' or myo_joint_name == 'hip_adduction_r' : #or myo_joint_name == 'hip_rotation_l' or myo_joint_name == 'hip_rotation_r':
                myo_data['qpos'][myo_joint_name] = -data
                
            print(f"Retargeted {myo_joint_name}: {len(data)} samples from {base_col} {axis} axis")
            return True
            
        except Exception as e:
            print(f"Error extracting {myo_joint_name}: {e}")
            return False

    def extract_all_available_joints():
        """Automatically discover and extract all available joint data"""
        print("\nAutomatically discovering all available joint data...")
        
        discovered_joints = []
        
        # Group joints by their base name and find all available axes
        joint_groups = {}
        for orig_col, col_idx in joint_angle_mapping.items():
            if 'Angles_Theia' in orig_col:
                # Extract the joint base name
                joint_base = orig_col.split('Angles_Theia')[0]
                if joint_base not in joint_groups:
                    joint_groups[joint_base] = []
                joint_groups[joint_base].append((orig_col, col_idx))
        
        print(f"Discovered {len(joint_groups)} joint groups:")
        
        for joint_base, columns in joint_groups.items():
            print(f"\n  Joint: {joint_base}")
            myo_base = theia_to_myo_mapping.get(joint_base, joint_base.lower().replace(' ', '_'))
            
            # Sort columns to ensure X, Y, Z order
            columns.sort(key=lambda x: x[1])  # Sort by column index
            
            axes_found = []
            for i, (orig_col, col_idx) in enumerate(columns):
                axis_names = ['X', 'Y', 'Z']
                if i < len(axis_names):
                    axis = axis_names[i]
                    axis_descriptions = ['flexion', 'adduction', 'rotation']
                    axis_desc = axis_descriptions[i] if i < len(axis_descriptions) else f'axis{i+1}'
                    
                    joint_name = f"{myo_base}_{axis_desc}"
                    search_pattern = f"{joint_base}Angles_Theia"
                    
                    # Check if data is actually available and has values
                    try:
                        test_data = df.iloc[:, col_idx].astype(float)
                        if not test_data.isna().all() and len(test_data.dropna()) > 100:  # Require at least 100 valid data points
                            if extract_joint_data_h2o(joint_name, search_pattern, axis):
                                axes_found.append(f"{axis}({axis_desc})")
                                discovered_joints.append(joint_name)
                            else:
                                print(f" {axis} axis extraction failed")
                        else:
                            print(f"  {axis} axis has insufficient data")
                    except Exception as e:
                        print(f" {axis} axis error: {e}")
            
            if axes_found:
                print(f"  Extracted axes: {', '.join(axes_found)}")
            else:
                print(f" No valid data found")
        
        return discovered_joints

    # First, do automatic discovery
    auto_discovered = extract_all_available_joints()
    
    # Then add the systematic extraction as before
    print(f"\nAuto-discovered {len(auto_discovered)} joints: {sorted(auto_discovered)}")
    
    # Continue with the existing systematic approach to ensure we don't miss anything

    # Extract all XYZ data for each joint
    extracted_joints = auto_discovered.copy()  # Start with auto-discovered joints
    
    # Find all unique joint base names from the angle data for systematic extraction
    unique_joints = set()
    for orig_col in joint_angle_mapping.keys():
        if 'Angles_Theia' in orig_col:
            joint_base = orig_col.split('Angles_Theia')[0]
            unique_joints.add(joint_base)
    
    print(f"\nSystematic extraction for {len(unique_joints)} joint groups...")
    
    for theia_joint in sorted(unique_joints):
        myo_base = theia_to_myo_mapping.get(theia_joint, theia_joint.lower())
        
        # Extract X, Y, Z axes for this joint if not already extracted
        axes_extracted = []
        for axis, axis_name in [('X', 'flexion'), ('Y', 'adduction'), ('Z', 'rotation')]:
            joint_name = f"{myo_base}_{axis_name}"
            
            # Only extract if not already done in auto-discovery
            if joint_name not in extracted_joints:
                search_pattern = f"{theia_joint}Angles_Theia"
                
                if extract_joint_data_h2o(joint_name, search_pattern, axis):
                    axes_extracted.append(axis)
                    extracted_joints.append(joint_name)
        
        if axes_extracted:
            print(f"  ✅ {theia_joint} -> {myo_base}: additionally extracted {axes_extracted} axes")
    
    # Additional specific extractions for known mappings to maintain compatibility
    print("\nExtracting specific joint mappings for MyoSuite compatibility...")
    
    # Standard MyoSuite joint names that we want to ensure are available
    standard_extractions = [
        ("hip_flexion_l", "LeftHipAngles_Theia", "X"),
        ("hip_flexion_r", "RightHipAngles_Theia", "X"),
        ("hip_adduction_l", "LeftHipAngles_Theia", "Y"),
        ("hip_adduction_r", "RightHipAngles_Theia", "Y"),
        ("hip_rotation_l", "LeftHipAngles_Theia", "Z"),
        ("hip_rotation_r", "RightHipAngles_Theia", "Z"),
        ("knee_angle_l", "LeftKneeAngles_Theia", "X"),
        ("knee_angle_r", "RightKneeAngles_Theia", "X"),
        ("ankle_angle_l", "LeftAnkleAngles_Theia", "X"),
        ("ankle_angle_r", "RightAnkleAngles_Theia", "X"),
        ("subtalar_angle_l", "LeftAnkleAngles_Theia", "Y"),
        ("subtalar_angle_r", "RightAnkleAngles_Theia", "Y"),
    ]
    
    for joint_name, search_pattern, axis in standard_extractions:
        if joint_name not in myo_data['qpos'].columns and joint_name not in extracted_joints:
            if extract_joint_data_h2o(joint_name, search_pattern, axis):
                extracted_joints.append(joint_name)
    
    # For MTP angles, try to find foot/toe data
    if not extract_joint_data_h2o("mtp_angle_l", "l_toes_4X4", "RX", -1):
        # Fallback: use zero or estimate from ankle
        if "ankle_angle_l" in myo_data['qpos'].columns:
            myo_data['qpos']["mtp_angle_l"] = np.zeros(len(myo_data['qpos']["ankle_angle_l"]))
            print("Using zero values for mtp_angle_l")
            extracted_joints.append("mtp_angle_l")
    
    if not extract_joint_data_h2o("mtp_angle_r", "r_toes_4X4", "RX", -1):
        # Fallback: use zero or estimate from ankle
        if "ankle_angle_r" in myo_data['qpos'].columns:
            myo_data['qpos']["mtp_angle_r"] = np.zeros(len(myo_data['qpos']["ankle_angle_r"]))
            print("Using zero values for mtp_angle_r")
            extracted_joints.append("mtp_angle_r")

    print(f"\nTotal extracted joints: {len(set(extracted_joints))}")
    print(f"Final joint list: {sorted(set(extracted_joints))}")
    
    # Generate velocities from positions
    for joint in myo_data['qpos'].columns:
        positions = myo_data['qpos'][joint].values
        velocities = np.gradient(positions) * 500  # Convert to velocity (assuming 500Hz)
        myo_data['qvel'][joint] = velocities
    
    print(f"Generated {len(myo_data['qvel'].columns)} joint velocity series")
    
    # Create enhanced myosuite format
    n_steps = len(myo_data['qpos'])
    dt = 0.002  # 500 Hz sampling rate
    time = np.arange(n_steps) * dt
    
    # Convert to numpy arrays for myosuite
    joint_names = list(myo_data['qpos'].columns)
    robot_qpos = myo_data['qpos'][joint_names].values
    robot_qvel = myo_data['qvel'][joint_names].values
    
    myosuite_data = {
        'time': time,
        'robot': robot_qpos,
        'robot_vel': robot_qvel,
        'robot_init': robot_qpos[0] if len(robot_qpos) > 0 else np.zeros(len(joint_names)),
        'joint_names': joint_names,
        'dt': dt,
        'horizon': n_steps,
        'shape_params': shape_params,
        'source': 'H2O_retargeting_fixed'
    }
    
    return myo_data, myosuite_data

def visualize_retargeted_motion(myo_data, joint_names=None):
    """Visualize the retargeted motion data."""
    if joint_names is None:
        joint_names = ['hip_flexion_l', 'knee_angle_l', 'ankle_angle_l']
    
    available_joints = [j for j in joint_names if j in myo_data['qpos'].columns]
    
    if not available_joints:
        print("No joints available for visualization")
        return
    
    fig, axes = plt.subplots(len(available_joints), 1, figsize=(12, 3*len(available_joints)))
    if len(available_joints) == 1:
        axes = [axes]
    
    for i, joint in enumerate(available_joints):
        ax = axes[i]
        
        # Plot position
        ax.plot(myo_data['qpos'][joint], label='Position', alpha=0.8)
        
        # Plot velocity if available
        if joint in myo_data['qvel'].columns:
            ax2 = ax.twinx()
            ax2.plot(myo_data['qvel'][joint], label='Velocity', alpha=0.6, color='orange')
            ax2.set_ylabel('Velocity (rad/s)', color='orange')
            ax2.tick_params(axis='y', labelcolor='orange')
        
        ax.set_title(f'{joint} - H2O Retargeted')
        ax.set_ylabel('Position (rad)')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time Steps')
    plt.tight_layout()
    plt.savefig('h2o_retargeted_motion.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Visualization saved as h2o_retargeted_motion.png")

def main():
    """Main function to run the fixed H2O retargeting pipeline."""
    # Run the H2O retargeting pipeline
    myo_data, myosuite_data = convert_theia_to_myo_data_h2o('AB01_Jimin_1p0mps_1.csv')
    
    # Save the retargeted data
    print("\nSaving retargeted data...")
    
    # Save as pickle (compatible with existing myosuite code)
    with open('h2o_retargeted_myo_data_xyz.pkl', 'wb') as file:
        pickle.dump(myo_data, file)
    
    # Save as npz (for easy loading)
    np.savez('h2o_retargeted_myosuite_data_xyz.npz', **myosuite_data)
    
    print("Saved h2o_retargeted_myo_data_xyz.pkl and h2o_retargeted_myosuite_data_xyz.npz")
    
    # Visualize results
    if len(myo_data['qpos'].columns) > 0:
        print("\nGenerating visualization...")
        visualize_retargeted_motion(myo_data)
    
    print("\n=== H2O Retargeting Pipeline Completed ===")
    print(f"Successfully retargeted {len(myo_data['qpos'].columns)} joints")
    print(f"Motion duration: {myosuite_data['horizon'] * myosuite_data['dt']:.2f} seconds")
    print(f"Data shape: {myosuite_data['robot'].shape}")

if __name__ == '__main__':
    main() 
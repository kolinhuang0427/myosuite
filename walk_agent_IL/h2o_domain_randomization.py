"""
H2O Domain Randomization
Implements domain randomization techniques for robust policy training
Based on the H2O paper's Table VI domain randomization parameters
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import random


@dataclass
class DomainRandomizationConfig:
    """Configuration for domain randomization parameters"""
    
    # Physics randomization
    mass_scale_range: Tuple[float, float] = (0.8, 1.2)
    friction_range: Tuple[float, float] = (0.5, 1.5)
    joint_damping_range: Tuple[float, float] = (0.8, 1.2)
    joint_stiffness_range: Tuple[float, float] = (0.8, 1.2)
    
    # Motor randomization
    motor_strength_range: Tuple[float, float] = (0.8, 1.2)
    motor_delay_range: Tuple[float, float] = (0.0, 0.02)  # seconds
    
    # Sensor noise
    joint_pos_noise_std: float = 0.01  # radians
    joint_vel_noise_std: float = 0.1   # rad/s
    imu_noise_std: float = 0.05        # m/s^2 for acceleration, rad/s for gyro
    
    # External disturbances
    force_disturbance_prob: float = 0.1    # probability per step
    force_disturbance_range: Tuple[float, float] = (50.0, 200.0)  # N
    force_duration_range: Tuple[int, int] = (5, 20)  # timesteps
    
    # Ground properties
    ground_friction_range: Tuple[float, float] = (0.6, 1.4)
    ground_height_noise_std: float = 0.02  # meters
    
    # Latency and delays
    action_delay_range: Tuple[int, int] = (0, 3)  # timesteps
    observation_delay_range: Tuple[int, int] = (0, 2)  # timesteps
    
    # Enable/disable individual randomizations
    enable_physics: bool = True
    enable_motor: bool = True
    enable_sensor_noise: bool = True
    enable_disturbances: bool = True
    enable_ground: bool = True
    enable_delays: bool = True


class DomainRandomizer:
    """Domain randomization for H2O training"""
    
    def __init__(self, config: DomainRandomizationConfig, env):
        self.config = config
        self.env = env
        
        # Initialize randomization state
        self.current_physics_params = {}
        self.current_motor_params = {}
        self.force_disturbance_counter = 0
        self.force_disturbance_force = np.zeros(3)
        
        # Action and observation delay buffers
        self.action_buffer = []
        self.observation_buffer = []
        
        # Ground height map (if applicable)
        self.ground_height_map = None
        
        print("Domain randomizer initialized with config:")
        print(f"  Physics randomization: {config.enable_physics}")
        print(f"  Motor randomization: {config.enable_motor}")
        print(f"  Sensor noise: {config.enable_sensor_noise}")
        print(f"  External disturbances: {config.enable_disturbances}")
        print(f"  Ground randomization: {config.enable_ground}")
        print(f"  Action/observation delays: {config.enable_delays}")
    
    def reset_randomization(self):
        """Reset randomization parameters for new episode"""
        if self.config.enable_physics:
            self._randomize_physics()
        
        if self.config.enable_motor:
            self._randomize_motor_properties()
        
        if self.config.enable_ground:
            self._randomize_ground_properties()
        
        # Reset disturbance state
        self.force_disturbance_counter = 0
        self.force_disturbance_force = np.zeros(3)
        
        # Clear delay buffers
        self.action_buffer.clear()
        self.observation_buffer.clear()
        
        print("Domain randomization reset for new episode")
    
    def _randomize_physics(self):
        """Randomize physics parameters"""
        if not hasattr(self.env, 'sim') or self.env.sim is None:
            return
        
        # Mass scaling
        mass_scale = np.random.uniform(*self.config.mass_scale_range)
        if hasattr(self.env.sim.model, 'body_mass'):
            original_masses = self.env.sim.model.body_mass.copy()
            self.env.sim.model.body_mass[:] = original_masses * mass_scale
        
        # Joint damping
        damping_scale = np.random.uniform(*self.config.joint_damping_range)
        if hasattr(self.env.sim.model, 'dof_damping'):
            original_damping = self.env.sim.model.dof_damping.copy()
            self.env.sim.model.dof_damping[:] = original_damping * damping_scale
        
        # Joint stiffness (for position control)
        stiffness_scale = np.random.uniform(*self.config.joint_stiffness_range)
        if hasattr(self.env.sim.model, 'actuator_gainprm'):
            original_gains = self.env.sim.model.actuator_gainprm.copy()
            self.env.sim.model.actuator_gainprm[:, 0] *= stiffness_scale
        
        # Friction randomization
        friction_scale = np.random.uniform(*self.config.friction_range)
        if hasattr(self.env.sim.model, 'geom_friction'):
            original_friction = self.env.sim.model.geom_friction.copy()
            self.env.sim.model.geom_friction[:, 0] *= friction_scale
        
        self.current_physics_params = {
            'mass_scale': mass_scale,
            'damping_scale': damping_scale,
            'stiffness_scale': stiffness_scale,
            'friction_scale': friction_scale
        }
    
    def _randomize_motor_properties(self):
        """Randomize motor/actuator properties"""
        # Motor strength scaling
        strength_scale = np.random.uniform(*self.config.motor_strength_range)
        
        # Motor delay (will be applied during action processing)
        motor_delay = np.random.uniform(*self.config.motor_delay_range)
        
        self.current_motor_params = {
            'strength_scale': strength_scale,
            'motor_delay': motor_delay
        }
    
    def _randomize_ground_properties(self):
        """Randomize ground/terrain properties"""
        if not hasattr(self.env, 'sim') or self.env.sim is None:
            return
        
        # Ground friction
        ground_friction = np.random.uniform(*self.config.ground_friction_range)
        
        # Ground height variations (simplified)
        if self.config.ground_height_noise_std > 0:
            # This would typically modify heightfield data
            # For now, we'll store the noise level
            self.ground_height_noise = np.random.normal(0, self.config.ground_height_noise_std)
        
        # Find ground geoms and randomize their friction
        if hasattr(self.env.sim.model, 'geom_name2id'):
            try:
                terrain_id = self.env.sim.model.geom_name2id('terrain')
                if terrain_id < len(self.env.sim.model.geom_friction):
                    self.env.sim.model.geom_friction[terrain_id, 0] = ground_friction
            except:
                pass  # terrain geom not found
    
    def apply_sensor_noise(self, observation: np.ndarray, 
                          joint_positions: Optional[np.ndarray] = None,
                          joint_velocities: Optional[np.ndarray] = None,
                          imu_data: Optional[np.ndarray] = None) -> np.ndarray:
        """Apply sensor noise to observations"""
        if not self.config.enable_sensor_noise:
            return observation
        
        noisy_obs = observation.copy()
        
        # Joint position noise
        if joint_positions is not None and self.config.joint_pos_noise_std > 0:
            pos_noise = np.random.normal(0, self.config.joint_pos_noise_std, joint_positions.shape)
            # Assume joint positions are at the beginning of observation
            pos_dim = len(joint_positions)
            noisy_obs[:pos_dim] += pos_noise
        
        # Joint velocity noise
        if joint_velocities is not None and self.config.joint_vel_noise_std > 0:
            vel_noise = np.random.normal(0, self.config.joint_vel_noise_std, joint_velocities.shape)
            # Assume joint velocities follow joint positions
            pos_dim = len(joint_positions) if joint_positions is not None else 0
            vel_dim = len(joint_velocities)
            noisy_obs[pos_dim:pos_dim+vel_dim] += vel_noise
        
        # IMU noise (accelerometer and gyroscope)
        if imu_data is not None and self.config.imu_noise_std > 0:
            imu_noise = np.random.normal(0, self.config.imu_noise_std, imu_data.shape)
            # IMU data location depends on observation structure
            # This is a simplified implementation
            if len(noisy_obs) > len(imu_data):
                noisy_obs[-len(imu_data):] += imu_noise
        
        return noisy_obs
    
    def apply_motor_randomization(self, action: np.ndarray) -> np.ndarray:
        """Apply motor randomization to actions"""
        if not self.config.enable_motor:
            return action
        
        randomized_action = action.copy()
        
        # Apply motor strength scaling
        if 'strength_scale' in self.current_motor_params:
            randomized_action *= self.current_motor_params['strength_scale']
        
        # Motor delay is handled in the delay buffer
        
        return randomized_action
    
    def apply_external_disturbances(self, env) -> np.ndarray:
        """Apply external force disturbances"""
        if not self.config.enable_disturbances:
            return np.zeros(3)
        
        # Check if we should apply a new disturbance
        if (self.force_disturbance_counter <= 0 and 
            np.random.random() < self.config.force_disturbance_prob):
            
            # Generate random force
            force_magnitude = np.random.uniform(*self.config.force_disturbance_range)
            force_direction = np.random.randn(3)
            force_direction /= np.linalg.norm(force_direction)
            
            self.force_disturbance_force = force_direction * force_magnitude
            self.force_disturbance_counter = np.random.randint(*self.config.force_duration_range)
        
        # Apply current disturbance
        if self.force_disturbance_counter > 0:
            # Apply force to the robot (this would need environment-specific implementation)
            if hasattr(env, 'sim') and env.sim is not None:
                # In MuJoCo, you would use mj_applyFT or modify xfrc_applied
                # For now, we'll return the force for external application
                self.force_disturbance_counter -= 1
                return self.force_disturbance_force
        
        return np.zeros(3)
    
    def apply_action_delay(self, action: np.ndarray) -> np.ndarray:
        """Apply action delay using buffer"""
        if not self.config.enable_delays:
            return action
        
        # Add current action to buffer
        self.action_buffer.append(action.copy())
        
        # Determine delay
        if len(self.action_buffer) == 1:
            # First action, determine delay for this episode
            self.current_action_delay = np.random.randint(*self.config.action_delay_range)
        
        # Return delayed action or current if buffer not full
        if len(self.action_buffer) > self.current_action_delay:
            delayed_action = self.action_buffer.pop(0)
        else:
            delayed_action = action  # Use current action if buffer not full yet
        
        return delayed_action
    
    def apply_observation_delay(self, observation: np.ndarray) -> np.ndarray:
        """Apply observation delay using buffer"""
        if not self.config.enable_delays:
            return observation
        
        # Add current observation to buffer
        self.observation_buffer.append(observation.copy())
        
        # Determine delay
        if len(self.observation_buffer) == 1:
            # First observation, determine delay for this episode
            self.current_obs_delay = np.random.randint(*self.config.observation_delay_range)
        
        # Return delayed observation or current if buffer not full
        if len(self.observation_buffer) > self.current_obs_delay:
            delayed_obs = self.observation_buffer.pop(0)
        else:
            delayed_obs = observation  # Use current observation if buffer not full yet
        
        return delayed_obs
    
    def randomize_step(self, observation: np.ndarray, action: np.ndarray, 
                      env) -> Tuple[np.ndarray, np.ndarray]:
        """Apply all randomizations for a single step"""
        
        # Apply sensor noise to observation
        noisy_obs = self.apply_sensor_noise(observation)
        
        # Apply observation delay
        delayed_obs = self.apply_observation_delay(noisy_obs)
        
        # Apply motor randomization to action
        randomized_action = self.apply_motor_randomization(action)
        
        # Apply action delay
        delayed_action = self.apply_action_delay(randomized_action)
        
        # Apply external disturbances (returns force for external application)
        disturbance_force = self.apply_external_disturbances(env)
        
        return delayed_obs, delayed_action
    
    def get_randomization_info(self) -> Dict:
        """Get current randomization parameters for logging"""
        info = {
            'physics_params': self.current_physics_params,
            'motor_params': self.current_motor_params,
            'force_disturbance_active': self.force_disturbance_counter > 0,
            'force_disturbance_magnitude': np.linalg.norm(self.force_disturbance_force)
        }
        
        if hasattr(self, 'current_action_delay'):
            info['action_delay'] = self.current_action_delay
        if hasattr(self, 'current_obs_delay'):
            info['observation_delay'] = self.current_obs_delay
        
        return info


class H2ODomainRandomizedEnv:
    """Environment wrapper that applies H2O domain randomization"""
    
    def __init__(self, env, randomization_config: DomainRandomizationConfig = None):
        self.env = env
        
        if randomization_config is None:
            randomization_config = DomainRandomizationConfig()
        
        self.randomizer = DomainRandomizer(randomization_config, env)
        self.randomization_info = {}
    
    def reset(self):
        """Reset environment with new randomization"""
        obs = self.env.reset()
        
        # Apply new randomization parameters
        self.randomizer.reset_randomization()
        
        # Apply observation randomization
        obs, _ = self.randomizer.randomize_step(obs, np.zeros(self.env.action_space.shape[0]), self.env)
        
        return obs
    
    def step(self, action):
        """Step environment with domain randomization"""
        # Get current observation for randomization
        if hasattr(self.env, '_get_obs'):
            current_obs = self.env._get_obs()
        else:
            # Fallback: use previous observation (not ideal)
            current_obs = np.zeros(self.env.observation_space.shape[0])
        
        # Apply randomization
        randomized_obs, randomized_action = self.randomizer.randomize_step(
            current_obs, action, self.env
        )
        
        # Step with randomized action
        obs, reward, done, info = self.env.step(randomized_action)
        
        # Apply observation randomization to returned observation
        obs, _ = self.randomizer.randomize_step(obs, action, self.env)
        
        # Add randomization info
        self.randomization_info = self.randomizer.get_randomization_info()
        info.update({'randomization': self.randomization_info})
        
        return obs, reward, done, info
    
    @property
    def action_space(self):
        return self.env.action_space
    
    @property
    def observation_space(self):
        return self.env.observation_space
    
    @property
    def sim(self):
        return getattr(self.env, 'sim', None)


def create_h2o_domain_randomization_config() -> DomainRandomizationConfig:
    """Create H2O domain randomization configuration based on paper's Table VI"""
    return DomainRandomizationConfig(
        # Physics parameters (based on typical humanoid robot parameters)
        mass_scale_range=(0.85, 1.15),
        friction_range=(0.7, 1.3),
        joint_damping_range=(0.8, 1.2),
        joint_stiffness_range=(0.8, 1.2),
        
        # Motor parameters
        motor_strength_range=(0.9, 1.1),
        motor_delay_range=(0.0, 0.01),
        
        # Sensor noise (realistic values for IMU and encoders)
        joint_pos_noise_std=0.005,
        joint_vel_noise_std=0.05,
        imu_noise_std=0.02,
        
        # External disturbances
        force_disturbance_prob=0.05,
        force_disturbance_range=(20.0, 100.0),
        force_duration_range=(3, 15),
        
        # Ground properties
        ground_friction_range=(0.8, 1.2),
        ground_height_noise_std=0.01,
        
        # Latency (typical values for real-time systems)
        action_delay_range=(0, 2),
        observation_delay_range=(0, 1),
        
        # Enable all randomizations
        enable_physics=True,
        enable_motor=True,
        enable_sensor_noise=True,
        enable_disturbances=True,
        enable_ground=True,
        enable_delays=True
    )


def main():
    """Test domain randomization"""
    
    # Create dummy environment for testing
    class DummyEnv:
        def __init__(self):
            self.action_space = type('Space', (), {'shape': (22,)})()
            self.observation_space = type('Space', (), {'shape': (80,)})()
            self.sim = None
            
        def reset(self):
            return np.random.randn(80) * 0.1
            
        def step(self, action):
            obs = np.random.randn(80) * 0.1
            reward = 0.0
            done = False
            info = {}
            return obs, reward, done, info
    
    # Create environment and wrapper
    base_env = DummyEnv()
    config = create_h2o_domain_randomization_config()
    env = H2ODomainRandomizedEnv(base_env, config)
    
    print("Testing H2O domain randomization...")
    
    # Test episode
    obs = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    
    for step in range(10):
        action = np.random.randn(22) * 0.1
        obs, reward, done, info = env.step(action)
        
        randomization_info = info.get('randomization', {})
        print(f"Step {step}: Randomization active - {len(randomization_info)} parameters")
        
        if done:
            break
    
    print("Domain randomization test completed!")


if __name__ == '__main__':
    main() 
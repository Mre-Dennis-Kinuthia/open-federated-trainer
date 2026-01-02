"""
Client Behavior Simulation

Simulates realistic volunteer compute behavior:
- Random startup delays
- Random dropouts
- Variable training speed
- Temporary coordinator unavailability handling
"""

import os
import time
import random
from typing import Optional


class BehaviorSimulator:
    """
    Simulates realistic client behavior for federated learning.
    
    Configurable via environment variables to enable/disable behaviors.
    """
    
    def __init__(self):
        """Initialize the behavior simulator."""
        # Startup delay configuration
        self.enable_startup_delay = os.getenv("BEHAVIOR_STARTUP_DELAY", "false").lower() == "true"
        self.startup_delay_min = float(os.getenv("BEHAVIOR_STARTUP_DELAY_MIN", "0.0"))
        self.startup_delay_max = float(os.getenv("BEHAVIOR_STARTUP_DELAY_MAX", "10.0"))
        
        # Dropout configuration
        self.enable_dropouts = os.getenv("BEHAVIOR_ENABLE_DROPOUTS", "false").lower() == "true"
        self.dropout_probability = float(os.getenv("BEHAVIOR_DROPOUT_PROBABILITY", "0.1"))  # 10% chance
        
        # Training speed variation
        self.enable_speed_variation = os.getenv("BEHAVIOR_SPEED_VARIATION", "false").lower() == "true"
        self.speed_multiplier_min = float(os.getenv("BEHAVIOR_SPEED_MIN", "0.5"))
        self.speed_multiplier_max = float(os.getenv("BEHAVIOR_SPEED_MAX", "2.0"))
        
        # Coordinator unavailability simulation
        self.enable_coordinator_issues = os.getenv("BEHAVIOR_COORDINATOR_ISSUES", "false").lower() == "true"
        self.coordinator_issue_probability = float(os.getenv("BEHAVIOR_COORDINATOR_ISSUE_PROB", "0.05"))  # 5% chance
        
        # Initialize random seed if provided
        seed = os.getenv("BEHAVIOR_RANDOM_SEED")
        if seed:
            random.seed(int(seed))
    
    def simulate_startup_delay(self) -> float:
        """
        Simulate random startup delay.
        
        Returns:
            Delay in seconds (0 if disabled)
        """
        if not self.enable_startup_delay:
            return 0.0
        
        delay = random.uniform(self.startup_delay_min, self.startup_delay_max)
        return delay
    
    def should_dropout(self) -> bool:
        """
        Determine if client should dropout (exit early).
        
        Returns:
            True if client should dropout
        """
        if not self.enable_dropouts:
            return False
        
        return random.random() < self.dropout_probability
    
    def get_training_speed_multiplier(self) -> float:
        """
        Get a random training speed multiplier.
        
        Returns:
            Multiplier for training time (1.0 if disabled)
        """
        if not self.enable_speed_variation:
            return 1.0
        
        return random.uniform(self.speed_multiplier_min, self.speed_multiplier_max)
    
    def should_simulate_coordinator_issue(self) -> bool:
        """
        Determine if coordinator unavailability should be simulated.
        
        Returns:
            True if coordinator issue should be simulated
        """
        if not self.enable_coordinator_issues:
            return False
        
        return random.random() < self.coordinator_issue_probability
    
    def simulate_coordinator_delay(self) -> float:
        """
        Simulate coordinator response delay.
        
        Returns:
            Delay in seconds (0 if disabled)
        """
        if not self.should_simulate_coordinator_issue():
            return 0.0
        
        # Simulate temporary network issues (1-5 seconds)
        return random.uniform(1.0, 5.0)
    
    def apply_training_delay(self, base_duration: float) -> float:
        """
        Apply speed variation to training duration.
        
        Args:
            base_duration: Base training duration in seconds
            
        Returns:
            Modified training duration
        """
        multiplier = self.get_training_speed_multiplier()
        return base_duration * multiplier


# Global behavior simulator instance
_behavior_simulator = BehaviorSimulator()


def get_simulator() -> BehaviorSimulator:
    """
    Get the global behavior simulator instance.
    
    Returns:
        BehaviorSimulator instance
    """
    return _behavior_simulator


def simulate_startup_delay() -> float:
    """
    Simulate random startup delay.
    
    Returns:
        Delay in seconds
    """
    return _behavior_simulator.simulate_startup_delay()


def should_dropout() -> bool:
    """
    Determine if client should dropout.
    
    Returns:
        True if client should dropout
    """
    return _behavior_simulator.should_dropout()


def get_training_speed_multiplier() -> float:
    """
    Get training speed multiplier.
    
    Returns:
        Speed multiplier
    """
    return _behavior_simulator.get_training_speed_multiplier()


def should_simulate_coordinator_issue() -> bool:
    """
    Determine if coordinator issue should be simulated.
    
    Returns:
        True if coordinator issue should be simulated
    """
    return _behavior_simulator.should_simulate_coordinator_issue()


def simulate_coordinator_delay() -> float:
    """
    Simulate coordinator response delay.
    
    Returns:
        Delay in seconds
    """
    return _behavior_simulator.simulate_coordinator_delay()


def apply_training_delay(base_duration: float) -> float:
    """
    Apply speed variation to training duration.
    
    Args:
        base_duration: Base training duration
        
    Returns:
        Modified duration
    """
    return _behavior_simulator.apply_training_delay(base_duration)


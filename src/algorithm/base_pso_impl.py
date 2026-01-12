"""
Base PSO class - Foundation for all PSO variants
"""

import numpy as np
from typing import List, Dict, Tuple
import torch


class BasePSO:
    """Base class for all PSO variants"""
    
    def __init__(self, config):
        self.config = config
        
        # PSO parameters
        self.num_particles = config.num_particles
        self.max_iterations = config.max_iterations
        self.inertia_weight = config.inertia_weight
        self.cognitive_coef = config.cognitive_coef
        self.social_coef = config.social_coef
        
        # Particle swarm
        self.particles = []
        self.global_best_position = None
        self.global_best_fitness = -np.inf
        
        # Convergence tracking
        self.convergence_history = []
        
    def initialize_particle(self) -> Dict:
        """
        Initialize a single particle
        
        Returns:
            Dictionary containing particle state
        """
        particle = {
            'position': np.random.rand(4),  # [x, y, w, h] normalized
            'velocity': np.random.randn(4) * 0.1,
            'fitness': -np.inf,
            'best_position': np.random.rand(4),
            'best_fitness': -np.inf
        }
        return particle
    
    def initialize_swarm(self):
        """Initialize the particle swarm"""
        self.particles = []
        for _ in range(self.num_particles):
            self.particles.append(self.initialize_particle())
        
        self.global_best_position = np.random.rand(4)
        self.global_best_fitness = -np.inf
        self.convergence_history = []
    
    def position_to_bbox(self, position: np.ndarray) -> np.ndarray:
        """
        Convert normalized position to bounding box coordinates
        
        Args:
            position: Normalized position [x, y, w, h]
            
        Returns:
            Bounding box [x1, y1, x2, y2] in image coordinates
        """
        img_h, img_w = self.config.image_size
        
        # Denormalize
        x_center = position[0] * img_w
        y_center = position[1] * img_h
        width = position[2] * img_w
        height = position[3] * img_h
        
        # Convert to corner format
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        
        # Clip to image bounds
        x1 = np.clip(x1, 0, img_w)
        y1 = np.clip(y1, 0, img_h)
        x2 = np.clip(x2, 0, img_w)
        y2 = np.clip(y2, 0, img_h)
        
        return np.array([x1, y1, x2, y2])
    
    def bbox_to_position(self, bbox: np.ndarray) -> np.ndarray:
        """
        Convert bounding box to normalized position
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Normalized position [x, y, w, h]
        """
        img_h, img_w = self.config.image_size
        
        x1, y1, x2, y2 = bbox
        x_center = (x1 + x2) / 2 / img_w
        y_center = (y1 + y2) / 2 / img_h
        width = (x2 - x1) / img_w
        height = (y2 - y1) / img_h
        
        return np.array([x_center, y_center, width, height])
    
    def calculate_fitness(self, particle: Dict, ground_truth: List[Dict]) -> float:
        """
        Calculate fitness for a particle (to be overridden)
        
        Args:
            particle: Particle dictionary
            ground_truth: Ground truth annotations
            
        Returns:
            Fitness score
        """
        raise NotImplementedError("Subclasses must implement calculate_fitness")
    
    def update_velocity(self, particle: Dict, iteration: int) -> np.ndarray:
        """
        Update particle velocity (standard PSO update)
        
        Args:
            particle: Particle dictionary
            iteration: Current iteration
            
        Returns:
            Updated velocity
        """
        r1, r2 = np.random.rand(2)
        
        cognitive = self.cognitive_coef * r1 * (particle['best_position'] - particle['position'])
        social = self.social_coef * r2 * (self.global_best_position - particle['position'])
        
        velocity = (
            self.inertia_weight * particle['velocity'] +
            cognitive + social
        )
        
        # Velocity clamping
        max_velocity = 0.5
        velocity = np.clip(velocity, -max_velocity, max_velocity)
        
        return velocity
    
    def update_position(self, particle: Dict) -> np.ndarray:
        """
        Update particle position
        
        Args:
            particle: Particle dictionary
            
        Returns:
            Updated position
        """
        position = particle['position'] + particle['velocity']
        position = np.clip(position, 0, 1)
        return position
    
    def detect(self, images: torch.Tensor, ground_truths: List[Dict]) -> Tuple[List[Dict], float]:
        """
        Detect objects in images (to be overridden)
        
        Args:
            images: Batch of input images
            ground_truths: Ground truth annotations
            
        Returns:
            Tuple of (predictions, inference_time)
        """
        raise NotImplementedError("Subclasses must implement detect")

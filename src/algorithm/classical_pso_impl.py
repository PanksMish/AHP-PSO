"""
Classical Particle Swarm Optimization Implementation
Standard PSO with velocity and position updates
"""

import numpy as np
import time
from typing import List, Dict, Tuple
import torch

from .base_pso import BasePSO
from ..utils.metrics import calculate_iou


class ClassicalPSO(BasePSO):
    """
    Classical PSO implementation for object detection
    
    Implements standard PSO algorithm with:
    - Velocity-based particle updates
    - Linear inertia weight decay
    - Global and personal best tracking
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        # Classical PSO specific parameters
        self.inertia_decay = 0.99
        self.velocity_clamp = 0.5
        
    def calculate_fitness(self, particle: Dict, ground_truth: List[Dict]) -> float:
        """
        Calculate fitness based on IoU with ground truth
        
        Args:
            particle: Particle dictionary
            ground_truth: Ground truth annotations
            
        Returns:
            Fitness score
        """
        position = particle['position']
        bbox = self.position_to_bbox(position)
        
        if len(ground_truth) == 0:
            return 0.0
        
        # Calculate IoU with each ground truth box
        ious = []
        for gt in ground_truth:
            iou = calculate_iou(bbox, np.array(gt['bbox']))
            ious.append(iou)
        
        # Use maximum IoU as fitness
        fitness = max(ious) if ious else 0.0
        
        return fitness
    
    def update_velocity(self, particle: Dict, iteration: int) -> np.ndarray:
        """
        Update particle velocity with inertia decay
        
        Args:
            particle: Particle dictionary
            iteration: Current iteration
            
        Returns:
            Updated velocity
        """
        # Apply inertia decay
        current_inertia = self.inertia_weight * (self.inertia_decay ** iteration)
        
        r1, r2 = np.random.rand(2)
        
        cognitive = self.cognitive_coef * r1 * (particle['best_position'] - particle['position'])
        social = self.social_coef * r2 * (self.global_best_position - particle['position'])
        
        velocity = (
            current_inertia * particle['velocity'] +
            cognitive + social
        )
        
        # Velocity clamping
        velocity = np.clip(velocity, -self.velocity_clamp, self.velocity_clamp)
        
        return velocity
    
    def detect(self, images: torch.Tensor, ground_truths: List[Dict]) -> Tuple[List[Dict], float]:
        """
        Detect objects using Classical PSO
        
        Args:
            images: Batch of input images
            ground_truths: Ground truth annotations
            
        Returns:
            Tuple of (predictions, inference_time)
        """
        start_time = time.time()
        
        batch_predictions = []
        
        for idx, image in enumerate(images):
            # Initialize swarm
            self.initialize_swarm()
            
            gt = ground_truths[idx] if idx < len(ground_truths) else []
            
            # PSO iterations
            for iteration in range(self.max_iterations):
                # Update each particle
                for particle in self.particles:
                    # Update velocity
                    particle['velocity'] = self.update_velocity(particle, iteration)
                    
                    # Update position
                    particle['position'] = self.update_position(particle)
                    
                    # Evaluate fitness
                    fitness = self.calculate_fitness(particle, gt)
                    particle['fitness'] = fitness
                    
                    # Update personal best
                    if fitness > particle['best_fitness']:
                        particle['best_fitness'] = fitness
                        particle['best_position'] = particle['position'].copy()
                    
                    # Update global best
                    if fitness > self.global_best_fitness:
                        self.global_best_fitness = fitness
                        self.global_best_position = particle['position'].copy()
                
                # Track convergence
                self.convergence_history.append(self.global_best_fitness)
            
            # Convert best position to detection
            best_bbox = self.position_to_bbox(self.global_best_position)
            prediction = {
                'bbox': best_bbox,
                'score': self.global_best_fitness,
                'class': 0  # Simplified
            }
            batch_predictions.append(prediction)
        
        inference_time = time.time() - start_time
        
        return batch_predictions, inference_time

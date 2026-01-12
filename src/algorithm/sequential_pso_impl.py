"""
Sequential Particle Swarm Optimization
Motion-aware PSO for video sequences with temporal consistency
"""

import numpy as np
import time
from typing import List, Dict, Tuple
import torch

from .base_pso import BasePSO
from ..utils.metrics import calculate_iou


class SequentialPSO(BasePSO):
    """
    Sequential PSO for video-based object detection
    
    Features:
    - Motion vector estimation
    - Temporal consistency tracking
    - Frame-to-frame adaptation
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        # Sequential PSO parameters
        self.temporal_weight = 0.6
        self.motion_estimation = True
        self.frame_window = 3
        self.gaussian_noise_std = 0.01
        
        # Motion history
        self.previous_positions = []
        self.motion_vectors = []
        
    def estimate_motion_vector(self, current_position: np.ndarray) -> np.ndarray:
        """
        Estimate motion vector from previous frames
        
        Args:
            current_position: Current particle position
            
        Returns:
            Estimated motion vector
        """
        if len(self.previous_positions) < 2:
            return np.zeros(4)
        
        # Calculate average motion from recent frames
        recent_positions = self.previous_positions[-self.frame_window:]
        motion = np.zeros(4)
        
        for i in range(1, len(recent_positions)):
            motion += recent_positions[i] - recent_positions[i-1]
        
        if len(recent_positions) > 1:
            motion /= (len(recent_positions) - 1)
        
        return motion
    
    def sequential_update(self, particle: Dict, motion_vector: np.ndarray) -> np.ndarray:
        """
        Sequential position update with motion estimation
        
        Args:
            particle: Particle dictionary
            motion_vector: Estimated motion vector
            
        Returns:
            Updated position
        """
        # Add motion vector
        new_position = particle['position'] + motion_vector
        
        # Add small Gaussian noise for exploration
        noise = np.random.normal(0, self.gaussian_noise_std, 4)
        new_position += noise
        
        # Clip to valid range
        new_position = np.clip(new_position, 0, 1)
        
        return new_position
    
    def calculate_temporal_consistency(
        self, 
        current_bbox: np.ndarray, 
        previous_bbox: np.ndarray
    ) -> float:
        """
        Calculate temporal consistency score
        
        Args:
            current_bbox: Current bounding box
            previous_bbox: Previous bounding box
            
        Returns:
            Consistency score (0-1)
        """
        if previous_bbox is None:
            return 1.0
        
        # Calculate IoU between consecutive frames
        iou = calculate_iou(current_bbox, previous_bbox)
        
        return iou
    
    def calculate_fitness(
        self, 
        particle: Dict, 
        ground_truth: List[Dict],
        previous_bbox: np.ndarray = None
    ) -> float:
        """
        Calculate fitness with temporal consistency
        
        Args:
            particle: Particle dictionary
            ground_truth: Ground truth annotations
            previous_bbox: Previous frame's bounding box
            
        Returns:
            Fitness score
        """
        position = particle['position']
        bbox = self.position_to_bbox(position)
        
        # Detection accuracy (mAP component)
        if len(ground_truth) > 0:
            ious = [calculate_iou(bbox, np.array(gt['bbox'])) for gt in ground_truth]
            detection_score = max(ious)
        else:
            detection_score = 0.0
        
        # Temporal consistency
        if previous_bbox is not None:
            temporal_score = self.calculate_temporal_consistency(bbox, previous_bbox)
        else:
            temporal_score = 1.0
        
        # Combined fitness
        fitness = (
            self.temporal_weight * detection_score +
            (1 - self.temporal_weight) * temporal_score
        )
        
        return fitness
    
    def detect(self, images: torch.Tensor, ground_truths: List[Dict]) -> Tuple[List[Dict], float]:
        """
        Detect objects using Sequential PSO
        
        Args:
            images: Batch of input images
            ground_truths: Ground truth annotations
            
        Returns:
            Tuple of (predictions, inference_time)
        """
        start_time = time.time()
        
        batch_predictions = []
        previous_bbox = None
        
        for idx, image in enumerate(images):
            # Initialize swarm
            if idx == 0:
                self.initialize_swarm()
            else:
                # Use previous positions to initialize
                for particle in self.particles:
                    motion_vector = self.estimate_motion_vector(particle['position'])
                    particle['position'] = self.sequential_update(particle, motion_vector)
            
            gt = ground_truths[idx] if idx < len(ground_truths) else []
            
            # PSO iterations
            for iteration in range(self.max_iterations):
                # Update each particle
                for particle in self.particles:
                    # Estimate motion
                    if self.motion_estimation and len(self.previous_positions) > 0:
                        motion_vector = self.estimate_motion_vector(particle['position'])
                        particle['position'] = self.sequential_update(particle, motion_vector)
                    else:
                        # Standard velocity update
                        particle['velocity'] = self.update_velocity(particle, iteration)
                        particle['position'] = self.update_position(particle)
                    
                    # Evaluate fitness with temporal consistency
                    fitness = self.calculate_fitness(particle, gt, previous_bbox)
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
                'class': 0
            }
            batch_predictions.append(prediction)
            
            # Update history
            self.previous_positions.append(self.global_best_position.copy())
            if len(self.previous_positions) > self.frame_window:
                self.previous_positions.pop(0)
            
            previous_bbox = best_bbox
        
        inference_time = time.time() - start_time
        
        return batch_predictions, inference_time

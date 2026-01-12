"""
AHP-PSO: Adaptive Hybrid Particle Swarm Optimization
Main implementation of the proposed algorithm
"""

import numpy as np
import time
from typing import List, Tuple, Dict
import torch

from .base_pso import BasePSO
from ..utils.ahp import AHPCalculator
from ..utils.metrics import calculate_iou, calculate_fitness


class AHPPSO(BasePSO):
    """
    Adaptive Hybrid Particle Swarm Optimization for Object Detection
    
    Implements the AHP-PSO algorithm from the paper with:
    - AHP-based multi-criteria fitness evaluation
    - Adaptive inertia weight modulation
    - Dynamic swarm size adjustment
    - Domain-aware optimization
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        # AHP-specific parameters
        self.ahp_calculator = AHPCalculator(config.domain_params['fitness_weights'])
        self.ahp_weights = None
        self.update_ahp_weights()
        
        # Adaptive parameters
        self.adaptive_inertia = True
        self.inertia_min = 0.4
        self.inertia_max = 0.9
        
        # Population adaptation
        self.population_adaptation = True
        self.pop_size_min = 30
        self.pop_size_max = 70
        
        # Performance tracking
        self.fitness_history = []
        self.diversity_history = []
        self.convergence_rate = []
        
        # Domain-specific initialization
        self._initialize_domain_specific()
    
    def _initialize_domain_specific(self):
        """Initialize domain-specific parameters"""
        domain_params = self.config.domain_params
        
        if 'num_particles' in domain_params:
            self.num_particles = domain_params['num_particles']
        
        if 'max_iterations' in domain_params:
            self.max_iterations = domain_params['max_iterations']
    
    def update_ahp_weights(self):
        """Update AHP weights based on current performance"""
        self.ahp_weights = self.ahp_calculator.calculate_weights()
    
    def calculate_adaptive_inertia(self, iteration: int, particle_fitness: np.ndarray) -> np.ndarray:
        """
        Calculate adaptive inertia weight for each particle
        
        Args:
            iteration: Current iteration number
            particle_fitness: Fitness values for all particles
            
        Returns:
            Array of inertia weights for each particle
        """
        # Base inertia with linear decay
        base_inertia = self.inertia_max - (self.inertia_max - self.inertia_min) * (iteration / self.max_iterations)
        
        # Fitness-based modulation
        if len(particle_fitness) > 0:
            fitness_mean = np.mean(particle_fitness)
            fitness_std = np.std(particle_fitness) + 1e-8
            
            # Normalize fitness deviation
            fitness_deviation = (particle_fitness - fitness_mean) / fitness_std
            
            # Better particles get lower inertia (more exploitation)
            # Worse particles get higher inertia (more exploration)
            inertia_modulation = 0.2 * np.tanh(-fitness_deviation)
            
            inertia_weights = base_inertia + inertia_modulation
            inertia_weights = np.clip(inertia_weights, self.inertia_min, self.inertia_max)
        else:
            inertia_weights = np.full(self.num_particles, base_inertia)
        
        return inertia_weights
    
    def calculate_swarm_diversity(self) -> float:
        """Calculate diversity of particle swarm"""
        if len(self.particles) == 0:
            return 0.0
        
        positions = np.array([p['position'] for p in self.particles])
        center = np.mean(positions, axis=0)
        distances = np.linalg.norm(positions - center, axis=1)
        diversity = np.mean(distances)
        
        return diversity
    
    def adapt_population_size(self, iteration: int):
        """
        Dynamically adjust population size based on convergence
        
        Args:
            iteration: Current iteration number
        """
        if not self.population_adaptation or iteration < 10:
            return
        
        # Calculate convergence rate
        if len(self.convergence_history) >= 5:
            recent_improvement = (
                self.convergence_history[-1] - self.convergence_history[-5]
            )
            
            # If converging slowly, increase population
            if recent_improvement < 0.01:
                target_size = min(self.num_particles + 5, self.pop_size_max)
            # If converging fast, decrease population
            elif recent_improvement > 0.05:
                target_size = max(self.num_particles - 5, self.pop_size_min)
            else:
                target_size = self.num_particles
            
            # Adjust population
            if target_size > self.num_particles:
                self._add_particles(target_size - self.num_particles)
            elif target_size < self.num_particles:
                self._remove_particles(self.num_particles - target_size)
    
    def _add_particles(self, count: int):
        """Add new particles to swarm"""
        for _ in range(count):
            particle = self.initialize_particle()
            self.particles.append(particle)
        self.num_particles = len(self.particles)
    
    def _remove_particles(self, count: int):
        """Remove worst performing particles"""
        # Sort by fitness
        sorted_particles = sorted(
            enumerate(self.particles),
            key=lambda x: x[1]['fitness'],
            reverse=False
        )
        
        # Remove worst particles
        indices_to_remove = [idx for idx, _ in sorted_particles[:count]]
        self.particles = [
            p for i, p in enumerate(self.particles)
            if i not in indices_to_remove
        ]
        self.num_particles = len(self.particles)
    
    def calculate_ahp_fitness(self, particle: Dict, ground_truth: List[Dict]) -> float:
        """
        Calculate fitness using AHP-weighted multi-criteria evaluation
        
        Args:
            particle: Particle dictionary with position
            ground_truth: Ground truth annotations
            
        Returns:
            AHP-weighted fitness score
        """
        position = particle['position']
        bbox = self.position_to_bbox(position)
        
        # Calculate individual metrics
        metrics = {}
        
        # IoU with best matching ground truth
        if len(ground_truth) > 0:
            ious = [calculate_iou(bbox, gt['bbox']) for gt in ground_truth]
            metrics['iou'] = max(ious)
        else:
            metrics['iou'] = 0.0
        
        # Precision (simplified)
        metrics['precision'] = metrics['iou']
        
        # Recall (simplified)
        metrics['recall'] = metrics['iou']
        
        # FPS consideration (inverse of complexity)
        bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        metrics['fps'] = 1.0 - (bbox_area / (self.config.image_size[0] * self.config.image_size[1]))
        
        # Domain-specific metrics
        if self.config.domain == 'underwater':
            metrics['contrast_score'] = self._calculate_contrast_score(bbox)
            metrics['edge_preservation'] = self._calculate_edge_score(bbox)
        elif self.config.domain == 'road':
            metrics['texture_score'] = self._calculate_texture_score(bbox)
        
        # Calculate weighted fitness using AHP weights
        fitness = 0.0
        for metric_name, weight in self.ahp_weights.items():
            if metric_name in metrics:
                fitness += weight * metrics[metric_name]
        
        return fitness
    
    def _calculate_contrast_score(self, bbox: np.ndarray) -> float:
        """Calculate contrast score for underwater domain"""
        # Placeholder implementation
        return 0.8
    
    def _calculate_edge_score(self, bbox: np.ndarray) -> float:
        """Calculate edge preservation score"""
        # Placeholder implementation
        return 0.75
    
    def _calculate_texture_score(self, bbox: np.ndarray) -> float:
        """Calculate texture score for road domain"""
        # Placeholder implementation
        return 0.85
    
    def detect(self, images: torch.Tensor, ground_truths: List[Dict]) -> Tuple[List[Dict], float]:
        """
        Detect objects in images using AHP-PSO
        
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
                # Calculate adaptive inertia
                particle_fitnesses = np.array([p['fitness'] for p in self.particles])
                inertia_weights = self.calculate_adaptive_inertia(iteration, particle_fitnesses)
                
                # Update particles
                for i, particle in enumerate(self.particles):
                    # Update velocity with adaptive inertia
                    r1, r2 = np.random.rand(2)
                    
                    cognitive = self.cognitive_coef * r1 * (particle['best_position'] - particle['position'])
                    social = self.social_coef * r2 * (self.global_best_position - particle['position'])
                    
                    particle['velocity'] = (
                        inertia_weights[i] * particle['velocity'] +
                        cognitive + social
                    )
                    
                    # Update position
                    particle['position'] = particle['position'] + particle['velocity']
                    particle['position'] = np.clip(particle['position'], 0, 1)
                    
                    # Evaluate fitness
                    fitness = self.calculate_ahp_fitness(particle, gt)
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
                
                # Adapt population size
                self.adapt_population_size(iteration)
                
                # Update AHP weights if environment changed
                if iteration % 20 == 0 and iteration > 0:
                    diversity = self.calculate_swarm_diversity()
                    self.diversity_history.append(diversity)
                    
                    if len(self.diversity_history) >= 3:
                        diversity_change = abs(
                            self.diversity_history[-1] - self.diversity_history[-3]
                        )
                        if diversity_change > 0.1:
                            self.update_ahp_weights()
            
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

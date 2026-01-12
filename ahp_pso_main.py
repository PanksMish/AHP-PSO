"""
AHP-PSO: Adaptive Hybrid Particle Swarm Optimization for Real-Time Object Detection
Main entry point for the implementation

Authors: Pankaj Mishra, V Venkataramanan, Anand Nayyar
Paper: AHP-PSO for Real-Time Cross-Domain Object Detection
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.config import Config
from src.data_loader import COCODataLoader
from src.algorithms.classical_pso import ClassicalPSO
from src.algorithms.quantum_pso import QuantumPSO
from src.algorithms.sequential_pso import SequentialPSO
from src.algorithms.ahp_pso import AHPPSO
from src.evaluator import Evaluator
from src.utils.logger import setup_logger
from src.utils.visualization import Visualizer


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='AHP-PSO Object Detection')
    
    parser.add_argument('--algorithm', type=str, default='ahp_pso',
                        choices=['classical_pso', 'quantum_pso', 'sequential_pso', 'ahp_pso'],
                        help='Algorithm to run')
    
    parser.add_argument('--domain', type=str, default='aerial',
                        choices=['aerial', 'underwater', 'road', 'general'],
                        help='Domain-specific configuration')
    
    parser.add_argument('--dataset_path', type=str, default='data/coco',
                        help='Path to COCO dataset')
    
    parser.add_argument('--num_particles', type=int, default=50,
                        help='Number of particles in swarm')
    
    parser.add_argument('--max_iterations', type=int, default=150,
                        help='Maximum iterations')
    
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for evaluation')
    
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Output directory for results')
    
    parser.add_argument('--visualize', action='store_true',
                        help='Enable visualization')
    
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU id to use')
    
    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_algorithm(config, algorithm_name):
    """Factory function to get algorithm instance"""
    algorithms = {
        'classical_pso': ClassicalPSO,
        'quantum_pso': QuantumPSO,
        'sequential_pso': SequentialPSO,
        'ahp_pso': AHPPSO
    }
    
    if algorithm_name not in algorithms:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")
    
    return algorithms[algorithm_name](config)


def main():
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Setup device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    
    # Create output directory
    output_dir = Path(args.output_dir) / args.algorithm / args.domain
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logger
    logger = setup_logger(output_dir / 'experiment.log')
    logger.info(f"Starting experiment: {args.algorithm} on {args.domain} domain")
    logger.info(f"Arguments: {args}")
    
    # Load configuration
    config = Config(
        domain=args.domain,
        num_particles=args.num_particles,
        max_iterations=args.max_iterations,
        device=device
    )
    
    # Load dataset
    logger.info("Loading dataset...")
    data_loader = COCODataLoader(
        dataset_path=args.dataset_path,
        domain=args.domain,
        batch_size=args.batch_size,
        config=config
    )
    
    # Initialize algorithm
    logger.info(f"Initializing {args.algorithm}...")
    algorithm = get_algorithm(config, args.algorithm)
    
    # Initialize evaluator
    evaluator = Evaluator(config)
    
    # Initialize visualizer
    visualizer = Visualizer(output_dir) if args.visualize else None
    
    # Run detection on validation set
    logger.info("Running object detection...")
    all_predictions = []
    all_ground_truths = []
    all_inference_times = []
    
    for batch_idx, batch in enumerate(tqdm(data_loader.val_loader, desc="Processing")):
        images = batch['image']
        ground_truths = batch['annotations']
        
        # Run algorithm
        predictions, inference_time = algorithm.detect(images, ground_truths)
        
        all_predictions.extend(predictions)
        all_ground_truths.extend(ground_truths)
        all_inference_times.append(inference_time)
        
        # Visualize first batch
        if args.visualize and batch_idx == 0:
            visualizer.plot_detections(images[0], predictions[0], ground_truths[0])
    
    # Evaluate results
    logger.info("Evaluating results...")
    metrics = evaluator.evaluate(all_predictions, all_ground_truths)
    
    # Add timing metrics
    metrics['avg_inference_time'] = np.mean(all_inference_times)
    metrics['fps'] = 1.0 / metrics['avg_inference_time']
    
    # Log results
    logger.info("\n" + "="*50)
    logger.info("FINAL RESULTS")
    logger.info("="*50)
    for metric_name, value in metrics.items():
        logger.info(f"{metric_name}: {value:.4f}")
    
    # Save results
    evaluator.save_results(metrics, output_dir / 'metrics.json')
    
    # Generate visualizations
    if args.visualize:
        logger.info("Generating visualizations...")
        visualizer.plot_metrics(metrics)
        visualizer.plot_convergence(algorithm.convergence_history)
    
    logger.info(f"Results saved to {output_dir}")
    logger.info("Experiment completed successfully!")


if __name__ == '__main__':
    main()

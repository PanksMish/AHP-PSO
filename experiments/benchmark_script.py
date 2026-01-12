"""
Run all algorithms and generate comparison results
"""

import argparse
import sys
import os
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.data_loader import COCODataLoader
from src.algorithms.classical_pso import ClassicalPSO
from src.algorithms.quantum_pso import QuantumPSO
from src.algorithms.sequential_pso import SequentialPSO
from src.algorithms.ahp_pso import AHPPSO
from src.evaluator import Evaluator
from src.utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Run all algorithms benchmark')
    parser.add_argument('--domain', type=str, default='aerial',
                        choices=['aerial', 'underwater', 'road', 'general'])
    parser.add_argument('--dataset_path', type=str, default='data/coco')
    parser.add_argument('--output_dir', type=str, default='results/benchmark')
    parser.add_argument('--num_images', type=int, default=100,
                        help='Number of images to evaluate')
    return parser.parse_args()


def run_algorithm(algorithm_name, algorithm_class, data_loader, config, evaluator, num_images):
    """Run a single algorithm and return metrics"""
    
    print(f"\n{'='*60}")
    print(f"Running {algorithm_name}")
    print(f"{'='*60}")
    
    algorithm = algorithm_class(config)
    
    all_predictions = []
    all_ground_truths = []
    inference_times = []
    
    # Limit number of batches
    max_batches = num_images // data_loader.val_loader.batch_size + 1
    
    for batch_idx, batch in enumerate(data_loader.val_loader):
        if batch_idx >= max_batches:
            break
        
        images = batch['image']
        ground_truths = batch['annotations']
        
        # Run detection
        start_time = time.time()
        try:
            predictions, _ = algorithm.detect(images, ground_truths)
            inference_time = time.time() - start_time
            
            all_predictions.extend(predictions)
            all_ground_truths.extend(ground_truths)
            inference_times.append(inference_time)
            
            print(f"Batch {batch_idx + 1}/{max_batches} completed - "
                  f"Time: {inference_time:.3f}s")
        except Exception as e:
            print(f"Error in batch {batch_idx}: {str(e)}")
            continue
    
    # Evaluate
    print(f"\nEvaluating {algorithm_name}...")
    metrics = evaluator.evaluate(all_predictions, all_ground_truths)
    
    # Add timing metrics
    metrics['avg_inference_time'] = np.mean(inference_times)
    metrics['fps'] = 1.0 / metrics['avg_inference_time'] if metrics['avg_inference_time'] > 0 else 0
    metrics['convergence_iterations'] = len(algorithm.convergence_history)
    
    return metrics


def create_comparison_plots(results, output_dir):
    """Create comparison visualizations"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract metrics
    algorithms = list(results.keys())
    metrics = ['precision', 'recall', 'f1_score', 'mAP', 'iou', 'fps']
    
    # Create DataFrame
    data = []
    for algo in algorithms:
        row = {'Algorithm': algo}
        for metric in metrics:
            if metric in results[algo]:
                row[metric] = results[algo][metric]
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # 1. Bar chart comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        df.plot(x='Algorithm', y=metric, kind='bar', ax=ax, legend=False)
        ax.set_title(f'{metric.upper()}', fontsize=12, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel(metric)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Radar chart
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    
    for algo in algorithms:
        values = []
        for metric in metrics:
            if metric in results[algo]:
                val = results[algo][metric]
                # Normalize fps to 0-1 range
                if metric == 'fps':
                    val = min(val / 30.0, 1.0)
                values.append(val)
            else:
                values.append(0)
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=algo)
        ax.fill(angles, values, alpha=0.15)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)
    
    plt.title('Algorithm Performance Comparison (Radar Chart)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.savefig(output_dir / 'radar_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Heatmap
    heatmap_data = df.set_index('Algorithm')[metrics].T
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd', 
                cbar_kws={'label': 'Score'}, linewidths=0.5)
    plt.title('Algorithm Performance Heatmap', fontsize=14, fontweight='bold')
    plt.xlabel('Algorithm', fontsize=12)
    plt.ylabel('Metric', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / 'heatmap_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Speed vs Accuracy scatter
    plt.figure(figsize=(10, 8))
    
    for algo in algorithms:
        if 'fps' in results[algo] and 'mAP' in results[algo]:
            plt.scatter(results[algo]['fps'], results[algo]['mAP'], 
                       s=200, alpha=0.6, label=algo)
            plt.annotate(algo, 
                        (results[algo]['fps'], results[algo]['mAP']),
                        fontsize=10, ha='right')
    
    plt.xlabel('FPS (Speed)', fontsize=12)
    plt.ylabel('mAP (Accuracy)', fontsize=12)
    plt.title('Speed vs Accuracy Trade-off', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'speed_accuracy_tradeoff.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualizations saved to {output_dir}")


def main():
    args = parse_args()
    
    # Setup
    output_dir = Path(args.output_dir) / args.domain
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logger(output_dir / 'benchmark.log')
    logger.info(f"Starting benchmark for {args.domain} domain")
    
    # Load configuration
    config = Config(domain=args.domain)
    
    # Load dataset
    logger.info("Loading dataset...")
    data_loader = COCODataLoader(
        args.dataset_path,
        args.domain,
        batch_size=16,
        config=config
    )
    
    # Initialize evaluator
    evaluator = Evaluator(config)
    
    # Define algorithms
    algorithms = {
        'Classical PSO': ClassicalPSO,
        'Quantum PSO': QuantumPSO,
        'Sequential PSO': SequentialPSO,
        'AHP-PSO': AHPPSO
    }
    
    # Run all algorithms
    results = {}
    
    for algo_name, algo_class in algorithms.items():
        try:
            metrics = run_algorithm(
                algo_name, algo_class, data_loader, 
                config, evaluator, args.num_images
            )
            results[algo_name] = metrics
            
            # Log results
            logger.info(f"\n{algo_name} Results:")
            for metric, value in metrics.items():
                logger.info(f"  {metric}: {value:.4f}")
        
        except Exception as e:
            logger.error(f"Error running {algo_name}: {str(e)}")
            continue
    
    # Save results
    with open(output_dir / 'benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    # Create comparison table
    print(f"\n{'='*80}")
    print("FINAL RESULTS COMPARISON")
    print(f"{'='*80}")
    
    # Print table
    metrics_to_show = ['precision', 'recall', 'f1_score', 'mAP', 'iou', 'fps']
    
    print(f"\n{'Algorithm':<20}", end='')
    for metric in metrics_to_show:
        print(f"{metric:>12}", end='')
    print()
    print('-' * 100)
    
    for algo_name, metrics in results.items():
        print(f"{algo_name:<20}", end='')
        for metric in metrics_to_show:
            if metric in metrics:
                print(f"{metrics[metric]:>12.4f}", end='')
            else:
                print(f"{'N/A':>12}", end='')
        print()
    
    # Create visualizations
    logger.info("\nGenerating comparison plots...")
    create_comparison_plots(results, output_dir)
    
    logger.info(f"\nBenchmark completed! Results saved to {output_dir}")


if __name__ == '__main__':
    main()

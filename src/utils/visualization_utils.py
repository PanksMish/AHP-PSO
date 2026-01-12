"""
Visualization utilities for results and analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict
import cv2


class Visualizer:
    """Visualization tool for object detection results"""
    
    def __init__(self, output_dir: Path):
        """
        Initialize visualizer
        
        Args:
            output_dir: Output directory for saving plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        sns.set_style('whitegrid')
        plt.rcParams['figure.figsize'] = (12, 8)
    
    def plot_detections(
        self,
        image: np.ndarray,
        predictions: Dict,
        ground_truths: List[Dict],
        save_name: str = 'detection_result.png'
    ):
        """
        Visualize detection results on image
        
        Args:
            image: Input image
            predictions: Predicted bounding boxes
            ground_truths: Ground truth annotations
            save_name: Output filename
        """
        # Convert tensor to numpy if needed
        if hasattr(image, 'cpu'):
            image = image.cpu().numpy()
        
        # Denormalize image
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        
        # Convert to BGR for OpenCV
        if len(image.shape) == 3 and image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))
        
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Draw ground truths in green
        for gt in ground_truths:
            bbox = [int(x) for x in gt['bbox']]
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.putText(image, 'GT', (bbox[0], bbox[1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw predictions in red
        if isinstance(predictions, dict) and 'bbox' in predictions:
            bbox = [int(x) for x in predictions['bbox']]
            score = predictions.get('score', 0)
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
            cv2.putText(image, f'Pred: {score:.2f}', (bbox[0], bbox[1]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Save image
        output_path = self.output_dir / save_name
        cv2.imwrite(str(output_path), image)
    
    def plot_convergence(
        self,
        convergence_history: List[float],
        save_name: str = 'convergence.png'
    ):
        """
        Plot convergence curve
        
        Args:
            convergence_history: List of fitness values over iterations
            save_name: Output filename
        """
        plt.figure(figsize=(10, 6))
        plt.plot(convergence_history, linewidth=2)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Best Fitness', fontsize=12)
        plt.title('Convergence History', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_path = self.output_dir / save_name
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_metrics(
        self,
        metrics: Dict[str, float],
        save_name: str = 'metrics.png'
    ):
        """
        Plot bar chart of metrics
        
        Args:
            metrics: Dictionary of metric values
            save_name: Output filename
        """
        # Filter numeric metrics
        plot_metrics = {
            k: v for k, v in metrics.items()
            if isinstance(v, (int, float)) and k in [
                'precision', 'recall', 'f1_score', 'mAP', 'iou', 'fps'
            ]
        }
        
        if not plot_metrics:
            return
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(range(len(plot_metrics)), list(plot_metrics.values()))
        plt.xticks(range(len(plot_metrics)), list(plot_metrics.keys()), rotation=45)
        plt.ylabel('Score', fontsize=12)
        plt.title('Performance Metrics', fontsize=14, fontweight='bold')
        
        # Color bars
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # Add value labels
        for i, (k, v) in enumerate(plot_metrics.items()):
            plt.text(i, v, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        output_path = self.output_dir / save_name
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_pr_curve(
        self,
        precisions: List[float],
        recalls: List[float],
        save_name: str = 'pr_curve.png'
    ):
        """
        Plot Precision-Recall curve
        
        Args:
            precisions: List of precision values
            recalls: List of recall values
            save_name: Output filename
        """
        plt.figure(figsize=(8, 8))
        plt.plot(recalls, precisions, linewidth=2, marker='o')
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.tight_layout()
        
        output_path = self.output_dir / save_name
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

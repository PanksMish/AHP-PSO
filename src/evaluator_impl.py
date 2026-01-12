"""
Evaluation metrics for object detection
"""

import numpy as np
import json
from typing import List, Dict
from collections import defaultdict


class Evaluator:
    """Evaluator for object detection metrics"""
    
    def __init__(self, config):
        self.config = config
        self.iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        
    def calculate_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """
        Calculate Intersection over Union
        
        Args:
            box1: First bounding box [x1, y1, x2, y2]
            box2: Second bounding box [x1, y1, x2, y2]
            
        Returns:
            IoU score
        """
        # Calculate intersection
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        # Calculate union
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def match_predictions_to_ground_truth(
        self,
        predictions: List[Dict],
        ground_truths: List[Dict],
        iou_threshold: float = 0.5
    ) -> tuple:
        """
        Match predictions to ground truth boxes
        
        Args:
            predictions: List of prediction dictionaries
            ground_truths: List of ground truth dictionaries
            iou_threshold: IoU threshold for matching
            
        Returns:
            Tuple of (true_positives, false_positives, false_negatives, ious)
        """
        if len(predictions) == 0:
            return 0, 0, len(ground_truths), []
        
        if len(ground_truths) == 0:
            return 0, len(predictions), 0, []
        
        # Calculate IoU matrix
        iou_matrix = np.zeros((len(predictions), len(ground_truths)))
        for i, pred in enumerate(predictions):
            for j, gt in enumerate(ground_truths):
                iou_matrix[i, j] = self.calculate_iou(
                    np.array(pred['bbox']),
                    np.array(gt['bbox'])
                )
        
        # Match predictions to ground truths
        matched_gts = set()
        true_positives = 0
        false_positives = 0
        matched_ious = []
        
        # Sort predictions by confidence
        sorted_preds = sorted(
            enumerate(predictions),
            key=lambda x: x[1].get('score', 1.0),
            reverse=True
        )
        
        for pred_idx, pred in sorted_preds:
            # Find best matching ground truth
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx in range(len(ground_truths)):
                if gt_idx not in matched_gts:
                    iou = iou_matrix[pred_idx, gt_idx]
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
            
            # Check if match is valid
            if best_iou >= iou_threshold:
                true_positives += 1
                matched_gts.add(best_gt_idx)
                matched_ious.append(best_iou)
            else:
                false_positives += 1
        
        false_negatives = len(ground_truths) - len(matched_gts)
        
        return true_positives, false_positives, false_negatives, matched_ious
    
    def calculate_precision_recall(
        self,
        predictions: List[Dict],
        ground_truths: List[Dict],
        iou_threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Calculate precision, recall, and F1-score
        
        Args:
            predictions: List of predictions
            ground_truths: List of ground truths
            iou_threshold: IoU threshold
            
        Returns:
            Dictionary of metrics
        """
        tp, fp, fn, ious = self.match_predictions_to_ground_truth(
            predictions, ground_truths, iou_threshold
        )
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        mean_iou = np.mean(ious) if len(ious) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'iou': mean_iou,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }
    
    def calculate_average_precision(
        self,
        predictions: List[Dict],
        ground_truths: List[Dict],
        iou_threshold: float = 0.5
    ) -> float:
        """
        Calculate Average Precision (AP)
        
        Args:
            predictions: List of predictions
            ground_truths: List of ground truths
            iou_threshold: IoU threshold
            
        Returns:
            Average Precision score
        """
        if len(ground_truths) == 0:
            return 0.0
        
        # Sort predictions by confidence
        sorted_preds = sorted(
            predictions,
            key=lambda x: x.get('score', 1.0),
            reverse=True
        )
        
        # Calculate precision-recall curve
        tp_cumsum = 0
        fp_cumsum = 0
        precisions = []
        recalls = []
        
        matched_gts = set()
        
        for pred in sorted_preds:
            # Find best matching ground truth
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt in enumerate(ground_truths):
                if gt_idx not in matched_gts:
                    iou = self.calculate_iou(
                        np.array(pred['bbox']),
                        np.array(gt['bbox'])
                    )
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
            
            if best_iou >= iou_threshold:
                tp_cumsum += 1
                matched_gts.add(best_gt_idx)
            else:
                fp_cumsum += 1
            
            precision = tp_cumsum / (tp_cumsum + fp_cumsum)
            recall = tp_cumsum / len(ground_truths)
            
            precisions.append(precision)
            recalls.append(recall)
        
        # Calculate AP using 11-point interpolation
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            precs = [p for p, r in zip(precisions, recalls) if r >= t]
            if len(precs) > 0:
                ap += max(precs) / 11
        
        return ap
    
    def calculate_map(
        self,
        all_predictions: List[List[Dict]],
        all_ground_truths: List[List[Dict]]
    ) -> Dict[str, float]:
        """
        Calculate mean Average Precision (mAP) across images
        
        Args:
            all_predictions: List of prediction lists per image
            all_ground_truths: List of ground truth lists per image
            
        Returns:
            Dictionary with mAP metrics
        """
        # Calculate AP for each IoU threshold
        aps_per_threshold = defaultdict(list)
        
        for predictions, ground_truths in zip(all_predictions, all_ground_truths):
            for iou_thresh in self.iou_thresholds:
                ap = self.calculate_average_precision(
                    predictions, ground_truths, iou_thresh
                )
                aps_per_threshold[iou_thresh].append(ap)
        
        # Calculate mean AP for each threshold
        map_results = {}
        for iou_thresh in self.iou_thresholds:
            map_results[f'mAP_{int(iou_thresh*100)}'] = np.mean(
                aps_per_threshold[iou_thresh]
            )
        
        # Calculate overall mAP (average over all thresholds)
        map_results['mAP'] = np.mean([
            map_results[f'mAP_{int(t*100)}'] for t in self.iou_thresholds
        ])
        
        # Special mAP@50 and mAP@75
        map_results['mAP_50'] = map_results['mAP_50']
        map_results['mAP_75'] = map_results['mAP_75']
        
        return map_results
    
    def evaluate(
        self,
        all_predictions: List[List[Dict]],
        all_ground_truths: List[List[Dict]]
    ) -> Dict[str, float]:
        """
        Comprehensive evaluation
        
        Args:
            all_predictions: List of prediction lists per image
            all_ground_truths: List of ground truth lists per image
            
        Returns:
            Dictionary of all metrics
        """
        # Flatten for overall metrics
        flat_predictions = [p for preds in all_predictions for p in preds]
        flat_ground_truths = [gt for gts in all_ground_truths for gt in gts]
        
        # Calculate precision, recall, F1
        pr_metrics = self.calculate_precision_recall(
            flat_predictions, flat_ground_truths
        )
        
        # Calculate mAP
        map_metrics = self.calculate_map(all_predictions, all_ground_truths)
        
        # Combine metrics
        metrics = {**pr_metrics, **map_metrics}
        
        return metrics
    
    def save_results(self, metrics: Dict, output_path: str):
        """Save evaluation results to JSON"""
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=4)

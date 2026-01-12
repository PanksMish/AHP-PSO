"""
Utility functions for metric calculations
"""

import numpy as np
from typing import List, Dict, Tuple


def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Calculate Intersection over Union between two boxes
    
    Args:
        box1: First bounding box [x1, y1, x2, y2]
        box2: Second bounding box [x1, y1, x2, y2]
        
    Returns:
        IoU score (0-1)
    """
    # Calculate intersection coordinates
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    # Calculate intersection area
    inter_width = max(0, x2_inter - x1_inter)
    inter_height = max(0, y2_inter - y1_inter)
    intersection = inter_width * inter_height
    
    # Calculate union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    
    if union == 0:
        return 0.0
    
    iou = intersection / union
    return iou


def calculate_fitness(
    bbox: np.ndarray,
    ground_truths: List[Dict],
    weights: Dict[str, float] = None
) -> float:
    """
    Calculate fitness score for a bounding box
    
    Args:
        bbox: Predicted bounding box [x1, y1, x2, y2]
        ground_truths: List of ground truth annotations
        weights: Weights for different criteria
        
    Returns:
        Fitness score
    """
    if weights is None:
        weights = {'iou': 1.0}
    
    if len(ground_truths) == 0:
        return 0.0
    
    # Calculate IoU with each ground truth
    ious = []
    for gt in ground_truths:
        iou = calculate_iou(bbox, np.array(gt['bbox']))
        ious.append(iou)
    
    # Use maximum IoU
    max_iou = max(ious) if ious else 0.0
    
    # Apply weights
    fitness = weights.get('iou', 1.0) * max_iou
    
    return fitness


def non_max_suppression(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.5
) -> List[int]:
    """
    Apply Non-Maximum Suppression
    
    Args:
        boxes: Array of bounding boxes [N, 4]
        scores: Array of confidence scores [N]
        iou_threshold: IoU threshold for suppression
        
    Returns:
        List of indices to keep
    """
    if len(boxes) == 0:
        return []
    
    # Sort by score
    order = scores.argsort()[::-1]
    
    keep = []
    while len(order) > 0:
        # Keep box with highest score
        i = order[0]
        keep.append(i)
        
        if len(order) == 1:
            break
        
        # Calculate IoU with remaining boxes
        ious = np.array([calculate_iou(boxes[i], boxes[j]) for j in order[1:]])
        
        # Keep boxes with IoU below threshold
        remaining = np.where(ious <= iou_threshold)[0]
        order = order[remaining + 1]
    
    return keep


def calculate_ap(
    predictions: List[Dict],
    ground_truths: List[Dict],
    iou_threshold: float = 0.5
) -> float:
    """
    Calculate Average Precision for a single class
    
    Args:
        predictions: List of predictions with bbox and score
        ground_truths: List of ground truths
        iou_threshold: IoU threshold for matching
        
    Returns:
        Average Precision score
    """
    if len(ground_truths) == 0:
        return 0.0
    
    # Sort predictions by score
    predictions = sorted(predictions, key=lambda x: x['score'], reverse=True)
    
    tp = np.zeros(len(predictions))
    fp = np.zeros(len(predictions))
    
    matched_gt = set()
    
    for i, pred in enumerate(predictions):
        best_iou = 0
        best_gt_idx = -1
        
        for j, gt in enumerate(ground_truths):
            if j not in matched_gt:
                iou = calculate_iou(
                    np.array(pred['bbox']),
                    np.array(gt['bbox'])
                )
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
        
        if best_iou >= iou_threshold:
            tp[i] = 1
            matched_gt.add(best_gt_idx)
        else:
            fp[i] = 1
    
    # Calculate precision and recall
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    recall = tp_cumsum / len(ground_truths)
    precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
    
    # Calculate AP using 11-point interpolation
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        precisions_above_t = precision[recall >= t]
        if len(precisions_above_t) > 0:
            ap += np.max(precisions_above_t) / 11
    
    return ap


def bbox_area(bbox: np.ndarray) -> float:
    """Calculate bounding box area"""
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def bbox_center(bbox: np.ndarray) -> Tuple[float, float]:
    """Calculate bounding box center"""
    x_center = (bbox[0] + bbox[2]) / 2
    y_center = (bbox[1] + bbox[3]) / 2
    return x_center, y_center

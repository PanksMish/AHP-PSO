"""
Prepare domain-specific datasets from MS COCO
Creates aerial, underwater, and road detection subsets
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import List, Dict
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
from pycocotools.coco import COCO


class DomainDatasetPreparer:
    """Prepare domain-specific datasets"""
    
    def __init__(self, coco_root: str, output_root: str):
        """
        Initialize preparer
        
        Args:
            coco_root: Root directory of COCO dataset
            output_root: Output directory for domain datasets
        """
        self.coco_root = Path(coco_root)
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        
        # Load COCO annotations
        ann_file = self.coco_root / 'annotations' / 'instances_val2017.json'
        self.coco = COCO(ann_file)
        self.img_dir = self.coco_root / 'val2017'
    
    def prepare_aerial_dataset(self, num_images: int = 1500):
        """
        Prepare aerial surveillance dataset
        
        Args:
            num_images: Number of images to select
        """
        print("\n" + "="*60)
        print("Preparing Aerial Surveillance Dataset")
        print("="*60)
        
        # Categories for aerial surveillance
        categories = ['person', 'car', 'truck', 'bus', 'motorcycle', 'bicycle']
        
        # Get category IDs
        cat_ids = self.coco.getCatIds(catNms=categories)
        print(f"Selected categories: {categories}")
        print(f"Category IDs: {cat_ids}")
        
        # Get images containing these categories
        img_ids = set()
        for cat_id in cat_ids:
            img_ids.update(self.coco.getImgIds(catIds=[cat_id]))
        
        img_ids = list(img_ids)[:num_images]
        print(f"Total images selected: {len(img_ids)}")
        
        # Create output directory
        output_dir = self.output_root / 'aerial'
        img_output_dir = output_dir / 'images'
        img_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process images
        annotations = {
            'images': [],
            'annotations': [],
            'categories': []
        }
        
        # Add categories
        for cat_id in cat_ids:
            cat_info = self.coco.loadCats([cat_id])[0]
            annotations['categories'].append(cat_info)
        
        ann_id = 0
        for img_id in tqdm(img_ids, desc="Processing aerial images"):
            # Load image info
            img_info = self.coco.loadImgs([img_id])[0]
            
            # Copy image
            src_path = self.img_dir / img_info['file_name']
            dst_path = img_output_dir / img_info['file_name']
            
            # Apply aerial-specific preprocessing
            img = cv2.imread(str(src_path))
            img = self._apply_aerial_preprocessing(img)
            cv2.imwrite(str(dst_path), img)
            
            # Add image info
            annotations['images'].append(img_info)
            
            # Get annotations for this image
            ann_ids = self.coco.getAnnIds(imgIds=[img_id], catIds=cat_ids)
            anns = self.coco.loadAnns(ann_ids)
            
            # Add annotations with new IDs
            for ann in anns:
                ann['id'] = ann_id
                annotations['annotations'].append(ann)
                ann_id += 1
        
        # Save annotations
        ann_file = output_dir / 'annotations.json'
        with open(ann_file, 'w') as f:
            json.dump(annotations, f, indent=2)
        
        print(f"✓ Aerial dataset created: {output_dir}")
        print(f"  Images: {len(annotations['images'])}")
        print(f"  Annotations: {len(annotations['annotations'])}")
    
    def prepare_underwater_dataset(self, num_images: int = 1000):
        """
        Prepare underwater object detection dataset
        
        Args:
            num_images: Number of images to select
        """
        print("\n" + "="*60)
        print("Preparing Underwater Object Detection Dataset")
        print("="*60)
        
        # Categories for underwater
        categories = ['person', 'boat', 'surfboard', 'bird', 'cat', 'dog', 'fish']
        
        # Get category IDs (fish might not exist in COCO)
        available_cats = [cat['name'] for cat in self.coco.loadCats(self.coco.getCatIds())]
        categories = [c for c in categories if c in available_cats]
        
        cat_ids = self.coco.getCatIds(catNms=categories)
        print(f"Selected categories: {categories}")
        
        # Get images
        img_ids = set()
        for cat_id in cat_ids:
            img_ids.update(self.coco.getImgIds(catIds=[cat_id]))
        
        img_ids = list(img_ids)[:num_images]
        print(f"Total images selected: {len(img_ids)}")
        
        # Create output directory
        output_dir = self.output_root / 'underwater'
        img_output_dir = output_dir / 'images'
        img_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process images
        annotations = {
            'images': [],
            'annotations': [],
            'categories': []
        }
        
        # Add categories
        for cat_id in cat_ids:
            cat_info = self.coco.loadCats([cat_id])[0]
            annotations['categories'].append(cat_info)
        
        ann_id = 0
        for img_id in tqdm(img_ids, desc="Processing underwater images"):
            img_info = self.coco.loadImgs([img_id])[0]
            
            # Copy and process image
            src_path = self.img_dir / img_info['file_name']
            dst_path = img_output_dir / img_info['file_name']
            
            # Apply underwater-specific preprocessing
            img = cv2.imread(str(src_path))
            img = self._apply_underwater_preprocessing(img)
            cv2.imwrite(str(dst_path), img)
            
            annotations['images'].append(img_info)
            
            # Get annotations
            ann_ids = self.coco.getAnnIds(imgIds=[img_id], catIds=cat_ids)
            anns = self.coco.loadAnns(ann_ids)
            
            for ann in anns:
                ann['id'] = ann_id
                annotations['annotations'].append(ann)
                ann_id += 1
        
        # Save annotations
        ann_file = output_dir / 'annotations.json'
        with open(ann_file, 'w') as f:
            json.dump(annotations, f, indent=2)
        
        print(f"✓ Underwater dataset created: {output_dir}")
        print(f"  Images: {len(annotations['images'])}")
        print(f"  Annotations: {len(annotations['annotations'])}")
    
    def prepare_road_dataset(self, num_images: int = 1000):
        """
        Prepare road monitoring dataset
        
        Args:
            num_images: Number of images to select
        """
        print("\n" + "="*60)
        print("Preparing Road Monitoring Dataset")
        print("="*60)
        
        # Categories for road monitoring
        categories = ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 
                     'traffic light', 'stop sign']
        
        cat_ids = self.coco.getCatIds(catNms=categories)
        print(f"Selected categories: {categories}")
        
        # Get images
        img_ids = set()
        for cat_id in cat_ids:
            img_ids.update(self.coco.getImgIds(catIds=[cat_id]))
        
        img_ids = list(img_ids)[:num_images]
        print(f"Total images selected: {len(img_ids)}")
        
        # Create output directory
        output_dir = self.output_root / 'road'
        img_output_dir = output_dir / 'images'
        img_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process images
        annotations = {
            'images': [],
            'annotations': [],
            'categories': []
        }
        
        # Add categories
        for cat_id in cat_ids:
            cat_info = self.coco.loadCats([cat_id])[0]
            annotations['categories'].append(cat_info)
        
        # Add synthetic pothole category
        annotations['categories'].append({
            'id': 1000,
            'name': 'pothole',
            'supercategory': 'road_damage'
        })
        
        ann_id = 0
        for img_id in tqdm(img_ids, desc="Processing road images"):
            img_info = self.coco.loadImgs([img_id])[0]
            
            # Copy and process image
            src_path = self.img_dir / img_info['file_name']
            dst_path = img_output_dir / img_info['file_name']
            
            # Apply road-specific preprocessing
            img = cv2.imread(str(src_path))
            img = self._apply_road_preprocessing(img)
            cv2.imwrite(str(dst_path), img)
            
            annotations['images'].append(img_info)
            
            # Get annotations
            ann_ids = self.coco.getAnnIds(imgIds=[img_id], catIds=cat_ids)
            anns = self.coco.loadAnns(ann_ids)
            
            for ann in anns:
                ann['id'] = ann_id
                annotations['annotations'].append(ann)
                ann_id += 1
            
            # Add synthetic potholes (randomly)
            if np.random.rand() < 0.2:  # 20% chance
                ann = self._generate_synthetic_pothole(img_info, ann_id)
                annotations['annotations'].append(ann)
                ann_id += 1
        
        # Save annotations
        ann_file = output_dir / 'annotations.json'
        with open(ann_file, 'w') as f:
            json.dump(annotations, f, indent=2)
        
        print(f"✓ Road dataset created: {output_dir}")
        print(f"  Images: {len(annotations['images'])}")
        print(f"  Annotations: {len(annotations['annotations'])}")
    
    def _apply_aerial_preprocessing(self, img: np.ndarray) -> np.ndarray:
        """Apply aerial-specific preprocessing"""
        # Random brightness and contrast
        if np.random.rand() < 0.5:
            alpha = np.random.uniform(0.7, 1.3)  # Brightness
            beta = np.random.uniform(-20, 20)     # Contrast
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        
        # Slight perspective distortion
        if np.random.rand() < 0.3:
            h, w = img.shape[:2]
            pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
            margin = int(0.05 * min(h, w))
            pts2 = np.float32([
                [np.random.randint(0, margin), np.random.randint(0, margin)],
                [w - np.random.randint(0, margin), np.random.randint(0, margin)],
                [np.random.randint(0, margin), h - np.random.randint(0, margin)],
                [w - np.random.randint(0, margin), h - np.random.randint(0, margin)]
            ])
            M = cv2.getPerspectiveTransform(pts1, pts2)
            img = cv2.warpPerspective(img, M, (w, h))
        
        return img
    
    def _apply_underwater_preprocessing(self, img: np.ndarray) -> np.ndarray:
        """Apply underwater-specific preprocessing"""
        img = img.astype(np.float32)
        
        # Blue-green color shift
        img[:, :, 0] *= 0.7  # Reduce red
        img[:, :, 1] *= 1.1  # Increase green
        img[:, :, 2] *= 1.3  # Increase blue
        img = np.clip(img, 0, 255).astype(np.uint8)
        
        # Reduce contrast
        img = cv2.convertScaleAbs(img, alpha=0.8, beta=10)
        
        # Add blur
        if np.random.rand() < 0.5:
            kernel_size = np.random.choice([3, 5, 7])
            img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
        
        return img
    
    def _apply_road_preprocessing(self, img: np.ndarray) -> np.ndarray:
        """Apply road-specific preprocessing"""
        # Add texture noise
        if np.random.rand() < 0.3:
            noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
            img = cv2.add(img, noise)
        
        # Weather effects
        if np.random.rand() < 0.2:
            # Fog effect
            fog = np.ones_like(img) * 200
            alpha = np.random.uniform(0.3, 0.5)
            img = cv2.addWeighted(img, 1-alpha, fog.astype(np.uint8), alpha, 0)
        
        return img
    
    def _generate_synthetic_pothole(self, img_info: Dict, ann_id: int) -> Dict:
        """Generate synthetic pothole annotation"""
        h, w = img_info['height'], img_info['width']
        
        # Random position and size
        pothole_w = np.random.randint(w // 20, w // 10)
        pothole_h = np.random.randint(h // 20, h // 10)
        x = np.random.randint(0, w - pothole_w)
        y = np.random.randint(h // 2, h - pothole_h)  # Lower half of image
        
        return {
            'id': ann_id,
            'image_id': img_info['id'],
            'category_id': 1000,  # Pothole category
            'bbox': [x, y, pothole_w, pothole_h],
            'area': pothole_w * pothole_h,
            'iscrowd': 0
        }


def main():
    parser = argparse.ArgumentParser(description='Prepare domain-specific datasets')
    parser.add_argument('--coco_root', type=str, default='data/coco',
                        help='Root directory of COCO dataset')
    parser.add_argument('--output_root', type=str, default='data/domain_specific',
                        help='Output root directory')
    parser.add_argument('--domain', type=str, default='all',
                        choices=['aerial', 'underwater', 'road', 'all'],
                        help='Domain to prepare')
    parser.add_argument('--num_images', type=int, default=None,
                        help='Number of images per domain')
    
    args = parser.parse_args()
    
    # Initialize preparer
    preparer = DomainDatasetPreparer(args.coco_root, args.output_root)
    
    # Prepare datasets
    if args.domain == 'all' or args.domain == 'aerial':
        num_images = args.num_images or 1500
        preparer.prepare_aerial_dataset(num_images)
    
    if args.domain == 'all' or args.domain == 'underwater':
        num_images = args.num_images or 1000
        preparer.prepare_underwater_dataset(num_images)
    
    if args.domain == 'all' or args.domain == 'road':
        num_images = args.num_images or 1000
        preparer.prepare_road_dataset(num_images)
    
    print("\n" + "="*60)
    print("Dataset Preparation Complete!")
    print("="*60)
    print(f"\nOutput directory: {args.output_root}")
    print("\nNext steps:")
    print("  1. Verify datasets: python scripts/verify_dataset.py")
    print("  2. Run experiments: python main.py --domain aerial")


if __name__ == '__main__':
    main()

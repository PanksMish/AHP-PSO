"""
Generate synthetic dataset for testing without downloading COCO
Creates simple synthetic images with bounding boxes
"""

import json
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
import argparse
from tqdm import tqdm


class SyntheticDatasetGenerator:
    """Generate synthetic object detection dataset"""
    
    def __init__(self, output_dir: str, img_size: tuple = (640, 480)):
        """
        Initialize generator
        
        Args:
            output_dir: Output directory
            img_size: Image size (width, height)
        """
        self.output_dir = Path(output_dir)
        self.img_size = img_size
        
        # Categories
        self.categories = [
            {'id': 1, 'name': 'person', 'supercategory': 'person'},
            {'id': 2, 'name': 'car', 'supercategory': 'vehicle'},
            {'id': 3, 'name': 'truck', 'supercategory': 'vehicle'},
            {'id': 4, 'name': 'bicycle', 'supercategory': 'vehicle'},
            {'id': 5, 'name': 'motorcycle', 'supercategory': 'vehicle'}
        ]
    
    def generate_random_shape(self, img_draw, category_id):
        """
        Generate random shape based on category
        
        Args:
            img_draw: ImageDraw object
            category_id: Category ID
            
        Returns:
            Bounding box [x, y, w, h]
        """
        # Random position and size
        max_w = self.img_size[0] // 4
        max_h = self.img_size[1] // 4
        
        w = np.random.randint(max_w // 2, max_w)
        h = np.random.randint(max_h // 2, max_h)
        x = np.random.randint(0, self.img_size[0] - w)
        y = np.random.randint(0, self.img_size[1] - h)
        
        # Random color
        color = tuple(np.random.randint(50, 200, 3).tolist())
        
        # Draw shape based on category
        if category_id == 1:  # Person - draw rectangle
            img_draw.rectangle([x, y, x+w, y+h], fill=color, outline='black')
        elif category_id in [2, 3]:  # Car/Truck - draw rounded rectangle
            img_draw.rounded_rectangle([x, y, x+w, y+h], radius=10, 
                                      fill=color, outline='black')
        elif category_id == 4:  # Bicycle - draw circle
            img_draw.ellipse([x, y, x+w, y+h], fill=color, outline='black')
        elif category_id == 5:  # Motorcycle - draw polygon
            points = [(x+w//2, y), (x+w, y+h//2), (x+w//2, y+h), (x, y+h//2)]
            img_draw.polygon(points, fill=color, outline='black')
        
        return [x, y, w, h]
    
    def generate_image(self, img_id: int, num_objects: int = None):
        """
        Generate single synthetic image
        
        Args:
            img_id: Image ID
            num_objects: Number of objects (random if None)
            
        Returns:
            Tuple of (image, annotations)
        """
        if num_objects is None:
            num_objects = np.random.randint(1, 6)
        
        # Create background
        bg_color = tuple(np.random.randint(200, 255, 3).tolist())
        img = Image.new('RGB', self.img_size, bg_color)
        draw = ImageDraw.Draw(img)
        
        # Generate objects
        annotations = []
        for i in range(num_objects):
            # Random category
            category = np.random.choice(self.categories)
            category_id = category['id']
            
            # Generate shape and get bbox
            bbox = self.generate_random_shape(draw, category_id)
            
            # Create annotation
            ann = {
                'id': img_id * 100 + i,
                'image_id': img_id,
                'category_id': category_id,
                'bbox': bbox,
                'area': bbox[2] * bbox[3],
                'iscrowd': 0
            }
            annotations.append(ann)
        
        # Image info
        img_info = {
            'id': img_id,
            'file_name': f'{img_id:06d}.jpg',
            'height': self.img_size[1],
            'width': self.img_size[0]
        }
        
        return img, img_info, annotations
    
    def generate_dataset(self, num_images: int, split: str = 'train'):
        """
        Generate complete dataset
        
        Args:
            num_images: Number of images to generate
            split: Dataset split name
        """
        print(f"\nGenerating {num_images} synthetic images for '{split}' split...")
        
        # Create directories
        img_dir = self.output_dir / f'{split}_images'
        img_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize annotations structure
        coco_format = {
            'images': [],
            'annotations': [],
            'categories': self.categories
        }
        
        # Generate images
        for img_id in tqdm(range(num_images), desc=f"Generating {split} set"):
            # Generate image and annotations
            img, img_info, annotations = self.generate_image(img_id)
            
            # Save image
            img_path = img_dir / img_info['file_name']
            img.save(img_path)
            
            # Add to COCO format
            coco_format['images'].append(img_info)
            coco_format['annotations'].extend(annotations)
        
        # Save annotations
        ann_file = self.output_dir / f'{split}_annotations.json'
        with open(ann_file, 'w') as f:
            json.dump(coco_format, f, indent=2)
        
        print(f"✓ Generated {num_images} images")
        print(f"✓ Total annotations: {len(coco_format['annotations'])}")
        print(f"✓ Saved to: {self.output_dir}")
    
    def generate_domain_specific(self, domain: str, num_images: int):
        """
        Generate domain-specific dataset
        
        Args:
            domain: Domain type
            num_images: Number of images
        """
        print(f"\nGenerating {domain} dataset...")
        
        # Create domain directory
        domain_dir = self.output_dir / domain
        img_dir = domain_dir / 'images'
        img_dir.mkdir(parents=True, exist_ok=True)
        
        # Domain-specific categories
        if domain == 'aerial':
            cat_filter = ['person', 'car', 'truck', 'bicycle', 'motorcycle']
        elif domain == 'underwater':
            cat_filter = ['person']  # Simplified
        elif domain == 'road':
            cat_filter = ['car', 'truck', 'bicycle', 'motorcycle']
        else:
            cat_filter = [c['name'] for c in self.categories]
        
        filtered_cats = [c for c in self.categories if c['name'] in cat_filter]
        
        # Initialize annotations
        coco_format = {
            'images': [],
            'annotations': [],
            'categories': filtered_cats
        }
        
        # Generate images
        for img_id in tqdm(range(num_images), desc=f"Generating {domain}"):
            img, img_info, annotations = self.generate_image(img_id)
            
            # Filter annotations by category
            filtered_anns = [
                ann for ann in annotations 
                if any(c['id'] == ann['category_id'] for c in filtered_cats)
            ]
            
            # Apply domain-specific effects
            if domain == 'underwater':
                img = self._apply_underwater_effect(img)
            elif domain == 'aerial':
                img = self._apply_aerial_effect(img)
            elif domain == 'road':
                img = self._apply_road_effect(img)
            
            # Save image
            img_path = img_dir / img_info['file_name']
            img.save(img_path)
            
            # Add to dataset
            coco_format['images'].append(img_info)
            coco_format['annotations'].extend(filtered_anns)
        
        # Save annotations
        ann_file = domain_dir / 'annotations.json'
        with open(ann_file, 'w') as f:
            json.dump(coco_format, f, indent=2)
        
        print(f"✓ {domain} dataset created: {domain_dir}")
    
    def _apply_underwater_effect(self, img: Image.Image) -> Image.Image:
        """Apply underwater color effect"""
        img_array = np.array(img).astype(np.float32)
        img_array[:, :, 0] *= 0.7  # Reduce red
        img_array[:, :, 1] *= 1.1  # Increase green
        img_array[:, :, 2] *= 1.3  # Increase blue
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        return Image.fromarray(img_array)
    
    def _apply_aerial_effect(self, img: Image.Image) -> Image.Image:
        """Apply aerial perspective effect"""
        # Just increase brightness slightly
        img_array = np.array(img).astype(np.float32)
        img_array = img_array * 1.1
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        return Image.fromarray(img_array)
    
    def _apply_road_effect(self, img: Image.Image) -> Image.Image:
        """Apply road texture effect"""
        # Add slight noise
        img_array = np.array(img).astype(np.float32)
        noise = np.random.normal(0, 5, img_array.shape)
        img_array = img_array + noise
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        return Image.fromarray(img_array)


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic dataset')
    parser.add_argument('--output_dir', type=str, default='data/synthetic',
                        help='Output directory')
    parser.add_argument('--num_train', type=int, default=100,
                        help='Number of training images')
    parser.add_argument('--num_val', type=int, default=50,
                        help='Number of validation images')
    parser.add_argument('--domain', type=str, default='all',
                        choices=['all', 'aerial', 'underwater', 'road', 'general'],
                        help='Generate domain-specific dataset')
    parser.add_argument('--img_size', type=int, nargs=2, default=[640, 480],
                        help='Image size (width height)')
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = SyntheticDatasetGenerator(
        output_dir=args.output_dir,
        img_size=tuple(args.img_size)
    )
    
    print("="*60)
    print("SYNTHETIC DATASET GENERATOR")
    print("="*60)
    print(f"Output directory: {args.output_dir}")
    print(f"Image size: {args.img_size[0]}x{args.img_size[1]}")
    
    if args.domain == 'general':
        # Generate train and val splits
        generator.generate_dataset(args.num_train, 'train')
        generator.generate_dataset(args.num_val, 'val')
    
    elif args.domain == 'all':
        # Generate all domain-specific datasets
        generator.generate_domain_specific('aerial', args.num_train)
        generator.generate_domain_specific('underwater', args.num_val)
        generator.generate_domain_specific('road', args.num_val)
    
    else:
        # Generate specific domain
        generator.generate_domain_specific(args.domain, args.num_train)
    
    print("\n" + "="*60)
    print("GENERATION COMPLETE")
    print("="*60)
    print(f"\nDataset saved to: {args.output_dir}")
    print("\nNext steps:")
    print("  1. Verify: python scripts/verify_dataset.py --root_dir data/synthetic")
    print("  2. Run test: python main.py --dataset_path data/synthetic --domain aerial")


if __name__ == '__main__':
    main()

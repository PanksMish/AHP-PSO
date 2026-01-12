"""
Verify dataset integrity and statistics
"""

import json
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns


def verify_domain_dataset(domain_path: Path):
    """
    Verify a domain-specific dataset
    
    Args:
        domain_path: Path to domain dataset directory
    """
    print(f"\n{'='*60}")
    print(f"Verifying {domain_path.name.upper()} Dataset")
    print('='*60)
    
    # Check directory structure
    img_dir = domain_path / 'images'
    ann_file = domain_path / 'annotations.json'
    
    if not img_dir.exists():
        print(f"✗ Images directory not found: {img_dir}")
        return False
    
    if not ann_file.exists():
        print(f"✗ Annotations file not found: {ann_file}")
        return False
    
    print(f"✓ Directory structure correct")
    
    # Load annotations
    with open(ann_file, 'r') as f:
        data = json.load(f)
    
    # Check components
    print(f"\nDataset Statistics:")
    print(f"  Images: {len(data['images'])}")
    print(f"  Annotations: {len(data['annotations'])}")
    print(f"  Categories: {len(data['categories'])}")
    
    # Check image files
    image_files = list(img_dir.glob('*.jpg'))
    print(f"  Image files on disk: {len(image_files)}")
    
    if len(image_files) != len(data['images']):
        print(f"  ⚠ Warning: Mismatch between annotations and files")
    
    # Category distribution
    print(f"\nCategory Distribution:")
    cat_counts = defaultdict(int)
    cat_names = {cat['id']: cat['name'] for cat in data['categories']}
    
    for ann in data['annotations']:
        cat_id = ann['category_id']
        cat_name = cat_names.get(cat_id, 'unknown')
        cat_counts[cat_name] += 1
    
    for cat_name, count in sorted(cat_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cat_name}: {count}")
    
    # Bounding box statistics
    bbox_areas = [ann['area'] for ann in data['annotations']]
    print(f"\nBounding Box Statistics:")
    print(f"  Mean area: {sum(bbox_areas)/len(bbox_areas):.1f} px²")
    print(f"  Min area: {min(bbox_areas):.1f} px²")
    print(f"  Max area: {max(bbox_areas):.1f} px²")
    
    return True


def generate_dataset_report(root_dir: Path, output_file: Path):
    """
    Generate comprehensive dataset report
    
    Args:
        root_dir: Root directory containing domain datasets
        output_file: Output file for report
    """
    print(f"\n{'='*60}")
    print("Generating Dataset Report")
    print('='*60)
    
    domains = ['aerial', 'underwater', 'road']
    report = []
    
    for domain in domains:
        domain_path = root_dir / domain
        if domain_path.exists():
            ann_file = domain_path / 'annotations.json'
            
            if ann_file.exists():
                with open(ann_file, 'r') as f:
                    data = json.load(f)
                
                report.append({
                    'domain': domain,
                    'num_images': len(data['images']),
                    'num_annotations': len(data['annotations']),
                    'num_categories': len(data['categories']),
                    'categories': [cat['name'] for cat in data['categories']]
                })
    
    # Save report
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✓ Report saved to: {output_file}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("DATASET SUMMARY")
    print('='*60)
    
    total_images = sum(r['num_images'] for r in report)
    total_annotations = sum(r['num_annotations'] for r in report)
    
    print(f"\nTotal across all domains:")
    print(f"  Images: {total_images}")
    print(f"  Annotations: {total_annotations}")
    
    print(f"\nPer domain:")
    for r in report:
        print(f"\n{r['domain'].upper()}:")
        print(f"  Images: {r['num_images']}")
        print(f"  Annotations: {r['num_annotations']}")
        print(f"  Categories: {', '.join(r['categories'][:5])}{'...' if len(r['categories']) > 5 else ''}")


def visualize_dataset_statistics(root_dir: Path, output_dir: Path):
    """
    Create visualizations of dataset statistics
    
    Args:
        root_dir: Root directory containing domain datasets
        output_dir: Output directory for plots
    """
    print(f"\n{'='*60}")
    print("Generating Visualizations")
    print('='*60)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    domains = ['aerial', 'underwater', 'road']
    domain_stats = {}
    
    # Collect statistics
    for domain in domains:
        domain_path = root_dir / domain
        ann_file = domain_path / 'annotations.json'
        
        if ann_file.exists():
            with open(ann_file, 'r') as f:
                data = json.load(f)
            
            domain_stats[domain] = {
                'num_images': len(data['images']),
                'num_annotations': len(data['annotations']),
                'num_categories': len(data['categories'])
            }
    
    # Create bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Images per domain
    domains_list = list(domain_stats.keys())
    images_count = [domain_stats[d]['num_images'] for d in domains_list]
    
    ax1.bar(domains_list, images_count, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax1.set_ylabel('Number of Images', fontsize=12)
    ax1.set_title('Images per Domain', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(images_count):
        ax1.text(i, v + 20, str(v), ha='center', va='bottom', fontweight='bold')
    
    # Annotations per domain
    annotations_count = [domain_stats[d]['num_annotations'] for d in domains_list]
    
    ax2.bar(domains_list, annotations_count, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax2.set_ylabel('Number of Annotations', fontsize=12)
    ax2.set_title('Annotations per Domain', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(annotations_count):
        ax2.text(i, v + 50, str(v), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    output_file = output_dir / 'dataset_statistics.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Visualization saved to: {output_file}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Verify dataset integrity')
    parser.add_argument('--root_dir', type=str, default='data/domain_specific',
                        help='Root directory of domain datasets')
    parser.add_argument('--report', action='store_true',
                        help='Generate comprehensive report')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualizations')
    
    args = parser.parse_args()
    
    root_dir = Path(args.root_dir)
    
    if not root_dir.exists():
        print(f"Error: Directory not found: {root_dir}")
        print("Please run: python scripts/prepare_domains.py --domain all")
        return
    
    # Verify each domain
    domains = ['aerial', 'underwater', 'road']
    all_valid = True
    
    for domain in domains:
        domain_path = root_dir / domain
        if domain_path.exists():
            valid = verify_domain_dataset(domain_path)
            all_valid = all_valid and valid
        else:
            print(f"\n✗ {domain.upper()} dataset not found at {domain_path}")
            all_valid = False
    
    # Generate report
    if args.report:
        report_file = root_dir / 'dataset_report.json'
        generate_dataset_report(root_dir, report_file)
    
    # Generate visualizations
    if args.visualize:
        output_dir = root_dir / 'visualizations'
        visualize_dataset_statistics(root_dir, output_dir)
    
    # Final status
    print(f"\n{'='*60}")
    if all_valid:
        print("✓ All datasets verified successfully!")
    else:
        print("✗ Some datasets have issues. Please check above.")
    print('='*60)


if __name__ == '__main__':
    main()

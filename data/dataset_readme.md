# Dataset Guide for AHP-PSO Implementation

## 📊 Overview

This implementation uses **MS COCO 2017** dataset with domain-specific adaptations for:
- 🚁 **Aerial Surveillance** (1,500 images)
- 🌊 **Underwater Detection** (1,000 images)  
- 🚗 **Road Monitoring** (1,000 images)

## 🎯 Quick Start

### Option 1: Download Real COCO Dataset (Recommended)

```bash
# Download validation set + annotations (~1.2 GB)
python scripts/download_coco.py --val --annotations --output data/coco

# Prepare domain-specific subsets
python scripts/prepare_domains.py --domain all --coco_root data/coco
```

### Option 2: Generate Synthetic Dataset (For Testing)

```bash
# Generate synthetic dataset (no download required)
python scripts/generate_synthetic_dataset.py --domain all --num_train 100 --num_val 50
```

## 📥 Complete Dataset Download Guide

### Step 1: Download MS COCO 2017

**Required Files:**
```bash
# Validation images (1GB)
wget http://images.cocodataset.org/zips/val2017.zip

# Annotations (241MB)
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
```

**OR use our automated script:**
```bash
python scripts/download_coco.py --val --annotations --output data/coco
```

### Step 2: Extract Files

```bash
cd data/coco
unzip val2017.zip
unzip annotations_trainval2017.zip
```

Expected structure:
```
data/coco/
├── val2017/           # 5,000 images
│   ├── 000000000139.jpg
│   ├── 000000000285.jpg
│   └── ...
└── annotations/
    └── instances_val2017.json
```

### Step 3: Prepare Domain Datasets

```bash
# Prepare all domains
python scripts/prepare_domains.py --domain all --coco_root data/coco

# Or prepare specific domain
python scripts/prepare_domains.py --domain aerial --num_images 1500
python scripts/prepare_domains.py --domain underwater --num_images 1000
python scripts/prepare_domains.py --domain road --num_images 1000
```

### Step 4: Verify Dataset

```bash
python scripts/verify_dataset.py --root_dir data/domain_specific --report --visualize
```

## 📁 Final Directory Structure

```
data/
├── coco/                          # Original MS COCO 2017
│   ├── val2017/                   # 5,000 validation images
│   └── annotations/
│       └── instances_val2017.json
│
├── domain_specific/               # Domain-adapted datasets
│   ├── aerial/
│   │   ├── images/                # 1,500 images
│   │   └── annotations.json
│   │
│   ├── underwater/
│   │   ├── images/                # 1,000 images (with blue-green tint)
│   │   └── annotations.json
│   │
│   └── road/
│       ├── images/                # 1,000 images (with texture effects)
│       └── annotations.json
│
└── synthetic/                     # Optional: synthetic test data
    ├── aerial/
    ├── underwater/
    └── road/
```

## 🎨 Domain-Specific Adaptations

### Aerial Surveillance
**Categories:** person, car, truck, bus, motorcycle, bicycle

**Preprocessing:**
- Perspective warping (simulates drone angle)
- Brightness variation (0.7-1.3x)
- Contrast adjustment (±20)
- Altitude simulation

**Use Cases:**
- Drone monitoring
- Traffic surveillance
- Crowd detection

### Underwater Detection
**Categories:** person, boat, surfboard, bird, cat, dog

**Preprocessing:**
- Blue-green color shift (0.7R, 1.1G, 1.3B)
- Gaussian blur (kernel 3-7)
- Contrast reduction (0.6-1.0x)
- Hue shift (±20°)

**Use Cases:**
- Marine life detection
- Underwater robotics
- Submarine navigation

### Road Monitoring
**Categories:** car, truck, bus, motorcycle, bicycle, traffic light, stop sign

**Preprocessing:**
- Texture noise addition
- Weather effects (fog, haze)
- Surface degradation
- Synthetic potholes (20% of images)

**Use Cases:**
- Pothole detection
- Traffic monitoring
- Road damage assessment

## 📊 Dataset Statistics

### MS COCO 2017 (Source)
| Metric | Value |
|--------|-------|
| Training Images | 118,287 |
| Validation Images | 5,000 |
| Object Categories | 80 |
| Total Annotations | 1.5M+ |
| Small Objects | 41.4% |
| Medium Objects | 34.3% |
| Large Objects | 24.3% |

### Domain-Specific Subsets (Paper Setup)
| Domain | Images | Annotations | Categories |
|--------|--------|-------------|------------|
| Aerial | 1,500 | ~8,000 | 6 |
| Underwater | 1,000 | ~5,000 | 6 |
| Road | 1,000 | ~6,000 | 7+1 (pothole) |

## 🔗 External Dataset Sources

### Optional Reference Datasets

**Underwater:**
- UIEB Dataset: https://li-chongyi.github.io/proj_benchmark.html
- RUIE Dataset: https://github.com/dlut-dimt/Realworld-Underwater-Image-Enhancement-RUIE-Benchmark

**Road Damage:**
- Road Damage Dataset 2022: https://github.com/sekilab/RoadDamageDetector
- CrackForest: https://github.com/cuilimeng/CrackForest-dataset
- Gap Dataset: https://github.com/fyangneil/pavement-crack-detection

**Aerial:**
- VisDrone: https://github.com/VisDrone/VisDrone-Dataset
- UAVDT: https://sites.google.com/view/grli-uavdt

## 🧪 Synthetic Dataset (For Quick Testing)

If you want to test the code without downloading COCO:

```bash
# Generate 100 training + 50 validation images
python scripts/generate_synthetic_dataset.py \
    --domain all \
    --num_train 100 \
    --num_val 50 \
    --output_dir data/synthetic

# Run with synthetic data
python main.py --dataset_path data/synthetic/aerial --domain aerial
```

**Features:**
- ✅ No download required
- ✅ COCO-compatible format
- ✅ Quick generation (<1 minute)
- ✅ Configurable image size
- ⚠️ Simplified shapes (not real objects)

## 📖 Usage in Code

### Loading Domain Dataset

```python
from src.data_loader import COCODataLoader
from src.config import Config

config = Config(domain='aerial')
data_loader = COCODataLoader(
    dataset_path='data/domain_specific/aerial',
    domain='aerial',
    batch_size=16,
    config=config
)

# Iterate through batches
for batch in data_loader.val_loader:
    images = batch['image']
    annotations = batch['annotations']
    # Process batch...
```

### Custom Dataset Path

```python
python main.py \
    --algorithm ahp_pso \
    --domain aerial \
    --dataset_path data/domain_specific/aerial \
    --visualize
```

## ✅ Verification Checklist

After dataset preparation:

- [ ] COCO validation images extracted (5,000 files)
- [ ] Annotations file exists (`instances_val2017.json`)
- [ ] Domain subsets created (aerial, underwater, road)
- [ ] Each domain has `images/` folder and `annotations.json`
- [ ] Verification script passes: `python scripts/verify_dataset.py`
- [ ] Sample images viewable with correct preprocessing

## 🐛 Troubleshooting

### Issue: Download fails
```bash
# Use mirror sites or manual download
# Check https://cocodataset.org/#download for alternatives
```

### Issue: Out of disk space
```bash
# Use validation set only (1GB instead of 19GB)
python scripts/download_coco.py --val --annotations

# Or use synthetic dataset
python scripts/generate_synthetic_dataset.py --domain all
```

### Issue: Missing pycocotools
```bash
pip install pycocotools
```

### Issue: Domain preparation fails
```bash
# Check COCO dataset exists
ls data/coco/val2017/  # Should show 5000 images

# Check annotations
python -c "from pycocotools.coco import COCO; coco = COCO('data/coco/annotations/instances_val2017.json'); print(f'{len(coco.getImgIds())} images')"
```

## 📧 Support

For dataset issues:
- **COCO Dataset:** https://cocodataset.org/#download
- **GitHub Issues:** https://github.com/yourusername/ahp-pso-detection/issues
- **Email:** pankaj.mishra@somaiya.edu

## 📚 Citations

### MS COCO
```bibtex
@inproceedings{lin2014microsoft,
  title={Microsoft coco: Common objects in context},
  author={Lin, Tsung-Yi and Maire, Michael and Belongie, Serge and others},
  booktitle={ECCV},
  pages={740--755},
  year={2014}
}
```

### This Work
```bibtex
@article{mishra2024ahppso,
  title={AHP-PSO: Adaptive Hybrid Particle Swarm Optimization for Real-Time Cross-Domain Object Detection},
  author={Mishra, Pankaj and Venkataramanan, V and Nayyar, Anand},
  note={Uses MS COCO 2017 with domain adaptations},
  year={2024}
}
```

---

**Dataset Version:** MS COCO 2017  
**Total Size:** ~1.2 GB (validation + annotations)  
**Last Updated:** December 2024

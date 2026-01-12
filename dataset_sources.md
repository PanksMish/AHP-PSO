# Dataset Sources for AHP-PSO Implementation

## 📊 Primary Dataset: MS COCO 2017

### Official Source
- **Dataset Name:** Microsoft Common Objects in Context (MS COCO) 2017
- **Paper Reference:** Section 3.1.1, Table 1
- **Official Website:** https://cocodataset.org/

### Download URLs

#### Images
```
Training Images (18GB):
http://images.cocodataset.org/zips/train2017.zip

Validation Images (1GB):
http://images.cocodataset.org/zips/val2017.zip

Test Images (6GB):
http://images.cocodataset.org/zips/test2017.zip
```

#### Annotations
```
Training/Validation Annotations (241MB):
http://images.cocodataset.org/annotations/annotations_trainval2017.zip

Test Image Info (1MB):
http://images.cocodataset.org/annotations/image_info_test2017.zip
```

### Dataset Statistics (from Paper Table 1)
- **Training Images:** 118,287
- **Validation Images:** 5,000
- **Object Categories:** 80 (in 12 super-categories)
- **Annotation Types:** Bounding boxes, segmentation masks
- **Object Instances:** 1.5 million+
- **Size Distribution:**
  - Small objects: 41.4%
  - Medium objects: 34.3%
  - Large objects: 24.3%

---

## 🎯 Domain-Specific Subsets

### 1. Aerial Surveillance Dataset
**Paper Reference:** Section 3.1.5, Table 2
**Source:** Filtered from COCO 2017
**Size:** 1,500 images

**Categories Used:**
- person
- car
- truck
- bus
- motorcycle
- bicycle

**Download Script:**
```bash
python scripts/prepare_domains.py --domain aerial --output data/domain_specific/aerial
```

---

### 2. Underwater Object Detection Dataset
**Paper Reference:** Section 3.1.5, Table 2
**Source:** COCO 2017 + Synthetic Augmentation
**Size:** 1,000 images

**Categories Used:**
- person
- boat
- surfboard
- bird
- cat
- dog

**Preprocessing Applied:**
- Color shifting (blue-green tint)
- Gaussian blur (kernel size 3-7)
- Contrast reduction (0.6-1.0)
- Hue shift (-20 to +20 degrees)

**Additional References:**
- Underwater Image Enhancement: https://github.com/cameron-git/underwater-image-enhancement
- UIEB Dataset: https://li-chongyi.github.io/proj_benchmark.html

---

### 3. Road Surface / Pothole Detection Dataset
**Paper Reference:** Section 3.1.5, Table 2
**Source:** COCO 2017 + Synthetic Potholes
**Size:** 800 road images + 200 synthetic potholes

**Categories Used:**
- car
- truck
- bus
- motorcycle
- bicycle
- traffic light
- stop sign

**Additional Datasets (for texture reference):**
- **CrackForest Dataset:** https://github.com/cuilimeng/CrackForest-dataset
- **Road Damage Dataset:** https://github.com/sekilab/RoadDamageDetector

**Synthetic Generation:**
- Depth map simulation
- Crack texture overlay
- Surface degradation patterns

---

## 📥 Complete Download Script

Save as `data/DOWNLOAD_DATASETS.txt`:

```text
# MS COCO 2017 Dataset Download URLs

# ==================================================
# VALIDATION SET (Used in paper experiments)
# ==================================================
Validation Images (1GB):
http://images.cocodataset.org/zips/val2017.zip

Annotations:
http://images.cocodataset.org/annotations/annotations_trainval2017.zip


# ==================================================
# TRAINING SET (Optional - for extended experiments)
# ==================================================
Training Images (18GB):
http://images.cocodataset.org/zips/train2017.zip


# ==================================================
# ADDITIONAL DATASETS (For reference)
# ==================================================

# Underwater datasets
UIEB Dataset:
https://li-chongyi.github.io/proj_benchmark.html

# Road damage datasets  
Road Damage Dataset:
https://github.com/sekilab/RoadDamageDetector/releases

CrackForest Dataset:
https://github.com/cuilimeng/CrackForest-dataset


# ==================================================
# USAGE
# ==================================================
# Use our automated download script:
python scripts/download_coco.py --val --annotations

# Or download manually using wget:
cd data/coco
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

# Extract:
unzip val2017.zip
unzip annotations_trainval2017.zip
```

---

## 🔄 Dataset Preparation Pipeline

### Step 1: Download COCO
```bash
cd ahp-pso-detection
python scripts/download_coco.py --val --annotations --output data/coco
```

### Step 2: Prepare Domain-Specific Subsets
```bash
# Aerial
python scripts/prepare_domains.py --domain aerial --num_images 1500

# Underwater  
python scripts/prepare_domains.py --domain underwater --num_images 1000

# Road
python scripts/prepare_domains.py --domain road --num_images 1000
```

### Step 3: Verify Dataset
```bash
python scripts/verify_dataset.py
```

---

## 📊 Expected Directory Structure After Download

```
data/
├── coco/
│   ├── val2017/                    # 5,000 images
│   │   ├── 000000000139.jpg
│   │   ├── 000000000285.jpg
│   │   └── ...
│   └── annotations/
│       └── instances_val2017.json  # Annotations
│
├── domain_specific/
│   ├── aerial/
│   │   ├── images/                 # 1,500 filtered images
│   │   └── annotations.json
│   │
│   ├── underwater/
│   │   ├── images/                 # 1,000 augmented images
│   │   └── annotations.json
│   │
│   └── road/
│       ├── images/                 # 1,000 images
│       └── annotations.json
│
└── DOWNLOAD_DATASETS.txt           # This file
```

---

## 📋 Dataset Split (Used in Paper)

### Training/Validation Split
According to paper Section 5.1:
- **Training:** 80% of domain-specific subset
- **Validation:** 20% of domain-specific subset

### Evaluation Set Sizes (Table 2)
| Domain | Total Images | Training | Validation |
|--------|-------------|----------|------------|
| Aerial | 1,500 | 1,200 | 300 |
| Underwater | 1,000 | 800 | 200 |
| Road | 1,000 | 800 | 200 |
| General | 5,000 | 4,000 | 1,000 |

---

## ⚙️ Dataset Citation

### MS COCO
```bibtex
@inproceedings{lin2014microsoft,
  title={Microsoft coco: Common objects in context},
  author={Lin, Tsung-Yi and Maire, Michael and Belongie, Serge and others},
  booktitle={European conference on computer vision},
  pages={740--755},
  year={2014},
  organization={Springer}
}
```

### Paper Dataset Usage
```bibtex
@article{mishra2024ahppso,
  title={AHP-PSO: Adaptive Hybrid Particle Swarm Optimization for Real-Time Cross-Domain Object Detection},
  author={Mishra, Pankaj and Venkataramanan, V and Nayyar, Anand},
  note={Dataset: MS COCO 2017 with domain-specific adaptations},
  year={2024}
}
```

---

## 🔍 Dataset Verification

After downloading, verify the dataset:

```bash
python -c "
from pycocotools.coco import COCO
coco = COCO('data/coco/annotations/instances_val2017.json')
print(f'Total images: {len(coco.getImgIds())}')
print(f'Total categories: {len(coco.getCatIds())}')
print('Dataset loaded successfully!')
"
```

Expected output:
```
Total images: 5000
Total categories: 80
Dataset loaded successfully!
```

---

## 📞 Support

For dataset issues:
- COCO Dataset: https://cocodataset.org/#download
- GitHub Issues: https://github.com/yourusername/ahp-pso-detection/issues

---

**Last Updated:** December 2024
**COCO Version:** 2017
**Total Download Size:** ~1.2 GB (validation set + annotations)

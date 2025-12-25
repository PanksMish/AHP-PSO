# AHP-PSO: Adaptive Hybrid Particle Swarm Optimization for Real-Time Object Detection

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-red.svg)](https://pytorch.org/)

Official implementation of "AHP-PSO: Adaptive Hybrid Particle Swarm Optimization for Real-Time Cross-Domain Object Detection"

**Authors:** Pankaj Mishra, V Venkataramanan, Anand Nayyar

## 📋 Abstract

Real-time object detection is critical in smart systems like self-driving cars, robots, and security. This work introduces **AHP-PSO** (Adaptive Hybrid Particle Swarm Optimizer), a novel paradigm that unites quantum-behaved particle dynamics, motion-aware updates, and fitness-weighted inertia modulation. The framework achieves:

- 🎯 **12% improved accuracy** over standard PSO
- ⚡ **18% faster convergence**
- 📊 **9% better IoU**
- 🚀 **15 FPS** real-time performance
- 💪 **0.80 F1-score** across diverse environments

## 🌟 Key Features

- **Adaptive Inertia Modulation**: Per-particle fitness-weighted inertia for balanced exploration-exploitation
- **AHP-based Multi-Criteria Optimization**: Hierarchical decision-making for domain-aware fitness evaluation
- **Dynamic Population Adjustment**: Automatic swarm size adaptation based on convergence feedback
- **Cross-Domain Support**: Specialized configurations for aerial, underwater, and road scenarios
- **Real-time Performance**: Optimized for edge deployment with <50ms inference time

## 🏗️ Repository Structure

```
ahp-pso-detection/
├── data/
│   ├── coco/                    # COCO dataset (download separately)
│   ├── processed/               # Preprocessed data
│   └── domain_specific/         # Domain-adapted datasets
├── src/
│   ├── algorithms/
│   │   ├── base_pso.py         # Base PSO implementation
│   │   ├── classical_pso.py    # Classical PSO
│   │   ├── quantum_pso.py      # Quantum PSO (QPSO)
│   │   ├── sequential_pso.py   # Sequential PSO (SPSO)
│   │   └── ahp_pso.py          # Proposed AHP-PSO
│   ├── data_loader.py          # Dataset loading and preprocessing
│   ├── config.py               # Configuration management
│   ├── evaluator.py            # Evaluation metrics
│   └── utils/
│       ├── ahp.py              # AHP calculator
│       ├── metrics.py          # Performance metrics
│       ├── logger.py           # Logging utilities
│       └── visualization.py    # Visualization tools
├── experiments/
│   ├── run_all_algorithms.py   # Run all algorithm comparisons
│   ├── benchmark.py            # Benchmarking suite
│   └── ablation_study.py       # Ablation experiments
├── results/                     # Experimental results
├── notebooks/
│   ├── demo.ipynb              # Interactive demo
│   └── analysis.ipynb          # Results analysis
├── tests/                       # Unit tests
├── main.py                      # Main entry point
├── requirements.txt             # Python dependencies
├── setup.py                     # Package installation
└── README.md                    # This file
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ahp-pso-detection.git
cd ahp-pso-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

### Dataset Setup

1. Download MS COCO 2017 dataset:
```bash
# Download script provided
python scripts/download_coco.py --output data/coco
```

2. Preprocess for domain-specific tasks:
```bash
python scripts/prepare_domains.py --domain aerial
python scripts/prepare_domains.py --domain underwater
python scripts/prepare_domains.py --domain road
```

### Running Experiments

**Single Algorithm:**
```bash
# Run AHP-PSO on aerial domain
python main.py --algorithm ahp_pso --domain aerial --visualize

# Run Quantum PSO on underwater domain
python main.py --algorithm quantum_pso --domain underwater

# Run Sequential PSO with custom parameters
python main.py --algorithm sequential_pso --domain road \
    --num_particles 60 --max_iterations 150
```

**Benchmark All Algorithms:**
```bash
python experiments/run_all_algorithms.py --domain aerial
```

**Ablation Study:**
```bash
python experiments/ablation_study.py
```

## 📊 Experimental Results

### Performance Comparison

| Algorithm | Precision | Recall | F1-Score | mAP | IoU | FPS |
|-----------|-----------|--------|----------|-----|-----|-----|
| Classical PSO | 0.82 | 0.79 | 0.80 | 0.80 | 0.78 | 18 |
| Quantum PSO | 0.85 | 0.83 | 0.84 | 0.83 | 0.80 | 16 |
| Sequential PSO | 0.89 | 0.85 | 0.87 | 0.87 | 0.82 | 24 |
| **AHP-PSO (Ours)** | **0.92** | **0.89** | **0.90** | **0.89** | **0.84** | **15** |
| PSO + Conv Forest | 0.91 | 0.88 | 0.89 | 0.89 | 0.84 | 18 |

### Domain-Specific Results

**Aerial Surveillance:**
- Altitude Invariance: 0.89
- Scalability: 0.91
- Weather Robustness: 0.85

**Underwater Detection:**
- Turbidity Resistance: 0.88
- Light Refraction Handling: 0.91
- Color Correction: 0.86

**Road Monitoring:**
- Texture Recognition: 0.87
- False Alarm Resistance: 0.90
- Real-time Processing: 18 FPS

### Convergence Analysis

AHP-PSO achieves 90% optimal fitness by iteration 120, demonstrating:
- **18% faster convergence** than Classical PSO
- **Superior stability** in dynamic environments
- **Adaptive behavior** under varying scene complexity

## 🔬 Key Algorithms

### 1. Classical PSO
Standard particle swarm optimization with velocity and position updates.

### 2. Quantum PSO (QPSO)
Quantum-behaved particles using wave functions for enhanced exploration.

### 3. Sequential PSO (SPSO)
Motion-aware updates with temporal consistency for video sequences.

### 4. AHP-PSO (Proposed)
Hybrid approach combining:
- Analytic Hierarchy Process for multi-criteria decision making
- Adaptive inertia weights per particle
- Dynamic population size adjustment
- Domain-specific fitness modeling

## 📈 Usage Examples

### Basic Detection

```python
from src.algorithms.ahp_pso import AHPPSO
from src.config import Config
from src.data_loader import COCODataLoader

# Initialize
config = Config(domain='aerial', num_particles=50, max_iterations=150)
algorithm = AHPPSO(config)
data_loader = COCODataLoader('data/coco', domain='aerial', config=config)

# Run detection
for batch in data_loader.val_loader:
    predictions, inference_time = algorithm.detect(
        batch['image'], 
        batch['annotations']
    )
    print(f"Detected {len(predictions)} objects in {inference_time:.3f}s")
```

### Custom Domain Configuration

```python
from src.config import Config

config = Config(
    domain='custom',
    num_particles=60,
    max_iterations=200,
    domain_params={
        'fitness_weights': {
            'mAP': 0.5,
            'precision': 0.3,
            'fps': 0.2
        },
        'augmentation': {
            'brightness': (0.7, 1.3),
            'gaussian_noise': 0.02
        }
    }
)
```

## 🧪 Evaluation Metrics

The implementation includes comprehensive metrics:

- **Detection Accuracy**: Precision, Recall, F1-Score
- **Localization Quality**: IoU, mAP@0.5, mAP@0.5:0.95
- **Temporal Consistency**: Frame-to-frame stability
- **Computational Efficiency**: FPS, Inference Time, Memory Usage
- **Robustness**: Occlusion handling, weather invariance
- **Domain-Specific**: Altitude invariance, turbidity resistance, texture recognition

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@article{mishra2024ahppso,
  title={AHP-PSO: Adaptive Hybrid Particle Swarm Optimization for Real-Time Cross-Domain Object Detection},
  author={Mishra, Pankaj and Venkataramanan, V and Nayyar, Anand},
  journal={Journal Name},
  year={2024}
}
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

Please ensure:
- Code follows PEP 8 style guidelines
- All tests pass (`pytest tests/`)
- Documentation is updated

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Pankaj Mishra** - K J Somaiya School of Engineering - pankaj.mishra@somaiya.edu
- **V Venkataramanan** - K J Somaiya School of Engineering - venkataramanan@somaiya.edu
- **Anand Nayyar** - Duy Tan University - anandnayyar@duytan.edu.vn

## 🙏 Acknowledgments

- MS COCO dataset team for providing the benchmark dataset
- PyTorch team for the deep learning framework
- Research community for valuable feedback and suggestions

## 📧 Contact

For questions, issues, or collaborations:
- Open an issue on GitHub
- Email: pankaj.mishra@somaiya.edu

## 🔗 Related Resources

- [Project Page](https://yourusername.github.io/ahp-pso-detection)
- [Paper (arXiv)](https://arxiv.org/abs/xxxx.xxxxx)
- [Video Demo](https://youtube.com/watch?v=xxxxx)
- [Supplementary Material](https://drive.google.com/xxxxx)

---

**Keywords:** Adaptive PSO, Object Detection, Particle Swarm Optimization, Real-Time Tracking, Swarm Intelligence, Video Surveillance, AHP, Multi-Criteria Optimization

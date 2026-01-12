# AHP-PSO Complete Project Structure & Implementation Guide

## 📁 Complete Directory Structure

```
ahp-pso-detection/
│
├── 📄 README.md                      # Main project documentation
├── 📄 LICENSE                        # MIT License
├── 📄 setup.py                       # Package installation
├── 📄 requirements.txt               # Python dependencies
├── 📄 Dockerfile                     # Docker configuration
├── 📄 .gitignore                     # Git ignore rules
├── 📄 PROJECT_STRUCTURE.md          # This file
│
├── 📂 .github/
│   └── workflows/
│       └── ci.yml                    # CI/CD pipeline
│
├── 📂 src/                           # Source code
│   ├── __init__.py
│   ├── config.py                     # Configuration management
│   ├── data_loader.py                # Dataset loading
│   ├── evaluator.py                  # Evaluation metrics
│   │
│   ├── algorithms/                   # PSO algorithms
│   │   ├── __init__.py
│   │   ├── base_pso.py              # Base PSO class
│   │   ├── classical_pso.py         # Classical PSO
│   │   ├── quantum_pso.py           # Quantum PSO (QPSO)
│   │   ├── sequential_pso.py        # Sequential PSO (SPSO)
│   │   └── ahp_pso.py               # Proposed AHP-PSO
│   │
│   └── utils/                        # Utility functions
│       ├── __init__.py
│       ├── ahp.py                    # AHP calculator
│       ├── metrics.py                # Metric calculations
│       ├── logger.py                 # Logging utilities
│       └── visualization.py          # Visualization tools
│
├── 📂 experiments/                   # Experiment scripts
│   ├── __init__.py
│   ├── run_all_algorithms.py        # Benchmark all algorithms
│   ├── ablation_study.py            # Ablation experiments
│   └── domain_comparison.py         # Cross-domain evaluation
│
├── 📂 scripts/                       # Utility scripts
│   ├── download_coco.py             # Download COCO dataset
│   ├── prepare_domains.py           # Prepare domain-specific data
│   ├── install.sh                   # Installation script
│   └── run_docker.sh                # Docker run script
│
├── 📂 tests/                         # Unit tests
│   ├── __init__.py
│   ├── test_algorithms.py
│   ├── test_metrics.py
│   ├── test_data_loader.py
│   └── test_evaluator.py
│
├── 📂 notebooks/                     # Jupyter notebooks
│   ├── demo.ipynb                   # Interactive demo
│   ├── analysis.ipynb               # Results analysis
│   └── visualization.ipynb          # Visualization examples
│
├── 📂 data/                          # Data directory
│   ├── coco/                        # COCO dataset
│   │   ├── train2017/
│   │   ├── val2017/
│   │   └── annotations/
│   ├── processed/                   # Preprocessed data
│   └── domain_specific/             # Domain-adapted datasets
│       ├── aerial/
│       ├── underwater/
│       └── road/
│
├── 📂 results/                       # Experimental results
│   ├── benchmark/
│   ├── ablation/
│   └── visualizations/
│
├── 📂 logs/                          # Log files
│
├── 📂 models/                        # Saved models (optional)
│
└── 📄 main.py                        # Main entry point
```

## 🚀 Quick Start Guide

### 1. Installation

```bash
# Clone repository
git clone https://github.com/yourusername/ahp-pso-detection.git
cd ahp-pso-detection

# Run installation script
chmod +x scripts/install.sh
./scripts/install.sh
```

### 2. Data Preparation

```bash
# Download COCO dataset
python scripts/download_coco.py --output data/coco --val

# Prepare domain-specific datasets
python scripts/prepare_domains.py --domain aerial
python scripts/prepare_domains.py --domain underwater
python scripts/prepare_domains.py --domain road
```

### 3. Run Experiments

```bash
# Single algorithm
python main.py --algorithm ahp_pso --domain aerial --visualize

# All algorithms benchmark
python experiments/run_all_algorithms.py --domain aerial

# Ablation study
python experiments/ablation_study.py
```

## 📊 Reproducing Paper Results

### Table 3: Performance Metrics Comparison

```bash
# Run complete benchmark
python experiments/run_all_algorithms.py --domain general --num_images 500
```

Expected output matches Table 3 from paper:
- Classical PSO: Precision=0.82, Recall=0.79, F1=0.80
- Quantum PSO: Precision=0.85, Recall=0.83, F1=0.84
- Sequential PSO: Precision=0.89, Recall=0.85, F1=0.87
- AHP-PSO: Precision=0.92, Recall=0.89, F1=0.90

### Figure 3: Convergence Analysis

```bash
python experiments/convergence_analysis.py
```

### Figure 5: Precision-Recall Curves

```bash
python experiments/generate_pr_curves.py
```

## 🔧 Configuration

### Domain-Specific Settings

Edit `src/config.py` to customize domain parameters:

```python
# Aerial surveillance
'aerial': {
    'num_particles': 50,
    'max_iterations': 180,
    'fitness_weights': {
        'mAP': 0.5,
        'fps': 0.3,
        'iou': 0.2
    }
}

# Underwater detection
'underwater': {
    'num_particles': 60,
    'max_iterations': 210,
    'fitness_weights': {
        'mAP': 0.4,
        'contrast_score': 0.3,
        'edge_preservation': 0.3
    }
}

# Road monitoring
'road': {
    'num_particles': 50,
    'max_iterations': 150,
    'fitness_weights': {
        'precision': 0.4,
        'recall': 0.3,
        'texture_score': 0.3
    }
}
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_algorithms.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## 📈 Performance Benchmarks

### Expected Performance (from paper)

| Metric | Classical PSO | Quantum PSO | Sequential PSO | AHP-PSO |
|--------|--------------|-------------|----------------|---------|
| Precision | 0.82 | 0.85 | 0.89 | **0.92** |
| Recall | 0.79 | 0.83 | 0.85 | **0.89** |
| F1-Score | 0.80 | 0.84 | 0.87 | **0.90** |
| mAP | 0.80 | 0.83 | 0.87 | **0.89** |
| IoU | 0.78 | 0.80 | 0.82 | **0.84** |
| FPS | 18 | 16 | 24 | **15** |

### Convergence Speed

- Classical PSO: ~150 iterations
- Quantum PSO: ~140 iterations
- Sequential PSO: ~120 iterations
- AHP-PSO: ~110 iterations **(18% faster)**

## 🐳 Docker Usage

```bash
# Build image
docker build -t ahp-pso:latest .

# Run container
docker run -it --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/results:/app/results \
  ahp-pso:latest python main.py --algorithm ahp_pso --domain aerial

# Run with GPU
docker run -it --rm --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/results:/app/results \
  ahp-pso:latest python main.py --algorithm ahp_pso --domain aerial
```

## 📝 Key Implementation Details

### 1. AHP-PSO Algorithm (src/algorithms/ahp_pso.py)

**Core Features:**
- Adaptive per-particle inertia: Lines 80-110
- AHP weight calculation: Lines 45-55
- Dynamic population adjustment: Lines 130-160
- Domain-specific fitness: Lines 180-220

### 2. Data Preprocessing (src/data_loader.py)

**Domain Augmentations:**
- Aerial: Perspective warp, altitude variation (Lines 95-115)
- Underwater: Color shift, turbidity simulation (Lines 120-145)
- Road: Texture overlay, weather effects (Lines 150-175)

### 3. Evaluation Metrics (src/evaluator.py)

**Comprehensive Metrics:**
- Precision, Recall, F1-Score
- mAP@50, mAP@75, mAP@50:95
- IoU, FPS, Inference Time
- Domain-specific scores

## 🔍 Code Quality

### Linting

```bash
# Check code style
flake8 src/ --max-line-length=127

# Format code
black src/

# Type checking
mypy src/ --ignore-missing-imports
```

### Performance Profiling

```bash
# Profile execution
python -m cProfile -o profile.stats main.py --algorithm ahp_pso

# View results
python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative'); p.print_stats(20)"
```

## 📚 Additional Resources

### Paper Implementation Mapping

| Paper Section | Implementation File | Key Functions |
|--------------|-------------------|---------------|
| Algorithm 4 (AHP-PSO) | `src/algorithms/ahp_pso.py` | `detect()`, `calculate_ahp_fitness()` |
| Section 3.2.1 (Classical PSO) | `src/algorithms/classical_pso.py` | `update_velocity()`, `detect()` |
| Section 3.2.2 (QPSO) | `src/algorithms/quantum_pso.py` | `quantum_position_update()` |
| Section 3.2.3 (SPSO) | `src/algorithms/sequential_pso.py` | `sequential_update()` |
| Section 4 (AHP Calculator) | `src/utils/ahp.py` | `calculate_weights()` |
| Section 5.2 (Metrics) | `src/evaluator.py` | `evaluate()` |

### Experiment Scripts

- **Benchmark:** `experiments/run_all_algorithms.py` - Reproduces Table 3
- **Convergence:** `experiments/convergence_analysis.py` - Reproduces Figure 3
- **Ablation:** `experiments/ablation_study.py` - Component analysis
- **Domain Comparison:** `experiments/domain_comparison.py` - Cross-domain evaluation

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Make changes and test: `pytest tests/`
4. Commit: `git commit -m "Add feature"`
5. Push: `git push origin feature-name`
6. Create Pull Request

## 📧 Support

- **Issues:** GitHub Issues
- **Email:** pankaj.mishra@somaiya.edu
- **Documentation:** https://yourusername.github.io/ahp-pso-detection

## 📄 Citation

```bibtex
@article{mishra2024ahppso,
  title={AHP-PSO: Adaptive Hybrid Particle Swarm Optimization for Real-Time Cross-Domain Object Detection},
  author={Mishra, Pankaj and Venkataramanan, V and Nayyar, Anand},
  year={2024}
}
```

## ⚖️ License

MIT License - see LICENSE file for details.

---

**Last Updated:** December 2024
**Version:** 1.0.0

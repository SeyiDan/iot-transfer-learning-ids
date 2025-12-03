# 🔒 Cross-Domain Transfer Learning for IoT Intrusion Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Status: Research](https://img.shields.io/badge/Status-Research-green.svg)]()

> **Master's Thesis Project (2025)**  
> *Leveraging Cross-Domain Transfer Learning for Enhanced Multi-Protocol Network Intrusion Detection*

**Author:** Oluwaseyi Oladejo  
**Supervisor:** Dr. Ahmed A. Ahmed  
**Institution:** Prairie View A&M University

---

## 📋 Table of Contents
- [Overview](#overview)
- [Key Findings](#key-findings)
- [Repository Structure](#repository-structure)
- [Installation & Setup](#installation--setup)
- [Usage Guide](#usage-guide)
- [Datasets](#datasets)
- [Results](#results)
- [Citation](#citation)
- [License](#license)

---

## 🎯 Overview

This research investigates **zero-shot transfer learning** for IoT intrusion detection across heterogeneous domains. We address the critical challenge of deploying intrusion detection systems (IDS) in resource-constrained IoT environments where labeled training data is scarce or unavailable.

### Research Question
**Can machine learning models trained on one IoT domain (e.g., medical IoT) effectively detect attacks in a different IoT domain (e.g., smart home IoT) without additional training?**

### Approach
- **Source Domain:** CICIoMT (Medical IoT)
- **Target Domains:** CIC-IoT (Smart Home IoT), IoT-23 (Botnet IoT)
- **Attack Types:** DoS, Reconnaissance (binary classification)
- **Algorithms Evaluated:** Random Forest, Gradient Boosting, XGBoost, SVM, MLP

### Novel Contributions
1. **Tree Methods Outperform Neural Networks:** Tree-based algorithms achieve 99% accuracy, 62 percentage points higher than neural networks
2. **Feature Overlap Metric:** First quantifiable predictor of transfer learning success (66% overlap → 99% accuracy)
3. **Domain Compatibility Framework:** Systematic methodology for assessing cross-domain transfer viability

---

## 🏆 Key Findings

| Algorithm | CIC-IoT Accuracy | IoT-23 Accuracy | Training Time |
|-----------|------------------|-----------------|---------------|
| **Random Forest** | **99.0%** ✅ | 50.0% ❌ | 1.7s |
| **Gradient Boosting** | **98.9%** ✅ | 50.0% ❌ | 22.0s |
| **XGBoost** | **98.4%** ✅ | 50.0% ❌ | **1.0s** ⚡ |
| **SVM** | 80.0% | 50.0% ❌ | 3.4s |
| **MLP (Neural Net)** | 37.0% ❌ | 50.0% ❌ | 13.8s |

### Critical Insights
- ✅ **Compatible domains** (CIC-IoT: 66% feature overlap) → 99% accuracy
- ❌ **Incompatible domains** (IoT-23: 0% feature overlap) → 50% accuracy (random guessing)
- 🌳 **Tree methods >> Deep learning** for IoT intrusion detection transfer learning
- ⚡ **XGBoost:** Best balance of speed (1s training) and accuracy (98.4%)
- 🔒 **Security Metrics:** Gradient Boosting achieves lowest false positive rate (3.6%)

---

## 📁 Repository Structure

```
├── app.ipynb                              # Interactive Gradio demo application
├── Data_Preprocessing_Pipeline.ipynb      # Dataset preprocessing & feature engineering
├── Transfer_Learning_Pipeline.ipynb       # Model training & evaluation
├── README.md                              # This file
├── requirements.txt                       # Python dependencies
└── results/                               # Generated results (not tracked)
    ├── confusion_matrices/
    ├── classification_reports/
    └── performance_metrics/
```

### Notebook Descriptions

#### 1️⃣ `Data_Preprocessing_Pipeline.ipynb`
**Purpose:** Prepare and align datasets for transfer learning experiments

**Key Steps:**
- Download datasets from Kaggle (CICIoMT, CIC-IoT, IoT-23)
- Label standardization (map to DoS, Reconnaissance, Benign, etc.)
- Feature engineering:
  - Cybersecurity features (packet rate, byte rate, flow asymmetry)
  - Statistical aggregations (mean, std, min, max)
  - PCA dimensionality reduction
- Hybrid class balancing (SMOTE + undersampling)
- Feature alignment across domains

**Outputs:**
- `enhanced_aligned_ciciomt_for_cic-iot.pkl`
- `enhanced_aligned_cic-iot.pkl`
- `enhanced_aligned_ciciomt_for_iot-23.pkl`
- `enhanced_aligned_iot-23.pkl`

---

#### 2️⃣ `Transfer_Learning_Pipeline.ipynb`
**Purpose:** Train models on source domain, evaluate on target domains

**Key Steps:**
- Load aligned datasets
- Train 5 algorithms on CICIoMT (source)
- Zero-shot transfer to CIC-IoT and IoT-23 (targets)
- Comprehensive evaluation:
  - Accuracy, Precision, Recall, F1-Score
  - False Positive Rate (FPR), False Negative Rate (FNR)
  - Training time, Inference time
- Generate confusion matrices and classification reports
- Comparative analysis and visualizations

**Outputs:**
- `transfer_learning_complete_results_enhanced.csv`
- Confusion matrix plots (`.png`)
- Classification reports (`.csv`)
- Performance comparison tables

---

#### 3️⃣ `app.ipynb`
**Purpose:** Interactive web application for exploring results

**Features:**
- **Layer 1:** Data preprocessing simulation
- **Layer 2:** Model training visualization
- **Layer 3:** Transfer learning demonstration
- **Layer 4:** Real-time performance metrics
- Professional cyan-themed UI (dark mode)
- Pre-loaded results for instant exploration

**Deployment:** Run on Google Colab or local Jupyter environment

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Google Colab account (recommended) OR local Jupyter environment
- Google Drive (for dataset storage)
- Kaggle API credentials

### Step 1: Clone Repository
```bash
git clone https://github.com/yourusername/iot-transfer-learning-ids.git
cd iot-transfer-learning-ids
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

**Required Libraries:**
```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
xgboost>=1.5.0
matplotlib>=3.4.0
seaborn>=0.11.0
gradio>=3.0.0
kagglehub>=0.1.0
```

### Step 3: Kaggle Setup
1. Create Kaggle account at [kaggle.com](https://www.kaggle.com)
2. Go to Account Settings → API → Create New API Token
3. Download `kaggle.json` (contains username and API key)
4. Store credentials securely

---

## 📖 Usage Guide

### Quick Start (Google Colab - Recommended)

#### Option A: Run Complete Pipeline

**1. Data Preprocessing (30-60 minutes)**
```python
# Open Data_Preprocessing_Pipeline.ipynb in Colab
# Run all cells sequentially
# Datasets will be downloaded and preprocessed automatically
```

**2. Transfer Learning Experiments (15-30 minutes)**
```python
# Open Transfer_Learning_Pipeline.ipynb in Colab
# Run all cells to train and evaluate models
# Results saved to Google Drive
```

**3. Interactive Demo (5 minutes)**
```python
# Open app.ipynb in Colab
# Run all cells to launch Gradio interface
# Explore results interactively
```

---

#### Option B: Use Pre-processed Data (Fast Track)

If you already have preprocessed datasets:

```python
# 1. Upload preprocessed .pkl files to Google Drive
# 2. Open Transfer_Learning_Pipeline.ipynb
# 3. Update paths to your preprocessed files
# 4. Run transfer learning experiments directly
```

---

### Local Jupyter Setup

**1. Start Jupyter Notebook**
```bash
jupyter notebook
```

**2. Run Notebooks**
- Open `Data_Preprocessing_Pipeline.ipynb`
- Modify paths to local directories (no Google Drive needed)
- Run cells sequentially
- Proceed to `Transfer_Learning_Pipeline.ipynb`

---

### Interactive Demo Deployment

**Launch Gradio App:**
```python
# In app.ipynb, run all cells
# Gradio will generate a public URL (valid for 72 hours)
# Share the URL for interactive exploration
```

**Example Output:**
```
Running on public URL: https://xxxxx.gradio.live
```

---

## 📊 Datasets

### Source Domain

**CICIoMT (2024)**
- **Description:** Medical IoT network traffic (hospital environment)
- **Devices:** 40 IoT devices (pacemakers, insulin pumps, sensors)
- **Attack Types:** DoS, DDoS, Reconnaissance, Spoofing
- **Size:** 1M+ samples, 73 features
- **Kaggle:** [limamateus/cic-iomt-2024-wifi-mqtt](https://www.kaggle.com/datasets/limamateus/cic-iomt-2024-wifi-mqtt)

### Target Domains

**CIC-IoT-2023**
- **Description:** Smart home IoT traffic
- **Devices:** Connected home appliances (cameras, thermostats, locks)
- **Attack Types:** DoS, Reconnaissance
- **Size:** 1,636 samples, 48 features (66% overlap with CICIoMT)
- **Kaggle:** [akashdogra/cic-iot-2023](https://www.kaggle.com/datasets/akashdogra/cic-iot-2023)

**IoT-23**
- **Description:** Botnet-infected IoT devices
- **Devices:** Malicious IoT botnet traffic (Mirai, Torii)
- **Attack Types:** DoS, Reconnaissance
- **Size:** 3,000 samples, 29 features (0% overlap with CICIoMT)
- **Kaggle:** [engraqeel/iot23preprocesseddata](https://www.kaggle.com/datasets/engraqeel/iot23preprocesseddata)

### Download Instructions

Datasets are automatically downloaded in `Data_Preprocessing_Pipeline.ipynb` using Kaggle API. Manual download:

```bash
kaggle datasets download -d limamateus/cic-iomt-2024-wifi-mqtt
kaggle datasets download -d akashdogra/cic-iot-2023
kaggle datasets download -d engraqeel/iot23preprocesseddata
```

---

## 📈 Results

### Performance Summary

| Metric | CIC-IoT (Compatible) | IoT-23 (Incompatible) |
|--------|---------------------|----------------------|
| **Best Accuracy** | 99.0% (Random Forest) | 50.0% (all models) |
| **Best FPR** | 3.6% (Gradient Boosting) | 50.0% |
| **Training Time** | 1.0-22.0s | 0.6-9.7s |
| **Inference Time** | 0.2-2.5ms/sample | 0.1-1.8ms/sample |

### Key Visualizations

Generated in `Transfer_Learning_Pipeline.ipynb`:

1. **Confusion Matrices:** Per-model, per-dataset
2. **Performance Comparison:** Accuracy, Precision, Recall, F1
3. **Training Time Analysis:** Speed vs. accuracy tradeoff
4. **Security Metrics:** FPR/FNR analysis
5. **Feature Alignment:** Before/after comparison

### Accessing Results

After running notebooks:
```
results/
├── transfer_learning_complete_results_enhanced.csv  # All metrics
├── cm_RandomForest_CIC-IoT.png                      # Confusion matrices
├── report_RandomForest_CIC-IoT.csv                  # Classification reports
└── accuracy_comparison.csv                           # Cross-model comparison
```

---

## 📝 Citation

If you use this code or methodology in your research, please cite:

```bibtex
@mastersthesis{oladejo2025transfer,
  title={Leveraging Cross-Domain Transfer Learning for Enhanced Multi-Protocol Network Intrusion Detection},
  author={Oladejo, Oluwaseyi},
  year={2025},
  school={Prairie View A\&M University},
  type={Master's Thesis},
  supervisor={Ahmed A. Ahmed},
  
}
```

**Alternative Citation (APA):**
```
Oladejo, O. (2025). Leveraging Cross-Domain Transfer Learning for Enhanced 
Multi-Protocol Network Intrusion Detection [Master's thesis, Prairie View A&M 
University]. Supervised by Dr. Ahmed A. Ahmed.
```

---

## 🔬 Research Contributions

### Academic Impact
- First systematic comparison of tree-based vs. neural network methods in zero-shot IoT transfer learning
- Novel feature overlap metric for predicting transfer learning success
- Domain compatibility assessment framework applicable to other transfer learning domains

### Practical Impact
- Reduces IDS deployment time from **months to hours**
- Achieves **99% accuracy** on compatible domains with **3.6% false positive rate**
- Enables rapid intrusion detection deployment in resource-constrained IoT environments

---

## 🛠️ Troubleshooting

### Common Issues

**1. Kaggle Authentication Error**
```python
# Solution: Re-enter credentials in Data_Preprocessing_Pipeline.ipynb
kaggle_username = input("Enter your Kaggle username: ")
kaggle_key = input("Enter your Kaggle API key: ")
```

**2. Google Drive Mount Failure**
```python
# Solution: Manually authorize Google Drive access
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
```

**3. Memory Error (Colab)**
```python
# Solution: Use Colab Pro or reduce sample_frac in preprocessing
# In Data_Preprocessing_Pipeline.ipynb:
preprocess_dataset(csv_path, dataset_name, sample_frac=0.2)  # Reduce from 0.3
```

**4. Missing Preprocessed Files**
```python
# Solution: Check Google Drive paths
import os
ALIGNED_DIR = '/content/drive/My Drive/Project_Final_Submission/enhanced_aligned_datasets'
print(os.listdir(ALIGNED_DIR))
```

---

## 🚧 Future Work

- [ ] Multi-source transfer learning (train on multiple domains simultaneously)
- [ ] Few-shot learning (minimal fine-tuning with 10-100 target samples)
- [ ] Adversarial robustness testing against evasion attacks
- [ ] Edge deployment optimization (Raspberry Pi, Arduino)
- [ ] Multi-class extension (10+ attack types beyond DoS/Reconnaissance)
- [ ] Real-time adaptive learning in production environments
- [ ] Explainable AI integration for security analysts

---

## 📧 Contact

**Oluwaseyi Oladejo**  
Master's Student, Computer Science  
Prairie View A&M University  
Email: oladejo.seyi2@gmail.com  
GitHub: [@yourusername](https://github.com/SeyiDan)

**Supervisor:** Dr. Ahmed A. Ahmed  
Email: amahmed@pvamu.edu

  


---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 Oluwaseyi Oladejo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 🙏 Acknowledgments

- **Datasets:** CICIoMT, CIC-IoT, IoT-23, Canadian Institute of Cybersecurity
- **Libraries:** Scikit-learn, XGBoost, Gradio communities
- **Support:** Prairie View A&M University Computer Science Department
- **Inspiration:** IoT security research community

---

## ⭐ Star This Repository

If you find this research helpful, please consider starring this repository to support open science in cybersecurity!

[![GitHub stars](https://img.shields.io/github/stars/yourusername/iot-transfer-learning-ids?style=social)](https://github.com/yourusername/iot-transfer-learning-ids)

---

**Last Updated:** December 2025  
**Status:** Active Research Project  
**Version:** 1.0.0

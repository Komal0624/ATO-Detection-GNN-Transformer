# Graph Neural Network and Transformer Based Account Takeover Detection

A hybrid deep learning framework that combines **Graph Attention Networks (GATv2)** and a **Transformer Encoder** to detect Account Takeover (ATO) attacks in login authentication data, under extreme class imbalance conditions.

> Internship Project — Department of Computer Engineering, National Institute of Technology, Kurukshetra (2026)
> Author: Komal (B.Tech CSE, Central University of Haryana)
> Guide: Dr. Kuldeep Kumar, Assistant Professor, Dept. of Computer Engineering, NIT Kurukshetra

---

## 📌 Problem Statement

Account Takeover (ATO) attacks occur when an attacker gains unauthorized access to a user account using stolen credentials. Detecting these attacks is difficult because of:

1. **Extreme class imbalance** — legitimate logins vastly outnumber takeover events (in this dataset, ~1 : 221,766).
2. **Relational dependencies** — login events are not independent; coordinated attacks reuse shared infrastructure (IPs, ASNs), which traditional tabular models fail to capture.

## 🧠 Approach

This project models login sessions as an **entity-relation graph**, where:
- **Nodes** = individual login sessions
- **Edges** = shared User ID, IP Address, or ASN between sessions (weighted by relationship strength)

The graph is then passed through a **hybrid GATv2 + Transformer architecture**:

```
Raw RBA Dataset (31.2M sessions)
        │
        ▼
Feature Selection (User ID, IP, ASN, RTT)
        │
        ▼
Preprocessing (missing values, log transform, hash encoding, scaling)
        │
        ▼
Class Imbalance Handling (Undersampling 20:1 + SMOTE 0.25)
        │
        ▼
Graph Construction (nodes = sessions, edges = shared identifiers)
        │
        ▼
Hybrid Model: GATv2 (4 heads) → GATv2 (2 heads) → Transformer Encoder
        │
        ▼
Residual Fusion → Linear + Softmax
        │
        ▼
Threshold Optimization (maximize validation F1)
        │
        ▼
Final ATO Prediction
```

## 📊 Dataset

- **Source:** [RBA Dataset (Risk-Based Authentication)](https://www.kaggle.com/datasets/dasgroup/rba-dataset) on Kaggle
- **Size:** 31,269,264 login sessions (31,269,123 legitimate, 141 account takeover)
- **Selected features:** User ID, IP Address, ASN, Round-Trip Time (RTT)
- **Target:** `Is Account Takeover` (binary)

## 🏗️ Model Architecture

| Component | Configuration |
|---|---|
| GATv2 Layer 1 | 4 attention heads, 64 features/head |
| GATv2 Layer 2 | 2 attention heads, 64 features/head |
| Dropout | 0.3 |
| Transformer Encoder | dim=128, heads=4, feedforward=256, layers=1 |
| Fusion | Residual: `X_final = X_gnn + X_transformer` |
| Classifier | Linear (128 → 2) + Softmax |
| Loss | Cross-entropy with label smoothing (0.05) |
| Optimizer | Adam (lr = 0.0008) |
| Early stopping | Patience = 20 epochs (on validation F1) |

## 📈 Results

### Test Set Performance

| Metric | Score |
|---|---|
| Precision | 0.8579 |
| Recall | 0.9261 |
| F1-score | 0.8907 |
| MCC | 0.8632 |
| PR-AUC | 0.8240 |

**Confusion Matrix (Test Set):**

| | Predicted Legitimate | Predicted Takeover |
|---|---|---|
| **Actual Legitimate** | 679 | 27 |
| **Actual Takeover** | 13 | 163 |

The model achieves high recall (0.9261), meaning most takeover attempts are correctly flagged — critical in fraud detection where missed attacks (false negatives) are far costlier than false alarms.

## 🛠️ Tech Stack

- Python 3.x
- PyTorch & PyTorch Geometric
- Scikit-learn, Imbalanced-learn (SMOTE)
- Polars, NumPy
- Matplotlib, Seaborn
- Google Colab (NVIDIA Tesla T4 GPU)

## 📂 Repository Structure

```
.
├── notebook/
│   └── ato_detection.ipynb        # Full training & evaluation pipeline
├── report/
│   └── ATO_Detection_Report.pdf   # Detailed internship project report
├── requirements.txt
├── .gitignore
└── README.md
```

## 🚀 Getting Started

```bash
git clone https://github.com/<your-username>/ato-detection-gnn-transformer.git
cd ato-detection-gnn-transformer
pip install -r requirements.txt
```

Then open `notebook/ato_detection.ipynb` in Jupyter or Google Colab. The notebook downloads the RBA dataset automatically via `kagglehub` (requires a Kaggle API key).

## ⚠️ Limitations & Future Work

- Graph is **static** — doesn't model temporal evolution of login behavior
- Only identity/network-level features used (no device fingerprinting or behavioral signals)
- SMOTE-generated synthetic samples may not perfectly reflect real attack patterns
- Single-layer transformer encoder — deeper architectures could be explored
- Evaluated offline (batch), not under real-time streaming conditions

Planned improvements: temporal/dynamic GNNs, GAN-based synthetic sample generation, deeper transformer stacks, and real-time deployment feasibility testing.

## 📄 Full Report

The complete internship report with literature review, methodology, and detailed analysis is available in [`report/ATO_Detection_Report.pdf`](./report/ATO_Detection_Report.pdf).

## 📜 License

This project is released for academic and educational purposes.

# FARW
requirements_content = """# Core Scientific Libraries
numpy==1.24.3
pandas==2.0.2
scipy==1.10.1

# Graph Processing and Modeling
networkx==3.1
gensim==4.3.1

# Machine Learning & Evaluation
scikit-learn==1.2.2

# Visualization (Optional but recommended)
matplotlib==3.7.1
seaborn==0.12.2
"""

readme_content = """# FARW: A Feature-Aware Random Walk for Node Classification

This repository is based on the paper:
**"FARW: A Feature-Aware Random Walk for node classification"**

## 📝 Overview
FARW is a graph embedding approach that combines **node features** with **graph topology** to generate more informative node representations for downstream tasks such as **node classification** and **link prediction**.

Unlike traditional random walk methods, which are mostly structure-based and may suffer from bias toward high-degree nodes, slow convergence, and difficulty handling disconnected components, FARW uses **feature-aware traversal** guided by **cosine similarity** between node attributes.

## 🛠️ Prerequisites & Installation
This project requires **Python 3.9** or **Python 3.10**. 

### 1. Create a Virtual Environment (Recommended)
```bash
python -m venv farw_env
source farw_env/bin/activate  # On Windows: farw_env\\Scripts\\activate


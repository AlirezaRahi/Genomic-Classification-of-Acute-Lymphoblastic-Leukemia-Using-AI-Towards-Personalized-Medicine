# Genomic Classification of Acute Lymphoblastic Leukemia Using AI Towards Personalized Medicine

This repository contains the code for a sophisticated deep learning ensemble model designed to classify subtypes of Acute Lymphoblastic Leukemia (ALL) from gene expression microarray data. The model combines Convolutional Neural Networks (CNN), Long Short-Term Memory networks (LSTM), and Dense networks, with a Gradient Boosting meta-learner for final prediction.

## Publication
**Publication**: Genomic Classification of Acute Lymphoblastic Leukemia Using AI: Towards Personalized Medicine


**DOI**: https://doi.org/10.1101/2025.09.20.25336225


**Publication Date**: September 21, 2025.

##  Model Performance Summary

The final ensemble model achieved the following performance on the held-out test set:

- **Overall Accuracy:** 78.57%
- **Mean ROC-AUC Score:** 0.9685

### Table 1. Detailed Classification Metrics per Subtype

| Subtype | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Acute Lymphoblastic Leukemia TEL-AML1** | 0.80 | 1.00 | 0.89 | 4 |
| **Acute Lymphoblastic Leukemia Hyperdiploid** | 0.80 | 0.67 | 0.73 | 6 |
| **T-cell Acute Lymphoblastic Leukemia** | 0.60 | 1.00 | 0.75 | 3 |
| **Acute Lymphoblastic Leukemia E2A-PBX1** | 1.00 | 1.00 | 1.00 | 2 |
| **Acute Lymphoblastic Leukemia Other** | 0.80 | 0.80 | 0.80 | 5 |
| **Acute Lymphoblastic Leukemia Ph-positive** | 0.80 | 1.00 | 0.89 | 4 |
| **Acute Lymphoblastic Leukemia MLL-rearranged** | 1.00 | 0.25 | 0.40 | 4 |

##  Model Architecture

The project implements a powerful stacking ensemble method:
1. **Base Learners:** Three distinct deep learning architectures are trained independently on the genomic data.
   - **1D-CNN:** For capturing local spatial patterns in gene expression.
   - **LSTM:** For modeling potential sequential dependencies.
   - **Dense Network:** A standard fully-connected network for robust feature learning.
2. **Meta-Learner:** The predictions (class probabilities) from all three base models are concatenated to form a meta-feature set. A **Gradient Boosting Classifier** is then trained on these meta-features to make the final prediction.

##  Installation & Usage

### Prerequisites
Ensure you have Python 3.8+ installed. The required libraries are listed in `requirements.txt`.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YourUsername/ALL-Subtype-Classification.git
   cd ALL-Subtype-Classification
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the training script:**
   The main script will train all models sequentially and output results and visualizations.
   ```bash
   python main_training_script.py
   ```

### Key Steps in the Code:
- Data loading and cleaning (handling NaN values).
- StandardScaler normalization and feature selection (SelectKBest).
- Class weighting to handle imbalanced data.
- Data augmentation with mild Gaussian noise and scaling.
- Model training with Early Stopping, ReduceLROnPlateau, and ModelCheckpoint callbacks.
- Ensemble prediction and comprehensive evaluation.

##  Outputs

After a successful run, the script generates:
- **Saved Models:** `cnn_model.keras`, `lstm_model.keras`, `dense_model.keras`, `meta_learner.pkl`
- **Preprocessing Artifacts:** `scaler.pkl`, `feature_selector.pkl`, `subtype_mapping.json`
- **Evaluation Plots:**
  - `confusion_matrix_all_subtypes.png`
  - `roc_curve_all_subtypes.png`
  - `training_history_*.png` (for each model)

##  Dataset

This work utilizes a novel, balanced gene expression dataset for ALL subtypes, created and published by the author. It is a comprehensive integration of several publicly available landmark studies.

**Primary Dataset:**
> **A. Rahi**, “A Global Reference Balanced Gene Expression Dataset for Acute Lymphoblastic Leukemia Subtypes,” Zenodo, 2025. doi: 10.5281/zenodo.17008431. [Online]. Available: https://zenodo.org/records/17008431

**Integrated Source Datasets (GEO Accession):**
1.  Chiaretti, S. et al. (2004). *Blood*, 103(7), 2771–2778. [GSE13159](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE13159)
2.  Lilljebjörn, H. et al. (2016). *Nature Genetics*, 48(4), 482–488. [GSE79533](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE79533)
3.  Andersson, A. et al. (2018). *Nature Communications*, 9(1), 401. [GSE60926](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE60926)
4.  Fischer, U. et al. (2010). *Leukemia*, 24, 750–757. [GSE28497](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE28497)
5.  Haferlach, T. et al. (2007). *Blood*, 109(4), 1551–1561. [GSE3910](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE3910)

*Note: The model is presented here for methodological and reproducibility purposes. Performance is intrinsically linked to the dataset it was trained on.*

## Citation

If you use this work in your research, please cite the paper:

> **Rahi, A., & Shafiabadi, M. H.** (2025). *Genomic Classification of Acute Lymphoblastic Leukemia Using AI: Towards Personalized Medicine*. medRxiv. https://doi.org/10.1101/2025.09.20.25336225 
If you use the code implementation (software, scripts, etc.), please also cite:

> **Rahi, A.** (2025). *Genomic Classification of Acute Lymphoblastic Leukemia Using AI: Towards Personalized Medicine* [Computer software]. GitHub repository, *AlirezaRahi/Genomic-Classification-of-Acute-Lymphoblastic-Leukemia-Using-AI-Towards-Personalized-Medicine*. Retrieved from https://github.com/AlirezaRahi/Genomic-Classification-of-Acute-Lymphoblastic-Leukemia-Using-AI-Towards-Personalized-Medicine
 

##  Author

- **Alireza Rahi**
    - Email: [alireza.rahi@outlook.com](alireza.rahi@outlook.com)
    - LinkedIn: [https://www.linkedin.com/in/alireza-rahi-6938b4154/](https://www.linkedin.com/in/alireza-rahi-6938b4154/)
    - GitHub: [https://github.com/AlirezaRahi](https://github.com/AlirezaRahi)

## 📜 License

This work is licensed under a Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License (CC BY-NC-ND 4.0).

This means you are free to:

Share — copy and redistribute the material in any medium or format for non-commercial purposes.

Under the following terms:

Attribution — You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.

NonCommercial — You may not use the material for commercial purposes.

NoDerivatives — If you remix, transform, or build upon the material, you may not distribute the modified material.

Summary: This work may be read and downloaded for personal use only. It may be shared in its complete and unaltered form for non-commercial purposes, provided that the author’s name, the title of the work, and a link to the original source (this repository) and the license are clearly cited. Any modification, adaptation, commercial use, or distribution for profit is strictly prohibited.

For permissions beyond the scope of this license, please contact the author directly.

https://licensebuttons.net/l/by-nc-nd/4.0/88x31.png


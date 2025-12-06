# Credit Card Fraud Detection and Profit-Driven Three-Action Policy Optimization

This repository contains the code for my STAT 507 final project:

> **“Credit Card Fraud Detection and Profit-Driven Three-Action Policy Optimization”**   

The project builds an end-to-end fraud detection pipeline on the public **Kaggle credit-card fraud dataset** and connects machine-learning outputs to **business decisions** via a three-action policy: **Approve**, **Step-up challenge**, or **Decline**.   

---

## 1. Project Overview

Key goals:

- Train and compare two models under **extreme class imbalance**:
  - Class-weighted **Logistic Regression**
  - **LightGBM** gradient-boosted trees   
- Use **Platt scaling** to obtain **well-calibrated fraud probabilities**.   
- Design and optimize a **three-action policy** (Approve / Step-up / Decline) that **maximizes expected profit** under a simple cost model.   
- Emphasize **Precision–Recall AUC (PR-AUC)** rather than ROC-AUC for model selection, because the dataset is highly skewed.   

On the held-out test set, the **calibrated logistic regression** model achieves:

- **ROC-AUC:** 0.9820  
- **PR-AUC:** 0.7436   

A grid search over two thresholds finds a policy that:

- Challenges only **0.27%** of transactions,
- Captures **82.7%** of fraudulent ones,
- Improves expected profit relative to an approve-all baseline.   

---

## 2. Data

- **Dataset:** [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)   
- **Samples:** 284,807 transactions with **492 frauds** (~0.17% positive rate).   
- **Features:** 30 numerical variables:
  - 28 PCA-transformed features `V1` … `V28`,
  - `Amount`,
  - `Time` (seconds since first transaction).   
- **Label:** `Class` = 1 for fraud, 0 for legitimate transactions.   

### Time-based split

To avoid leaking future information into the past, I use a **time-based split** on the `Time` column:

- 60% (170,884 rows): **train**
- 20% (56,961 rows): **validation**
- 20% (56,962 rows): **test**   

This mimics the realistic setting where a model trained on historical data is applied to future transactions.

---

## 3. Methodology

### 3.1 Preprocessing and Feature Scaling

- All 30 numerical features are used.
- Missing values are checked; none are found, so no imputation is needed.   
- Features are standardized to **zero mean and unit variance** using `StandardScaler` fitted on the **training set** and then applied to validation and test sets (to avoid information leakage).   

### 3.2 Models and Class Imbalance Handling

1. **Logistic Regression**
   - L2 regularization, `lbfgs` solver.
   - `class_weight="balanced"` to up-weight the rare fraud class.   

2. **LightGBM**
   - Gradient-boosted decision trees with:
     - 300 estimators,
     - learning rate 0.05,
     - subsample 0.8 for both rows and columns,
     - unrestricted depth (growth controlled by leaves).   
   - `scale_pos_weight = n_neg / n_pos` to address imbalance.   

Model selection is based primarily on **validation PR-AUC** (PR-AUC is more informative than ROC-AUC under heavy class imbalance). ROC-AUC is reported as a secondary measure.   

### 3.3 Probability Calibration (Platt Scaling)

- Raw model scores are often **poorly calibrated**: e.g., a predicted fraud probability of 0.8 may not correspond to an 80% empirical fraud rate.   
- I apply **Platt scaling**, which fits a one-dimensional logistic regression that maps the original score `s` to a calibrated probability  

  \[
  \hat{p}_{\text{cal}} = \sigma(a + b s).
  \]

- Procedure:
  1. Choose the better base model by **validation PR-AUC** (logistic regression in this project).
  2. Compute its predicted probabilities on the **validation set**.
  3. Train a logistic regression calibrator on those probabilities (single-feature input).
  4. Apply the calibrator to test-set probabilities to get calibrated scores.   

### 3.4 Three-Action Policy and Cost Model

The decision system uses **two thresholds** \( \tau_1 \le \tau_2 \) on the calibrated probability \( \hat{p}_{\text{cal}} \):   

- If \( \hat{p}_{\text{cal}} < \tau_1 \): **Approve**
- If \( \tau_1 \le \hat{p}_{\text{cal}} < \tau_2 \): **Step-up challenge**
- If \( \hat{p}_{\text{cal}} \ge \tau_2 \): **Decline**

Cost model per transaction (with \( y=0 \) legitimate, \( y=1 \) fraud):

- \( R = 1.0 \): revenue from approving a legitimate payment  
- \( C_{\text{fraud}} = 10.0 \): loss from approving a fraud  
- \( C_{\text{challenge}} = 0.2 \): cost of a step-up challenge  
- \( C_{\text{block}} = 1.0 \): cost of wrongly declining a good customer   

A grid search over \( \tau_1, \tau_2 \in [0,1] \) (25 candidates each) finds the pair that maximizes **average profit per transaction**.   

---

## 4. Results (Summary)

### 4.1 Model Comparison

On the **validation set**:

- Logistic Regression: ROC-AUC 0.9709, PR-AUC 0.7719  
- LightGBM: ROC-AUC 0.6996, PR-AUC 0.0356   

Because LightGBM performs poorly on the minority class, logistic regression is chosen as the base model.

On the **test set** (after Platt scaling):

- ROC-AUC: **0.9820**  
- PR-AUC: **0.7436**   

The ROC curve lies near the top-left corner, and the PR curve maintains high precision over a broad range of recalls, indicating strong ranking performance on the fraud class.   

### 4.2 Thresholding and Profit-Driven Policy

If we use the conventional threshold of 0.5, the classifier predicts **all transactions as non-fraud**, achieving accuracy 0.9987 but **recall 0 on frauds**.   

Using the profit-based grid search, the best policy is at:

- \( \tau_1 = 0.0833 \)
- \( \tau_2 = 0.1250 \)   

Under this policy:

- **Approve rate:** 99.73%  
- **Challenge rate:** 0.27%  
- **Decline rate:** ~0%  
- **Fraud recall:** 82.7% (62 / 75 frauds flagged)  
- **False alarms:** 89 legitimate transactions challenged  
- **Precision on fraud class:** 41.1%  
- **Overall accuracy:** 0.9982   

Despite the approve-all baseline already having high average profit (because fraud is rare), the optimized three-action policy **increases expected profit** and substantially improves fraud detection while keeping customer friction very low.   

---

## 5. Repository Structure

A minimal structure for this repository is:

```text
.
├── README.md               # This file
├── final project.ipynb     # Main script implementing the pipeline 
└── creditcard.csv          # Kaggle dataset

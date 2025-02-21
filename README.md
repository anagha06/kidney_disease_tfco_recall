# CKD_TFCO_Recall  

**Improving recall of sub-groups in Kidney Disease Prediction using TensorFlow Constrained Optimization (TFCO).**  

## Overview  

This project applies **TensorFlow Constrained Optimization (TFCO)** to enhance recall in sub-groups for **Chronic Kidney Disease (CKD) prediction**. Traditional supervised machine learning pipelines minimize a single loss function but often fail to consider real-world biases. This work incorporates **NeutralNet**, a fairness-aware optimization mechanism that mitigates feature bias by enforcing recall constraints through **Lagrangian proxy constraints**.  

The primary notebook, **ckd_study.ipynb**, contains the full study.  

## Features  

- **Bias Mitigation**: Uses TFCO to reduce harmful feature bias in CKD predictions.  
- **Fairness-Constrained Learning**: Applies Lagrangian proxy constraints to balance recall across subgroups.  
- **Optimized for Healthcare AI**: Ensures ethical AI applications by addressing disparities in model predictions.  
- **Modular Integration**: NeutralNet can be integrated into existing ML/DL models to enforce fairness constraints.  

## Methodology  

### **1. Problem Statement**  
- Traditional models **prioritize loss minimization** without considering fairness.  
- Feature biases can cause **skewed medical predictions**, leading to misdiagnoses.  

### **2. Model Design**  
- **Baseline Model**:  
  - Built with **TensorFlow & Keras**  
  - Trained with **ADAM optimizer** on a **UCI renal cancer dataset**  
- **NeutralNet Integration**:  
  - Implements **TFCO constraints** on group-specific recall rates  
  - Uses **Lagrangian proxy constraints** to enforce fairness during training  
  - Supports **ADAM & Adagrad optimizers** for improved convergence  

### **3. Results & Discussion**  
- **Baseline Model Findings**:  
  - Showed **recall bias** against younger CKD-positive patients.  
  - Model performance was affected by skewed feature associations.  
- **NeutralNet Impact**:  
  - Transformed recall decay from **parabolic to linear, high-recall** behavior.  
  - Effectively **balanced recall rates** across sub-groups.  
  - Prevented **hidden bias errors** that could lead to misdiagnosis.  

## Installation  

### **Requirements**  
- Anaconda  
- Python 3.x  
- TensorFlow  
- TensorFlow Constrained Optimization (TFCO)  

### **Setup**  
```bash
git clone https://github.com/yourusername/ckd_tfco_recall.git
cd ckd_tfco_recall
pip install -r requirements.txt

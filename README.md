# Health Risk Prediction Project

This project aims to predict critical health conditions using a dataset with 1,000 features. The goal is to classify whether an individual is at risk (positive class) or not (negative class) using a carefully selected subset of features.

## Overview
- High-dimensional dataset with limited feature descriptions.  
- Imbalanced target distribution (86% negative).  
- Focused on real-world challenges: feature selection, data imbalance, and model interpretability.

## Approach
1. **Preprocessing:**  
   - Removed columns with >30% missing values; imputed the rest with the median.  
   - Applied SMOTE to balance the target class.  
   - Selected features using the Kolmogorov-Smirnov test.

2. **Modeling:**  
   - Tested Logistic Regression, Decision Tree, and XGBoost.  
   - Evaluated with ROC-AUC, confusion matrix, decile/percentile analysis, and SHAP values.

3. **Results:**  
   - Best model (Logistic Regression) achieved AUC = 0.54.  
   - Limited predictive power due to feature quality and separability.  
   - Decile/percentile analysis showed better stratification for high-risk individuals.

## Tools & Libraries
Python, Pandas, NumPy, Scikit-learn, XGBoost, Imbalanced-learn, Matplotlib, Seaborn, Plotly, SHAP

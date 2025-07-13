# Breast Cancer Classifier with Explainable AI (SHAP)

This project builds a binary classifier for breast cancer diagnosis using the `scikit-learn` breast cancer dataset. It also demonstrates how to use **SHAP** (SHapley Additive exPlanations) to interpret model predictions.

## Features
- Logistic Regression, tree-Based (Random Forest, XGBoost, LightGBM) Models, and Neural network (Multi-Layer Perceptron; MLP) model
- SHAP explainability with summary and force plots
- Clean modular Python code + Jupyter notebook

## Project Structure
- `src/`: Model training and SHAP explainability
- `notebooks/`: Training pipeline and visualization
- `assets/`: example SHAP plots 

## Run the notebook
```bash
pip install -r requirements.txt
cd notebooks
jupyter notebook breast_cancer-classifier.ipynb

# COS801-Homework-Mpho-Muloiwa-
COS801 Homework
This project implements a hybrid machine learning pipeline for plant seedling classification. It combines a Convolutional Neural Network (CNN) trained in TensorFlow/Keras with downstream classical classifiers (SVM, XGBoost / GradientBoosting) using CNN feature embeddings. The pipeline evaluates models on a validation split and generates predictions for an unlabeled test set.
Automatic package installation (NumPy, TensorFlow, scikit-learn, matplotlib, pandas, etc.) 
CNN with data augmentation and training on plant seedling datasets.
Feature extraction from CNN’s dense embedding layer.
Classical ML classifiers on CNN embeddings: SVM (RBF kernel), XGBoost (or scikit-learn’s GradientBoosting as fallback)
Evaluation with: Micro-averaged F1-score, ROC-AUC curves (multi-class, micro-averaged), Prediction export for unlabeled test images.
Load and split the dataset (80% train, 20% validation).
Train the baseline CNN with augmentation.
Extract features from the CNN’s embedding layer.
Train and evaluate SVM and XGBoost (or GradientBoosting).
Plot ROC curves and display performance metrics.
Generate predictions for the test/ set.
During training and evaluation, you will see:
Validation results: CNN baseline micro F1, CNN + SVM micro F1, CNN + XGBoost (or GradientBoosting) micro F1
ROC curves comparing classifiers
A results table with F1-score and ROC-AUC values
A prediction DataFrame for test images, showing: File name, CNN / SVM / XGBoost predicted class, Confidence score per model
XGBoost requires installation; if unavailable, the pipeline falls back to GradientBoostingClassifier.
CNN uses ReLU activations in convolutional/dense layers (except final softmax for classification).
Model performance may vary depending on dataset balance and size

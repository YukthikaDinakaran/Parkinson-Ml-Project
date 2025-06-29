# Parkinson's Disease Early Detection using Machine Learning (SVM)

This project focuses on detecting Parkinson's Disease at an early stage using machine learning techniques applied to speech signal data. The model is built using Support Vector Machine (SVM) with a linear kernel, trained on vocal measurements from individuals diagnosed with and without Parkinson’s Disease.


## Dataset Description

The dataset used is a publicly available Parkinson’s dataset that includes a variety of vocal features extracted from voice recordings of healthy and affected individuals.

### Attribute Information

- `name`: ASCII subject name and recording number
- `MDVP:Fo(Hz)`: Average vocal fundamental frequency
- `MDVP:Fhi(Hz)`: Maximum vocal fundamental frequency
- `MDVP:Flo(Hz)`: Minimum vocal fundamental frequency
- `MDVP:Jitter(%)`, `MDVP:Jitter(Abs)`, `MDVP:RAP`, `MDVP:PPQ`, `Jitter:DDP`: Variation in fundamental frequency
- `MDVP:Shimmer`, `Shimmer(dB)`, `Shimmer:APQ3`, `Shimmer:APQ5`, `MDVP:APQ`, `Shimmer:DDA`: Variation in amplitude
- `NHR`, `HNR`: Noise-to-harmonics ratio features
- `RPDE`, `D2`: Nonlinear dynamical complexity measures
- `DFA`: Fractal scaling exponent
- `spread1`, `spread2`, `PPE`: Nonlinear measures of frequency variation
- `status`: Target variable (1 = Parkinson’s, 0 = Healthy)


## Project Features

- SVM model with a linear kernel for classification
- Synthetic data added to enlarge the dataset
- Missing values handled using KNNImputer
- StandardScaler applied for feature scaling
- Evaluation using multiple classification metrics
- Visualization of results including confusion matrix and performance bar chart


## Workflow

1. Load the dataset
2. Clean and preprocess the data (KNN imputation, rounding, scaling)
3. Train the SVM classifier
4. Evaluate using accuracy, precision, recall, F1-score, and confusion matrix
5. Visualize model performance using plots


## Evaluation Metrics

| Metric      | Description                          |
|-------------|--------------------------------------|
| Accuracy    | Proportion of total correct predictions (98.7) |
| Precision   | Correct positive predictions / All predicted positives (98.7) |
| Recall      | Correct positive predictions / All actual positives (1.0)|
| F1 Score    | Harmonic mean of precision and recall (99.3) |


## Visualization

- Confusion matrix heatmap
- Bar chart comparing accuracy, precision, recall, and F1 score


## Libraries and Tools Used

- pandas, numpy
- scikit-learn (SVM, metrics, preprocessing)
- matplotlib, seaborn
- KNNImputer from sklearn for missing value imputation


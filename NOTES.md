# DDoS Detection System Analysis & Design Notes

Given the dataset, we have 80 numerical features and 1 target variable and the other 4 features were dropped due because they were just identifier columns and were not relevant for modelling.


Target variable: Label
- 1: DDoS (56.713608%)
- 0: BENIGN (43.286392%)

Due to the target feature containing only 2 classes, we will use StratifiedKFold for cross-validation with no data being imputed Typically, we consider a dataset to be highly imbalanced when the minority class has less than 10-20% of the total samples.

Why didn't I use SMOTE or any sampling techniques?
The imbalance is mild, and using sampling techniques could introduce unnecessary noise.
Alternative Handling(imlemented in pipeline):
1. Class Weighting: Most classifiers (e.g., class_weight='balanced' in sklearn models) can handle this level of imbalance well.
2. Stratified Sampling: Ensuring train-test splits maintain the distribution.


## Data Exploration and Analysis (EDA)
Data columns have been cleaned and preprocessed to account for missing values, infinite values, duplicate rows and also the random spaces in the column names.
1. Missing values were dropped(0.015% of the data - negligible)
2. Duplicate rows were also dropped(0.02% of the data - negligible)
3. Features having inf were replaced with nan.
4. Label encoding was applied to the target variable(BENIGN -> 0, DDoS -> 1)


## 1. Data Split Strategy

- DDoS detection tasks can be sensitive to distribution drift, so it’s often beneficial to ensure that random sampling does not inadvertently end up with mostly similar flows in train or test sets.
- This ensures the model generalizes rather than memorizes a specific subset of flows.

- Training Set: 70% of data
  * Large enough for feature importance analysis

- Test Set: 30% of data
  * Held out for final evaluation(completely unseen data reserved for final evaluation)
  * Will provide unbiased performance metrics

- Validation Strategy:
  * K-fold(k=5) cross-validation on training set
  * Helps prevent overfitting
  * Ensures model generalization and optimal HP choices found using GridSearchCV

### 2. Dataset Characteristics
- Random seed: 42 (for reproducibility)


# Implementation Details
Since we have multiple model to choose from we can consider various factors to choose the best model. The goal is to find the best model that can generalize well on the test set and that too with a good F1 score and minimal data processing.

Since we have a lot of features to choose from, we can use feature importance to choose the top features and then use those features to train the model. But here we will use all the features and see how the model performs.

To ensure that we have a good model and to minimize the data processing, we will use the following models:
- XGBoost
- Random Forest
- Logistic Regression
- MLP Neural Network

To achieve minimal data processing we can choose Random Forest and XGBoost as tree based models don't require scaling of features and are also prone to overfitting and outliers. An added advantage is that they are also relatively fast to train and easily interpretable.\\

Logistic Regression and MLP are also reasonably good but they don't scale well and are also prone to overfitting easily.

1. Pre-limnary modelling:
We have implemented a pipeline for data preprocessing, model training and evaluation. We will use GridSearchCV to find the best hyperparameters for all the models and choose 1 to run the pipeline(here Random Forest is chosen).

2. Metrics:
Given the updated class distribution (56.71% DDoS, 43.29% BENIGN), F1-score is a better metric than accuracy for this case:

However, in security contexts, false negatives (missing attacks) are more costly than false positives
- F1-score is the harmonic mean of precision and recall and it helps balance:
    - Precision: Avoiding false DDoS alerts
    - Recall: Not missing actual DDoS attacks

- Particularly important because:
    - False Positives = Unnecessary system shutdowns/alerts
    - False Negatives = Missed attacks (very costly)

## Business Impact
- F1-score better reflects the business cost of errors
- Helps optimize the model for both:
    - Minimizing false alarms
    - Maximizing attack detection
- More suitable for security applications
- Model Tuning
    - F1-score helps tune the classification threshold
    - Balances the trade-off between precision and recall
    - More informative for model selection and optimization

Therefore, while accuracy is good for general performance understanding, F1-score provides a more nuanced and appropriate metric for this security application and finding anomalies.


# Results and Evaluation

## Feature Importance (Top Predictors)
1. Flow IAT Mean
2. Flow Duration
3. Packet Length Std
4. Flow Bytes/s
5. Flow Packets/s


## Data Relationships
1. Network Flow Characteristics
2. Strong correlation between flow metrics
3. Clear separation between benign and DDoS patterns
4. Temporal patterns in attack sequences (observed from scatter plot)


## Traffic Patterns
1. Flow duration strongly indicates attack type
2. Packet length distributions differ by class
3. IAT (Inter-Arrival Time) crucial for detection

## Strong Positive Correlations
1. Flow IAT Mean → DDoS attacks
2. Packet count → Attack likelihood
3. Flow duration → Attack patterns

## Strong Negative Correlations
1. Normal port usage → Benign traffic
2. Regular packet sizes → Benign traffic
3. Consistent flow rates → Benign traffic


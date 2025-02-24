# DDoS Detection System Analysis & Design Notes

Target variable: Label
- 1: DDoS (56.713608%)
- 0: BENIGN (43.286392%)

Due to the target feature containing only 2 classes, we will use StratifiedKFold for cross-validation with no data being imputed Typically, we consider a dataset to be highly imbalanced when the minority class has less than 10-20% of the total samples.

Why didn't I use SMOTE or any sampling techniques?
The imbalance is mild, and using sampling techniques could introduce unnecessary noise.
Alternative Handling(imlemented in pipeline):
1. Class Weighting: Most classifiers (e.g., class_weight='balanced' in sklearn models) can handle this level of imbalance well.
2. Stratified Sampling: Ensuring train-test splits maintain the distribution.


## Exploratory Data Analysis (EDA)
Data columns have been cleaned and preprocessed to account for missing values, infinite values, duplicate rows and also the random spaces in the column names.
1. Missing 

We have 80 numerical features and 1 target variable and the other 4 features were dropped due because they were just identifier columns and were not relevant for modelling.

Dropped non-predictive columns: Flow ID, IPs, Timestamp








## Data Analysis Insights

### 1. Data Split Strategy
- Training Set: 90% of data
  * Large enough for feature importance analysis

- Test Set: 10% of data
  * Held out for final evaluation
  * Never used during training
  * Will provide unbiased performance metrics

- Validation Strategy:
  * K-fold(k=5) cross-validation on training set
  * Helps prevent overfitting
  * Ensures model generalization and optimal HP choices found using GridSearchCV

### 2. Dataset Characteristics
- Total Records: [Your total number] network flows
- Random seed: 42 (for reproducibility)

Label encoding: BENIGN -> 0, DDoS -> 1

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


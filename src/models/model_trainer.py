from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import pandas as pd
from src.utils.logger import setup_logger
from tqdm import tqdm
import time

logger = setup_logger()

class ModelTrainer:
    def __init__(self):
        # Initialize with notebook's exact configuration
        self.model = RandomForestClassifier(
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'  # Important for imbalanced dataset
        )
        
        # Exact parameter grid from notebook
        self.param_grid = {
            "n_estimators": [50, 100],
            "max_depth": [None, 10, 20],
            "class_weight": ["balanced"],
            "min_samples_split": [2, 5]
        }

    def train(self, X_train, y_train):
        """Train the Random Forest model using GridSearchCV with progress bar."""
        try:
            logger.info("Starting Random Forest training...")
            logger.info(f"Training data shape: {X_train.shape}")
            logger.info(f"Class distribution:\n{pd.Series(y_train).value_counts(normalize=True)}")
            
            # Use StratifiedKFold as in notebook
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            # Calculate total iterations
            n_iterations = (len(self.param_grid['n_estimators']) * 
                          len(self.param_grid['max_depth']) * 
                          len(self.param_grid['min_samples_split']) * 
                          len(self.param_grid['class_weight']) * 5)  # 5-fold CV
            
            # Create progress bar
            pbar = tqdm(total=n_iterations, 
                       desc="Training Progress", 
                       unit="fit")
            
            start_time = time.time()
            
            grid_search = GridSearchCV(
                estimator=self.model,
                param_grid=self.param_grid,
                cv=cv,
                scoring='f1',
                n_jobs=-1,
                verbose=1
            )
            
            # Fit the model
            grid_search.fit(X_train, y_train)
            
            # Close progress bar
            pbar.close()
            
            # Calculate training time
            training_time = time.time() - start_time
            
            logger.info(f"\nTraining completed in {training_time:.2f} seconds")
            logger.info(f"Best parameters found: {grid_search.best_params_}")
            logger.info(f"Best cross-validation F1 score: {grid_search.best_score_:.4f}")
            
            # Store best model
            self.model = grid_search.best_estimator_
            
            # Get feature importance
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            logger.info("\nTop 10 most important features:")
            logger.info(feature_importance.head(10))
            
            return self.model
            
        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
            raise
        finally:
            if 'pbar' in locals():
                pbar.close()

    def predict(self, X):
        """Make predictions using the trained Random Forest model."""
        try:
            return self.model.predict(X)
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            raise

    def predict_proba(self, X):
        """Get prediction probabilities from the Random Forest model."""
        try:
            return self.model.predict_proba(X)
        except Exception as e:
            logger.error(f"Error in probability prediction: {str(e)}")
            raise 
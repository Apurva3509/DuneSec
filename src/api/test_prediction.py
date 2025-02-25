import requests
import pandas as pd
from typing import Dict
import sys

def load_test_data() -> pd.DataFrame:
    """Load and display test data samples."""
    try:
        test_data = pd.read_csv("data/processed/test.csv")
        return test_data
    except FileNotFoundError:
        print("Error: Test data file not found in data/processed/test.csv")
        sys.exit(1)

def display_samples(df: pd.DataFrame, n: int = 5):
    """Display first n samples with their indices and labels."""
    print("\nAvailable Test Samples:")
    print("-" * 50)
    for idx in range(n):
        label = "DDoS Attack" if df.iloc[idx]['Label'] == 1 else "Benign"
        print(f"Index {idx}: {label}")
    print(f"... and {len(df) - n} more samples")
    print("-" * 50)

def make_prediction(features: Dict) -> Dict:
    """Make prediction using the API."""
    url = "http://localhost:8000/predict"
    payload = {"features": features}
    
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error: {response.status_code}")
            print(response.json())
            return None
    except Exception as e:
        print(f"Error making request: {str(e)}")
        return None

def test_prediction():
    # Load test data
    test_data = load_test_data()
    
    while True:
        # Display sample options
        display_samples(test_data)
        
        # Get user input
        try:
            print("\nEnter sample index to test (or 'q' to quit):")
            user_input = input("> ")
            
            if user_input.lower() == 'q':
                print("Exiting...")
                break
            
            idx = int(user_input)
            if idx < 0 or idx >= len(test_data):
                print(f"Error: Index must be between 0 and {len(test_data)-1}")
                continue
            
            # Get features for selected sample
            sample = test_data.iloc[idx]
            features = sample.drop('Label', errors='ignore').to_dict()
            actual_label = "DDoS Attack" if sample['Label'] == 1 else "Benign"
            
            # Make prediction
            result = make_prediction(features)
            
            if result:
                print("\nPrediction Results:")
                print("-" * 20)
                print(f"Actual Label: {actual_label}")
                print(f"Predicted: {'DDoS Attack' if result['is_attack'] else 'Benign'}")
                print(f"Confidence: {result['probability']:.2%}")
                print(f"Response Time: {result['prediction_time_ms']:.2f}ms")
                
                # Show if prediction was correct
                predicted_label = "DDoS Attack" if result['is_attack'] else "Benign"
                if predicted_label == actual_label:
                    print("✅ Correct Prediction!")
                else:
                    print("❌ Incorrect Prediction!")
                
            print("\nWould you like to test another sample? (Press Enter to continue)")
            input()
            
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\nExiting...")
            break

if __name__ == "__main__":
    test_prediction() 
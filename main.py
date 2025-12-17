import os
from src.data_processing import load_and_split_data
from src.model_training import train_and_save

def main():
    # Ensure the model directory exists
    if not os.path.exists('models'):
        os.makedirs('models')
        
    data_path = 'data/raw/UCI_Credit_Card.csv'  # Modify if the extracted filename differs
    
    try:
        # 1. Load and split data
        print("Step 1: Loading Data...")
        X_train, X_test, y_train, y_test = load_and_split_data(data_path)
        
        # 2. Train model and save to file
        print("\nStep 2: Training & Saving Model...")
        train_and_save(X_train, y_train, X_test, y_test)
        
    except FileNotFoundError:
        print(f"Error: File not found -> {data_path}. Please ensure dataset has been downloaded correctly.")

if __name__ == "__main__":
    main()

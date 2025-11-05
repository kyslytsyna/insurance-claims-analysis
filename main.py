from src.data_preprocessing import load_and_clean_data
from src.feature_engineering import create_features
from src.model_training import train_model

def main():
    df = load_and_clean_data("data/raw/insurance_claims.csv")
    df = create_features(df)
    train_model(df)

if __name__ == "__main__":
    main()

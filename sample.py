import pandas as pd

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)

    # Drop biologically impossible values
    df = df[df['age'] > 0]
    df = df[df['resting_blood_pressure'] > 0]
    
    # Count missing values in each column
    missing_counts = df.isna().sum().sum()
    print(missing_counts)

    return df

def main():
    df = load_and_clean_data('patients.csv')

if __name__ == "__main__":
    main()

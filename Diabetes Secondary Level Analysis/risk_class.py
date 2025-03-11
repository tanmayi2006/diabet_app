import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
df = pd.read_csv('diabetes_prediction_dataset.csv')
df.head()
# Handling missing values (replace with mean for numerical features)
df['gender'].fillna(df['gender'].mode()[0], inplace=True)
df['smoking_history'].fillna(df['smoking_history'].mode()[0], inplace=True)
# ... handle other numerical features with missing values if any

# Convert categorical features to numerical using one-hot encoding
df = pd.get_dummies(df, columns=['gender', 'smoking_history'], drop_first=True)

# Feature scaling (optional, but often improves model performance)
from sklearn.preprocessing import StandardScaler

numerical_features = ['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level']
scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])
# Create composite features
df['health_index'] = df['hypertension'] + df['heart_disease']
df['glucose_HbA1c_interaction'] = df['HbA1c_level'] * df['blood_glucose_level']
# Drop original features
df.drop(['hypertension', 'heart_disease', 'HbA1c_level', 'blood_glucose_level'], axis=1, inplace=True)
def calculate_risk_factor(row):
    risk = 0
    if row['smoking_history_never'] == 0:  # Adjust based on your encoding
        risk += 1  # Example: +1 for any smoking history
    if row['bmi'] > 30:
        risk += 1
    if row['age'] > 50:
        risk += 1

    return risk

df['risk_factor'] = df.apply(calculate_risk_factor, axis=1)

# Example: Display the first few rows with the new 'risk_factor' column
print(df.head())

# Define features (X) and target (y) - assuming df is loaded
X = df.drop('diabetes', axis=1)
y = df['diabetes']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def categorize_risk(y_prob):
    """
    Categorize diabetes risk based on prediction probabilities

    Args:
        y_prob: Array of prediction probabilities

    Returns:
        Array of risk categories ('High', 'Moderate', 'Low')
    """
    risk_categories = np.empty(len(y_prob), dtype=object)

    # Define risk thresholds
    high_risk_threshold = 0.7
    moderate_risk_threshold = 0.3

    # Assign risk categories
    risk_categories[y_prob >= high_risk_threshold] = 'High'
    risk_categories[(y_prob >= moderate_risk_threshold) & (y_prob < high_risk_threshold)] = 'Moderate'
    risk_categories[y_prob < moderate_risk_threshold] = 'Low'

    return risk_categories

# Assuming you have y_test and model from your previous code
def analyze_risk_distribution(y_test, y_prob):
    """
    Analyze the distribution of risk categories in the test data

    Args:
        y_test: True labels
        y_prob: Prediction probabilities

    Returns:
        DataFrame with risk analysis
    """
    # Create a DataFrame with actual values and predictions
    results_df = pd.DataFrame({
        'actual': y_test,
        'probability': y_prob,
        'risk_category': categorize_risk(y_prob)
    })
    # Ensure y_prob and y_test have the same length
    assert len(y_prob) == len(y_test), "Mismatch in lengths of y_prob and y_test"

    # Reset index to prevent out-of-bounds error
    results_df = results_df.reset_index(drop=True)

    # Count of patients in each risk category
    risk_counts = results_df['risk_category'].value_counts().reset_index()
    risk_counts.columns = ['Risk Category', 'Count']

    # Calculate percentage
    total = risk_counts['Count'].sum()
    risk_counts['Percentage'] = (risk_counts['Count'] / total * 100).round(1)

    # Analyze actual diabetes cases in each risk category
    risk_actual = results_df.groupby('risk_category')['actual'].mean().reset_index()
    risk_actual.columns = ['Risk Category', 'Actual Diabetes Rate']
    risk_actual['Actual Diabetes Rate'] = (risk_actual['Actual Diabetes Rate'] * 100).round(1)

    # Merge the results
    risk_analysis = pd.merge(risk_counts, risk_actual, on='Risk Category')

    # Visualize the risk distribution
    plt.figure(figsize=(12, 5))

    # Plot 1: Risk category distribution
    plt.subplot(1, 2, 1)
    sns.barplot(x='Risk Category', y='Count', data=risk_counts, palette=['green', 'orange', 'red'])
    plt.title('Distribution of Risk Categories')
    plt.ylabel('Number of Patients')
    plt.xticks(rotation=0)

    # Plot 2: Actual diabetes rate by risk category
    plt.subplot(1, 2, 2)
    sns.barplot(x='Risk Category', y='Actual Diabetes Rate', data=risk_actual, palette=['green', 'orange', 'red'])
    plt.title('Actual Diabetes Rate by Risk Category')
    plt.ylabel('Actual Diabetes Rate (%)')
    plt.xticks(rotation=0)

    plt.tight_layout()
    plt.savefig('risk_distribution.png')
    plt.show()

    return risk_analysis, results_df

# Example usage:
# risk_analysis, results_df = analyze_risk_distribution(y_test, model.predict_proba(X_test)[:, 1])
# print(risk_analysis)

def identify_patients_by_risk(results_df, X_test, risk_level):
    """
    Identify patients in a specific risk category and their characteristics

    Args:
        results_df: DataFrame with risk categories
        X_test: Feature data for test patients
        risk_level: Risk level to filter ('High', 'Moderate', or 'Low')

    Returns:
        DataFrame with patient characteristics in the specified risk category
    """
    # Get indices of patients in the specified risk category
    risk_indices = results_df[results_df['risk_category'] == risk_level].index
    risk_indices = risk_indices[risk_indices < len(X_test)]  # Prevent out-of-bounds error

    # Get the corresponding patient data
    risk_patients = X_test.iloc[risk_indices].copy()

    # Add the probability and actual diabetes status
    risk_patients['probability'] = results_df.loc[risk_indices, 'probability']
    risk_patients['actual_diabetes'] = results_df.loc[risk_indices, 'actual']

    # Sort by probability in descending order
    risk_patients = risk_patients.sort_values('probability', ascending=(risk_level == 'Low'))
    
    # Check if risk_patients is empty before calculating summary statistics
    if risk_patients.empty:
        summary = {
            'count': 0,
            'diabetes_positive': 0,
            'diabetes_rate': np.nan,  # Set to NaN if empty
            'avg_probability': np.nan,  # Set to NaN if empty
            'min_probability': np.nan,  # Set to NaN if empty
            'max_probability': np.nan   # Set to NaN if empty
        }
    else:
        # Calculate summary statistics for this risk group
        summary = {
            'count': len(risk_patients),
            'diabetes_positive': risk_patients['actual_diabetes'].sum(),
            'diabetes_rate': (risk_patients['actual_diabetes'].mean() * 100).round(1),
            'avg_probability': risk_patients['probability'].mean().round(3),
        }

    return risk_patients, summary
# Example usage:
# # Get high risk patients
# high_risk_patients, high_risk_summary = identify_patients_by_risk(results_df, X_test, 'High')
# print(f"High Risk Summary: {high_risk_summary}")
# print(high_risk_patients.head())
#
# # Get moderate risk patients
# moderate_risk_patients, moderate_risk_summary = identify_patients_by_risk(results_df, X_test, 'Moderate')
# print(f"Moderate Risk Summary: {moderate_risk_summary}")
#
# # Get low risk patients
# low_risk_patients, low_risk_summary = identify_patients_by_risk(results_df, X_test, 'Low')
# print(f"Low Risk Summary: {low_risk_summary}")

def generate_risk_report(model, X_test, y_test):
    """
    Generate a comprehensive risk report from model predictions

    Args:
        model: Trained prediction model
        X_test: Test features
        y_test: True labels

    Returns:
        DataFrames with risk analysis and patient details
    """
    # Get prediction probabilities
    y_prob = model.predict_proba(X_test)[:, 1]

    # Analyze risk distribution
    risk_analysis, results_df = analyze_risk_distribution(y_test, y_prob)

    print("=== Diabetes Risk Distribution ===")
    print(risk_analysis)
    print("\n")

    # Get patients by risk category
    high_risk_patients, high_risk_summary = identify_patients_by_risk(results_df, X_test, 'High')
    moderate_risk_patients, moderate_risk_summary = identify_patients_by_risk(results_df, X_test, 'Moderate')
    low_risk_patients, low_risk_summary = identify_patients_by_risk(results_df, X_test, 'Low')

    print("=== High Risk Patients ===")
    print(f"Count: {high_risk_summary['count']}")
    print(f"Diabetes Rate: {high_risk_summary['diabetes_rate']}%")
    print(f"Avg Probability: {high_risk_summary['avg_probability']}")

    print("\n=== Moderate Risk Patients ===")
    print(f"Count: {moderate_risk_summary['count']}")
    print(f"Diabetes Rate: {moderate_risk_summary['diabetes_rate']}%")
    print(f"Avg Probability: {moderate_risk_summary['avg_probability']}")

    print("\n=== Low Risk Patients ===")
    print(f"Count: {low_risk_summary['count']}")
    print(f"Diabetes Rate: {low_risk_summary['diabetes_rate']}%")
    print(f"Avg Probability: {low_risk_summary['avg_probability']}")

    # Create a risk thresholds visualization
    plt.figure(figsize=(10, 6))
    sns.histplot(y_prob, bins=30, kde=True)
    plt.axvline(x=0.3, color='orange', linestyle='--', label='Moderate Risk Threshold (0.3)')
    plt.axvline(x=0.7, color='red', linestyle='--', label='High Risk Threshold (0.7)')
    plt.title('Distribution of Diabetes Probabilities with Risk Thresholds')
    plt.xlabel('Probability of Diabetes')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig('risk_thresholds.png')
    plt.show()

    return {
        'risk_analysis': risk_analysis,
        'results_df': results_df,
        'high_risk': high_risk_patients,
        'moderate_risk': moderate_risk_patients,
        'low_risk': low_risk_patients
    }
model = joblib.load('diabetes_gb_model.joblib')

risk_report = generate_risk_report(model, X_test, y_test)
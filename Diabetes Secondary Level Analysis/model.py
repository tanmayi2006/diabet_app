import matplotlib.pyplot as plt
import numpy as np
import joblib
import shap
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve, confusion_matrix
import pandas as pd
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

# Addressing class imbalance with class_weight
class_weights = {0: 1, 1: (y_train == 0).sum() / (y_train == 1).sum()}
print(f"Class weights: {class_weights}")

# Use RandomizedSearchCV for faster tuning
gb_params = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'subsample': [0.8, 1.0],
    'min_samples_split': [2, 5]
}

# Train Gradient Boosting model
gb_search = RandomizedSearchCV(
    GradientBoostingClassifier(random_state=42),
    param_distributions=gb_params,
    n_iter=10,
    cv=3,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)
gb_search.fit(X_train, y_train)
gb_model = gb_search.best_estimator_
print(f"Best parameters: {gb_search.best_params_}")

# Extract and visualize feature importances
def plot_feature_importance(model, X):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title('Feature Importance')
    plt.bar(range(X.shape[1]), importances[indices], align='center')
    plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.show()

    print("\nTop 5 important features:")
    for i in range(5):
        print(f"{X.columns[indices[i]]}: {importances[indices[i]]:.4f}")

    return X.columns[indices]

important_features = plot_feature_importance(gb_model, X)

# Evaluate the model
def evaluate_model(model, X_test, y_test, model_name="Model"):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f"\n{model_name} Evaluation:")
    print(f"F1 Score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"AUC: {auc:.4f}")

    # ROC Curve
    plt.figure(figsize=(8, 6))
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc='lower right')
    plt.savefig(f'roc_curve_{model_name.lower().replace(" ", "_")}.png')
    plt.show()

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Negative', 'Positive'])
    plt.yticks(tick_marks, ['Negative', 'Positive'])

    # Add text annotations to the confusion matrix
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png')
    plt.show()

    return {
        'f1': f1,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'auc': auc,
        'predictions': y_pred,
        'probabilities': y_prob
    }

gb_results = evaluate_model(gb_model, X_test, y_test, "Gradient Boosting")

# Select important features
top_features = []
threshold = 0.05
for i, feat in enumerate(X.columns):
    if gb_model.feature_importances_[i] > threshold:
        top_features.append(feat)

print(f"\nSelected {len(top_features)} important features out of {X.shape[1]}")
print(f"Selected features: {top_features}")

# Train a Random Forest model for better rule extraction
X_train_top = X_train[top_features]
X_test_top = X_test[top_features]

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42,
    class_weight=class_weights
)
rf_model.fit(X_train_top, y_train)
rf_results = evaluate_model(rf_model, X_test_top, y_test, "Random Forest")

# Train a Decision Tree for direct rule extraction (Tier 1 explainability)
dt_model = DecisionTreeClassifier(
    max_depth=4,
    random_state=42,
    class_weight=class_weights
)
dt_model.fit(X_train_top, y_train)
dt_results = evaluate_model(dt_model, X_test_top, y_test, "Decision Tree")

# Extract rules from the decision tree (Tier 1 explainability)
def extract_rules_from_tree(tree, feature_names):
    tree_rules = export_text(tree, feature_names=feature_names)
    print("\nDecision Tree Rules:")
    print(tree_rules)
    
    # Parse the rules into a more structured format
    parsed_rules = []
    lines = tree_rules.split('\n')
    for i, line in enumerate(lines):
        if "class: 1" in line:  # Focus on positive class rules
            rule_path = []
            indent = len(line) - len(line.lstrip('|'))
            j = i - 1
            while j >= 0 and ('class' not in lines[j] or len(lines[j]) - len(lines[j].lstrip('|')) < indent):
                if len(lines[j]) - len(lines[j].lstrip('|')) < indent and '|' in lines[j] and 'feature' in lines[j]:
                    rule_part = lines[j].strip().replace('|', '').strip()
                    rule_path.append(rule_part)
                    indent = len(lines[j]) - len(lines[j].lstrip('|'))
                j -= 1
            
            if rule_path:
                rule_path.reverse()
                parsed_rules.append(rule_path)
    
    return parsed_rules

rules = extract_rules_from_tree(dt_model, top_features)

# Create user-friendly rule explanations
def create_rule_explanations(rules):
    explanations = []
    for i, rule in enumerate(rules):
        explanation = f"Rule {i+1}: IF "
        conditions = []
        for condition in rule:
            # Clean up the condition format
            clean_condition = condition.replace('feature_', '').replace(' <= ', ' is less than or equal to ').replace(' > ', ' is greater than ')
            conditions.append(clean_condition)
        
        explanation += " AND ".join(conditions)
        explanation += " THEN risk of diabetes is HIGH"
        explanations.append(explanation)
    
    return explanations

rule_explanations = create_rule_explanations(rules)
print("\nUser-Friendly Rule Explanations:")
for explanation in rule_explanations:
    print(explanation)

# SHAP values for detailed feature contribution (enhanced explainability)
# Modified: Using interventional approach and disabling additivity check
# Fix the SHAP explainer code
try:
    # First try with TreeExplainer - using correct parameters
    explainer = shap.TreeExplainer(
        gb_model,
        feature_perturbation="interventional", 
        model_output="raw"  # Changed from "probability" to "raw"
    )
    shap_values = explainer.shap_values(X_test, check_additivity=False)  # Use full X_test instead of X_test_top
    
    # Convert shap_values to the right format if it's a list
    if isinstance(shap_values, list):
        # For binary classification, take the positive class (index 1)
        shap_values_for_viz = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        shap_values_for_viz = shap_values
    
    # Plot SHAP summary with correct feature set
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values_for_viz, X_test, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig('shap_summary.png')
    plt.show()

    # Plot SHAP values for individual features
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values_for_viz, X_test, show=False)
    plt.tight_layout()
    plt.savefig('shap_values.png')
    plt.show()
except Exception as e:
    print(f"Error using TreeExplainer: {e}")
    print("Falling back to KernelExplainer...")
    
    # Fallback to KernelExplainer with correct feature set
    # Make sure to use the SAME features that the model was trained on
    background = shap.kmeans(X_train, 10)  # Use X_train instead of X_train_top
    explainer = shap.KernelExplainer(gb_model.predict, background)  # Use predict instead of predict_proba
    shap_values = explainer.shap_values(X_test.iloc[:100])  # Use X_test instead of X_test_top
    
    # Plot SHAP summary
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test.iloc[:100], plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig('shap_summary.png')
    plt.show()
# Compute feature thresholds from decision tree
def extract_feature_thresholds(rules, features):
    thresholds = {feature: {'min': float('inf'), 'max': float('-inf')} for feature in features}
    
    for rule in rules:
        for condition in rule:
            for feature in features:
                if feature in condition:
                    value = float(condition.split()[-1])
                    operator = 'less than or equal to' if '<=' in condition else 'greater than'
                    
                    if operator == 'less than or equal to':
                        thresholds[feature]['max'] = min(thresholds[feature]['max'], value) if thresholds[feature]['max'] != float('-inf') else value
                    else:
                        thresholds[feature]['min'] = max(thresholds[feature]['min'], value) if thresholds[feature]['min'] != float('inf') else value
    
    # Clean up thresholds that weren't set
    for feature in thresholds:
        if thresholds[feature]['min'] == float('inf'):
            thresholds[feature]['min'] = None
        if thresholds[feature]['max'] == float('-inf'):
            thresholds[feature]['max'] = None
    
    return thresholds

feature_thresholds = extract_feature_thresholds(rules, top_features)
print("\nFeature Thresholds:")
for feature, values in feature_thresholds.items():
    print(f"{feature}: {values}")

# Create Tier 2 explanation templates
# Clinical explanations for features
clinical_explanations = {
    'BMI': {
        'high': 'Your BMI (Body Mass Index) is elevated, which increases your risk of developing type 2 diabetes. BMI is a measure of body fat based on height and weight.',
        'normal': 'Your BMI (Body Mass Index) is within a healthy range, which is positive for diabetes prevention.',
        'remedy': 'Aim for a BMI between 18.5 and 24.9 through balanced diet and regular exercise.'
    },
    'Age': {
        'high': 'Your age is a risk factor for diabetes. Diabetes risk increases with age, particularly after 45.',
        'normal': 'Your age currently places you in a lower risk category for diabetes.',
        'remedy': 'Regular health screenings are important as you age to catch early signs of diabetes.'
    },
    'Glucose': {
        'high': 'Your blood glucose level is elevated, which is a strong indicator of diabetes risk.',
        'normal': 'Your blood glucose level is within a normal range, which is positive for diabetes prevention.',
        'remedy': 'Maintain blood glucose levels through balanced meals, regular physical activity, and limiting sugary foods.'
    },
    'BloodPressure': {
        'high': 'Your blood pressure is elevated, which is associated with increased diabetes risk and cardiovascular complications.',
        'normal': 'Your blood pressure is within a healthy range, which reduces your risk of diabetes complications.',
        'remedy': 'Maintain healthy blood pressure through reduced sodium intake, regular exercise, and stress management.'
    },
    'Insulin': {
        'high': 'Your insulin level is elevated, which could indicate insulin resistance, a key factor in type 2 diabetes.',
        'normal': 'Your insulin level is within a normal range, indicating good insulin sensitivity.',
        'remedy': 'Improve insulin sensitivity through regular exercise, maintaining a healthy weight, and consuming complex carbohydrates.'
    },
    'DiabetesPedigreeFunction': {
        'high': 'Your genetic predisposition to diabetes is significant based on your family history.',
        'normal': 'Your genetic predisposition to diabetes appears to be lower than average.',
        'remedy': 'With family history of diabetes, more frequent screening and vigilant lifestyle management is recommended.'
    },
    'SkinThickness': {
        'high': 'Your skin fold thickness measurement may indicate higher body fat percentage, a risk factor for diabetes.',
        'normal': 'Your skin thickness measurement is within a normal range.',
        'remedy': 'Regular exercise and a balanced diet can help maintain healthy body composition.'
    },
    'Pregnancies': {
        'high': 'Multiple pregnancies can affect insulin sensitivity and increase diabetes risk in women.',
        'normal': 'Your pregnancy history doesn\'t significantly increase your diabetes risk.',
        'remedy': 'Regular postpartum health check-ups are recommended, especially after gestational diabetes.'
    }
}
clinical_explanations.update({
    'glucose_HbA1c_interaction': {
        'high': 'Your combination of blood glucose and HbA1c levels indicates potential insulin resistance and elevated risk for diabetes.',
        'normal': 'Your blood glucose and HbA1c interaction is within normal range.',
        'remedy': 'Monitor both your blood glucose levels and HbA1c. Maintain a low-glycemic diet and engage in regular physical activity.'
    },
    'health_index': {
        'high': 'Your combined hypertension and heart disease indicators suggest elevated cardiovascular risk, which often correlates with diabetes.',
        'normal': 'Your cardiovascular health indicators are within normal ranges.',
        'remedy': 'Maintain heart health through regular cardiovascular exercise, a low-sodium diet, and regular blood pressure monitoring.'
    },
    'bmi': {  # Make sure this matches exactly with your feature name (case-sensitive)
        'high': 'Your BMI (Body Mass Index) is elevated, which increases your risk of developing type 2 diabetes.',
        'normal': 'Your BMI (Body Mass Index) is within a healthy range, which is positive for diabetes prevention.',
        'remedy': 'Aim for a BMI between 18.5 and 24.9 through balanced diet and regular exercise.'
    },
    'age': {  # Make sure this matches exactly with your feature name (case-sensitive)
        'high': 'Your age is a risk factor for diabetes. Diabetes risk increases with age, particularly after 45.',
        'normal': 'Your age currently places you in a lower risk category for diabetes.',
        'remedy': 'Regular health screenings are important as you age to catch early signs of diabetes.'
    }
})
# Function to create personalized explanation without requiring SHAP
def create_personalized_explanation(patient_data, model, thresholds, clinical_info):
    # Get model prediction and feature contributions
    features = list(patient_data.keys())
    input_df = pd.DataFrame([patient_data])
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0, 1]
    
    # For feature importance, use the model's feature_importances_ instead of SHAP
    # and multiply by the feature values to get a rough approximation of contribution
    feature_importances = model.feature_importances_
    feature_indices = {f: i for i, f in enumerate(model.feature_names_in_)}
    
    contributions = []
    for feature in features:
        idx = feature_indices.get(feature)
        if idx is not None:
            # Scale the contribution by the feature value relative to its mean/range
            # Simple heuristic - if value is above average, importance is positive, otherwise negative
            raw_value = patient_data[feature]
            importance = feature_importances[idx]
            # This is a simplified approximation - not as accurate as SHAP values
            contribution = importance * (raw_value / (raw_value + 1))  # Simple scaling
            contributions.append((feature, contribution))
    
    # Sort features by contribution magnitude
    sorted_contributions = sorted(contributions, key=lambda x: abs(x[1]), reverse=True)
    
    # Create the explanation
    explanation = {
        "prediction": "Diabetes" if prediction == 1 else "No Diabetes",
        "probability": round(float(probability * 100), 1),
        "risk_level": "High" if probability > 0.7 else ("Medium" if probability > 0.3 else "Low"),
        "feature_analysis": [],
        "summary": "",
        "recommendations": []
    }
    
    # Add detailed feature analysis
    for feature, contribution in sorted_contributions:
        if feature not in clinical_info:
            continue
            
        # Determine if the feature value is high or normal
        value = patient_data[feature]
        status = "normal"
        
        # Check against thresholds if available
        if feature in thresholds:
            threshold = thresholds[feature]
            if threshold['min'] is not None and value > threshold['min']:
                status = "high"
            elif threshold['max'] is not None and value <= threshold['max']:
                status = "normal"
        
        # Use contribution to determine status if thresholds aren't available
        elif contribution > 0:
            status = "high"
        
        # Add feature analysis
        feature_detail = {
            "feature": feature,
            "value": value,
            "contribution": round(float(contribution), 3),
            "explanation": clinical_info[feature][status],
            "recommendation": clinical_info[feature]['remedy']
        }
        
        explanation["feature_analysis"].append(feature_detail)
        
        # Add recommendation to the list
        if status == "high":
            explanation["recommendations"].append(clinical_info[feature]['remedy'])
    
    # Create a summary based on the top contributing factors
    top_factors = explanation["feature_analysis"][:3]
    if prediction == 1:
        risk_factors = [f["feature"] for f in top_factors]
        if risk_factors:
            explanation["summary"] = f"Your primary risk factors for diabetes are: {', '.join(risk_factors)}. "
        explanation["summary"] += f"Your overall risk of diabetes is {explanation['risk_level'].lower()} ({explanation['probability']}%)."
    else:
        explanation["summary"] = f"Your diabetes risk is {explanation['risk_level'].lower()} ({explanation['probability']}%). "
        if top_factors:
            explanation["summary"] += f"Continue monitoring your {top_factors[0]['feature']} as it has the most significant impact on your assessment."
    
    return explanation

example_patient = {
    'Pregnancies': 0,
    'Glucose': 137,
    'BloodPressure': 40,
    'SkinThickness': 35,
    'Insulin': 168,
    'BMI': 43.1,
    'DiabetesPedigreeFunction': 2.288,
    'Age': 33
}

# Create explanation for the example patient
if set(example_patient.keys()).issubset(set(top_features)):
    patient_explanation = create_personalized_explanation(
        {feature: example_patient[feature] for feature in top_features if feature in example_patient},
        gb_model,
        feature_thresholds,
        clinical_explanations
    )
    
    print("\nPatient Explanation Example:")
    print(f"Prediction: {patient_explanation['prediction']}")
    print(f"Probability: {patient_explanation['probability']}%")
    print(f"Risk Level: {patient_explanation['risk_level']}")
    print(f"Summary: {patient_explanation['summary']}")
    
    print("\nFeature Analysis:")
    for feature in patient_explanation['feature_analysis']:
        print(f"- {feature['feature']} (value: {feature['value']}): {feature['explanation']}")
        
    print("\nRecommendations:")
    for recommendation in patient_explanation['recommendations']:
        print(f"- {recommendation}")

# Save models
joblib.dump(gb_model, 'diabetes_gb_model.joblib', compress=3)
joblib.dump(rf_model, 'diabetes_rf_model.joblib', compress=3)
joblib.dump(dt_model, 'diabetes_dt_model.joblib', compress=3)
print("\nAll models saved successfully")

# Create a simple tabular representation of feature rankings and explanations
def generate_feature_ranking_table():
    # Combine feature importances from different models
    feature_ranking = pd.DataFrame({
        'Feature': top_features,
        'GradientBoosting_Importance': [gb_model.feature_importances_[list(X.columns).index(f)] for f in top_features],
        'RandomForest_Importance': [rf_model.feature_importances_[list(X_train_top.columns).index(f)] for f in top_features]
    })
    
    # Sort by average importance
    feature_ranking['Average_Importance'] = (feature_ranking['GradientBoosting_Importance'] + 
                                            feature_ranking['RandomForest_Importance']) / 2
    feature_ranking = feature_ranking.sort_values('Average_Importance', ascending=False)
    
    # Add clinical explanations
    feature_ranking['Clinical_Explanation'] = [
        clinical_explanations.get(feature, {'high': 'No clinical explanation available'})['high']
        for feature in feature_ranking['Feature']
    ]
    
    feature_ranking['Recommendation'] = [
        clinical_explanations.get(feature, {'remedy': 'No recommendation available'})['remedy']
        for feature in feature_ranking['Feature']
    ]
    
    # Display the table
    print("\nFeature Ranking and Clinical Explanation Table:")
    print(feature_ranking[['Feature', 'Average_Importance', 'Clinical_Explanation', 'Recommendation']])
    
    # Save to CSV
    feature_ranking.to_csv('feature_ranking_table.csv', index=False)
    print("Feature ranking table saved to 'feature_ranking_table.csv'")
    
    return feature_ranking

feature_table = generate_feature_ranking_table()


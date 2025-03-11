import joblib
import numpy as np
import pandas as pd
from model import feature_thresholds
from model import clinical_explanations
# Load the models
gb_model = joblib.load('diabetes_gb_model.joblib')
dt_model = joblib.load('diabetes_dt_model.joblib')

# Important features in the correct order
top_features = [glucose_HbA1c_interaction: 0.7671,
age,
bmi,
health_index,
gender_Male]

# Feature thresholds for rule-based explanations


# Clinical explanations for features

def predict_diabetes_with_explanation(patient_data):
    '''
    Make a diabetes prediction with detailed explanation
    
    Args:
        patient_data: dict with patient features
        
    Returns:
        dict: Prediction results with detailed explanations
    '''
    # Convert input to DataFrame with correct features
    input_data = {}
    for feature in top_features:
        if feature in patient_data:
            input_data[feature] = patient_data[feature]
        else:
            print(f"Warning: Missing feature {feature}")
            return {"error": f"Missing required feature: {feature}"}
    
    df = pd.DataFrame([input_data])
    
    # Get prediction and probability using the gradient boosting model
    prediction = gb_model.predict(df)[0]
    probability = gb_model.predict_proba(df)[0, 1]
    
    # For feature importance, use the model's feature_importances_ instead of SHAP
    # and multiply by the feature values to get a rough approximation of contribution
    feature_importances = gb_model.feature_importances_
    feature_indices = {f: i for i, f in enumerate(gb_model.feature_names_in_)}
    
    contributions = []
    for feature in input_data.keys():
        idx = feature_indices.get(feature)
        if idx is not None:
            # Scale the contribution by the feature value relative to its mean/range
            # Simple heuristic - if value is above average, importance is positive, otherwise negative
            raw_value = input_data[feature]
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
        if feature not in clinical_explanations:
            continue
            
        # Determine if the feature value is high or normal
        value = patient_data[feature]
        status = "normal"
        
        # Check against thresholds if available
        if feature in feature_thresholds:
            threshold = feature_thresholds[feature]
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
            "explanation": clinical_explanations[feature][status],
            "recommendation": clinical_explanations[feature]['remedy']
        }
        
        explanation["feature_analysis"].append(feature_detail)
        
        # Add recommendation to the list
        if status == "high":
            explanation["recommendations"].append(clinical_explanations[feature]['remedy'])
    
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
    
    # Generate rule-based explanation using decision tree (Tier 1)
    rule_prediction = dt_model.predict(df)[0]
    decision_path = dt_model.decision_path(df)
    
    # Get the decisions made on this patient
    node_indicator = decision_path.toarray()[0]
    leaf_id = np.argmax(node_indicator)
    
    feature_names = top_features
    node_index = leaf_id
    
    tier1_rules = []
    if rule_prediction == 1:  # Only provide rules for positive predictions
        # Traverse decision path backwards
        while node_index != 0:  # 0 is the root node
            parent_node = np.where(node_indicator[:node_index] == 1)[0][-1]
            if dt_model.tree_.feature[parent_node] >= 0:  # -2 means leaf node
                feature = feature_names[dt_model.tree_.feature[parent_node]]
                threshold = dt_model.tree_.threshold[parent_node]
                
                if node_index == dt_model.tree_.children_left[parent_node]:
                    tier1_rules.append(f"{feature} <= {threshold:.2f}")
                else:
                    tier1_rules.append(f"{feature} > {threshold:.2f}")
            
            node_index = parent_node
    
    if tier1_rules:
        explanation["rule_based_explanation"] = "IF " + " AND ".join(reversed(tier1_rules)) + " THEN risk of diabetes is HIGH"
    else:
        explanation["rule_based_explanation"] = "No specific rule applies to this prediction."
    
    return explanation

# Example usage
if __name__ == "__main__":
    # Example patient data
    sample_data = {
        'Pregnancies': 0,
        'Glucose': 137,
        'BloodPressure': 40,
        'SkinThickness': 35,
        'Insulin': 168,
        'BMI': 43.1,
        'DiabetesPedigreeFunction': 2.288,
        'Age': 33
    }

    result = predict_diabetes_with_explanation(sample_data)
    
    print("============= DIABETES RISK ASSESSMENT =============")
    print(f"Prediction: {result['prediction']}")
    print(f"Probability: {result['probability']}%")
    print(f"Risk Level: {result['risk_level']}")
    print(f"Summary: {result['summary']}")
    
    print("\\nRule-Based Explanation (Tier 1):")
    print(result['rule_based_explanation'])
    
    print("\\nFeature Analysis (Tier 2):")
    for feature in result['feature_analysis']:
        print(f"- {feature['feature']} (value: {feature['value']}): {feature['explanation']}")
        
    print("\\nPersonalized Recommendations:")
    for i, recommendation in enumerate(result['recommendations']):
        print(f"{i+1}. {recommendation}")

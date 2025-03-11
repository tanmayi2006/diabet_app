import pandas as pd
import numpy as np
import lightgbm as lgb
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, recall_score
import shap
import ee
import geemap

# Initialize GEE
try:
    ee.Initialize(project='mess-ba866')
except Exception as e:
    ee.Authenticate()
    ee.Initialize(project='mess-ba866')

# Load and preprocess data
print("Loading and preprocessing data...")
df = pd.read_csv('/content/diabetes_data_upload.csv')
print(df.head())

# Preprocessing function (unchanged)
def preprocess_data(df):
    df_processed = df.copy()
    df_processed['Gender'] = df_processed['Gender'].map({'Male': 1, 'Female': 0})
    for col in df_processed.columns[2:-1]:
        df_processed[col] = df_processed[col].map({'Yes': 1, 'No': 0})
    df_processed['class'] = df_processed['class'].map({'Positive': 1, 'Negative': 0})
    df_processed['symptom_index'] = df_processed[['Polyuria', 'Polydipsia', 'Polyphagia']].sum(axis=1)
    df_processed['risk_factor'] = df_processed.apply(
        lambda row: sum([row['Polyuria'], row['Polydipsia'], row['Obesity'], row['Age'] > 50]), axis=1
    )
    return df_processed

df = preprocess_data(df)
X = df.drop('class', axis=1)
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train LightGBM (unchanged)
print("\nTraining LightGBM model...")
lgb_params = {
    'n_estimators': [50, 100], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 5], 'subsample': [0.8, 1.0]
}
lgb_search = RandomizedSearchCV(
    lgb.LGBMClassifier(random_state=42), param_distributions=lgb_params, n_iter=10, cv=3, scoring='f1', n_jobs=-1
)
lgb_search.fit(X_train, y_train)
lgb_model = lgb_search.best_estimator_
print(f"Best parameters: {lgb_search.best_params_}")

# Evaluate LightGBM (unchanged)
y_pred_lgb = lgb_model.predict(X_test)
y_pred_proba_lgb = lgb_model.predict_proba(X_test)[:, 1]
print("\nLightGBM Model Metrics:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_lgb):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_lgb):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred_lgb):.4f}")
print(f"AUC: {roc_auc_score(y_test, y_pred_proba_lgb):.4f}")

# Train TensorFlow model (unchanged)
num_features = X.shape[1]
X_all = X.values.astype(np.float32)
y_lgb_prob = lgb_model.predict_proba(X_all)[:, 1].astype(np.float32)

def create_tf_model(input_shape):
    inputs = tf.keras.Input(shape=(input_shape,))
    x = tf.keras.layers.Dense(64, activation='relu')(inputs)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dense(16, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

tf_model = create_tf_model(num_features)
tf_model.fit(X_all, y_lgb_prob, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# Convert to TFLite (unchanged)
converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

tflite_path = "diabetes_model.tflite"
with open(tflite_path, 'wb') as f:
    f.write(tflite_model)

# Test TFLite model (unchanged)
interpreter = tf.lite.Interpreter(model_path=tflite_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def tflite_predict(X):
    predictions = []
    for i in range(len(X)):
        sample = X.iloc[i:i+1].values.astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], sample)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])[0][0]
        predictions.append(output)
    return np.array(predictions)

y_pred_tflite_proba = tflite_predict(X_test)
y_pred_tflite = (y_pred_tflite_proba > 0.5).astype(int)

print("\nTFLite Model Metrics:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_tflite):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_tflite):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred_tflite):.4f}")
print(f"AUC: {roc_auc_score(y_test, y_pred_tflite_proba):.4f}")

# Updated fetch_gee_data function with PM2.5 instead of Aerosol Index

def fetch_gee_data(lat, lon, date_start='2023-01-01', date_end='2023-03-07'):
    point = ee.Geometry.Point([lon, lat])
    buffer = point.buffer(10000).bounds()  # 10km buffer for better coverage

    # Initialize return dictionary
    env_data = {'no2': None, 'lst': None, 'aerosol_index': None, 'ndvi': None}

    # NO2 from Sentinel-5P (already working)
    try:
        s5p_no2 = ee.ImageCollection('COPERNICUS/S5P/OFFL/L3_NO2')\
            .filterDate(date_start, date_end)\
            .filterBounds(buffer)\
            .select('tropospheric_NO2_column_number_density')\
            .mean()
        env_data['no2'] = s5p_no2.reduceRegion(
            reducer=ee.Reducer.mean(), geometry=buffer, scale=1000
        ).get('tropospheric_NO2_column_number_density').getInfo()
    except ee.EEException as e:
        print(f"Error fetching NO2 data: {e}")

    # LST from MODIS
    try:
        modis_lst = ee.ImageCollection('MODIS/061/MOD11A1')\
            .filterDate(date_start, date_end)\
            .filterBounds(buffer)\
            .select('LST_Day_1km')\
            .mean()
        lst_value = modis_lst.reduceRegion(
            reducer=ee.Reducer.mean(), geometry=buffer, scale=1000
        ).get('LST_Day_1km').getInfo()
        env_data['lst'] = (lst_value * 0.02) - 273.15 if lst_value else None
    except ee.EEException as e:
        print(f"Error fetching LST data: {e}")

    # Aerosol Index from Sentinel-5P (replacing PM2.5)
    try:
        s5p_aerosol = ee.ImageCollection('COPERNICUS/S5P/OFFL/L3_AER_AI')\
            .filterDate(date_start, date_end)\
            .filterBounds(buffer)\
            .select('absorbing_aerosol_index')\
            .mean()
        env_data['aerosol_index'] = s5p_aerosol.reduceRegion(
            reducer=ee.Reducer.mean(), geometry=buffer, scale=1000
        ).get('absorbing_aerosol_index').getInfo()
    except ee.EEException as e:
        print(f"Error fetching Aerosol Index data: {e}")

    # NDVI from MODIS
    try:
        ndvi = ee.ImageCollection('MODIS/061/MOD13Q1')\
            .filterDate(date_start, date_end)\
            .filterBounds(buffer)\
            .select('NDVI')\
            .mean()
        ndvi_raw = ndvi.reduceRegion(
            reducer=ee.Reducer.mean(), geometry=buffer, scale=250
        ).get('NDVI').getInfo()
        env_data['ndvi'] = ndvi_raw / 10000 if ndvi_raw else None  # Scale to 0-1
    except ee.EEException as e:
        print(f"Error fetching NDVI data: {e}")

    return env_data


# Sample prediction
sample_index = 234
sample = X.iloc[[sample_index]]
true_label = y.iloc[sample_index]

lat, lon = 37.7749, -122.4194  # San Francisco, CA
env_data = fetch_gee_data(lat, lon)
print(f"\nEnvironmental Data for Sample {sample_index} (Lat: {lat}, Lon: {lon}):")
print(f"NO2 Concentration: {env_data['no2']:.4e} mol/m²" if env_data['no2'] else "NO2: Unavailable")
print(f"Land Surface Temperature: {env_data['lst']:.2f} °C" if env_data['lst'] else "LST: Unavailable")

print(f"NDVI: {env_data['ndvi']:.2f}" if env_data['ndvi'] else "NDVI: Unavailable")

# LightGBM prediction
lgb_pred_proba = lgb_model.predict_proba(sample)[:, 1][0]
lgb_pred = int(lgb_pred_proba > 0.5)

# TFLite prediction
interpreter.set_tensor(input_details[0]['index'], sample.values.astype(np.float32))
interpreter.invoke()
tflite_pred_proba = interpreter.get_tensor(output_details[0]['index'])[0][0]
tflite_pred = int(tflite_pred_proba > 0.5)

print(f"\nSample {sample_index} Prediction:")
print(f"True Label: {true_label}")
print(f"LightGBM Prediction: {lgb_pred} (Probability: {lgb_pred_proba:.4f})")
print(f"TFLite Prediction: {tflite_pred} (Probability: {tflite_pred_proba:.4f})")

# SHAP explanation
background = shap.sample(X, 50)
explainer = shap.KernelExplainer(
    lambda x: tflite_predict(pd.DataFrame(x, columns=X.columns)), background
)
shap_values = explainer.shap_values(sample.values)[0]
top_indices = np.argsort(np.abs(shap_values))[::-1][:5]
top_features = X.columns[top_indices].tolist()

# Define clinical explanations
clinical_explanations = {
    "Age": {
        "high": "Advanced age is a known risk factor for diabetes and other metabolic disorders.",
        "normal": "Age is within a normal range.",
        "suggestion": "Regular health checkups are recommended."
    },
    "Gender": {
        "high": "Certain diabetes risk factors may vary by gender.",
        "normal": "No significant impact on diabetes risk.",
        "suggestion": "Monitor based on individual risk factors."
    },
    "Polyuria": {
        "high": "Frequent urination can be a sign of high blood sugar levels.",
        "normal": "No excessive urination reported.",
        "suggestion": "Monitor fluid intake and check blood sugar levels."
    },
    "Polydipsia": {
        "high": "Excessive thirst is a common symptom of diabetes.",
        "normal": "No unusual thirst levels reported.",
        "suggestion": "Consider checking blood glucose levels if persistent."
    },
    "sudden weight loss": {
        "high": "Unexplained weight loss may indicate uncontrolled diabetes.",
        "normal": "Weight is stable.",
        "suggestion": "Consult a doctor for potential metabolic concerns."
    },
    "weakness": {
        "high": "Weakness can be a symptom of high or low blood sugar.",
        "normal": "No unusual fatigue or weakness reported.",
        "suggestion": "Monitor energy levels and consider a balanced diet."
    },
    "Polyphagia": {
        "high": "Excessive hunger may indicate blood sugar fluctuations.",
        "normal": "Appetite is within a normal range.",
        "suggestion": "Monitor food intake and check glucose levels."
    },
    "Genital thrush": {
        "high": "Frequent fungal infections can be associated with diabetes.",
        "normal": "No infections reported.",
        "suggestion": "Maintain good hygiene and monitor blood sugar levels."
    },
    "visual blurring": {
        "high": "Blurry vision can result from fluctuating blood sugar levels.",
        "normal": "No visual disturbances reported.",
        "suggestion": "Consider an eye examination and blood sugar test."
    },
    "Itching": {
        "high": "Persistent itching can be linked to diabetes or skin infections.",
        "normal": "No unusual itching reported.",
        "suggestion": "Keep skin hydrated and monitor for infections."
    },
    "Irritability": {
        "high": "Mood swings and irritability can be related to blood sugar fluctuations.",
        "normal": "Mood is stable.",
        "suggestion": "Maintain stable blood sugar levels through diet."
    },
    "delayed healing": {
        "high": "Slow healing of wounds is a common sign of diabetes.",
        "normal": "Normal healing observed.",
        "suggestion": "Monitor for infections and maintain good wound care."
    },
    "partial paresis": {
        "high": "Weakness or partial paralysis can be a neurological complication.",
        "normal": "No muscle weakness reported.",
        "suggestion": "Consider neurological evaluation if persistent."
    },
    "muscle stiffness": {
        "high": "Muscle stiffness can be associated with metabolic imbalances.",
        "normal": "No unusual stiffness reported.",
        "suggestion": "Regular physical activity can help maintain mobility."
    },
    "Alopecia": {
        "high": "Hair loss may be associated with endocrine disorders.",
        "normal": "No significant hair loss reported.",
        "suggestion": "Monitor thyroid and metabolic health."
    },
    "Obesity": {
        "high": "Obesity is a major risk factor for diabetes and metabolic disorders.",
        "normal": "Weight is within a healthy range.",
        "suggestion": "Maintain a balanced diet and regular exercise routine."
    },
    # Add engineered features
    "symptom_index": {
        "high": "Multiple diabetes symptoms are present, significantly increasing risk.",
        "normal": "Few or no classic diabetes symptoms are present.",
        "suggestion": "Monitor for changes in urination, thirst, and hunger patterns."
    },
    "risk_factor": {
        "high": "Multiple risk factors for diabetes are present.",
        "normal": "Few risk factors for diabetes are present.",
        "suggestion": "Regularly monitor blood glucose levels and maintain a healthy lifestyle."
    },
   "NO2": {
        "high": "High nitrogen dioxide (NO2) levels, a marker of air pollution, can increase inflammation and oxidative stress, potentially worsening insulin resistance and raising diabetes risk.",
        "normal": "NO2 levels are within a safe range and unlikely to significantly impact diabetes risk.",
        "suggestion": "Avoid prolonged outdoor exposure during high NO2 periods to reduce health stress."
    },
    "Heat": {
        "high": "Elevated temperatures can cause dehydration and stress the body, potentially leading to blood sugar spikes, especially in those predisposed to diabetes.",
        "normal": "Current temperatures are comfortable and unlikely to affect blood sugar control.",
        "suggestion": "Stay hydrated and cool during hot weather to support metabolic health."
    },
    "AirPollution": {
        "high": "High aerosol levels (air pollution) may contribute to systemic inflammation, impairing insulin sensitivity and increasing diabetes risk over time.",
        "normal": "Air quality is satisfactory, posing minimal risk to diabetes development.",
        "suggestion": "Limit outdoor activity during high pollution days to protect your health."
    },
    "GreenSpace": {
        "high": "Access to green spaces (high NDVI) supports physical activity and stress reduction, which can lower diabetes risk.",
        "normal": "Limited green space might reduce opportunities for exercise, subtly elevating diabetes risk.",
        "suggestion": "Seek out parks or outdoor areas for regular physical activity to improve insulin sensitivity."
    }}

# Updated explanation function
def assign_risk(probability):
    if probability >= 0.7: return "High"
    elif probability >= 0.4: return "Moderate"
    else: return "Low"

def generate_explanation(patient_data, probability, top_features, env_data):
    risk_level = assign_risk(probability)
    explanation = {
        "risk_level": risk_level,
        "clinical_details": [],
        "environmental_details": [],
        "suggestions": [],
        "summary": ""
    }

    # Clinical features
    for feature in top_features:
        value = patient_data[feature]
        status = 'high' if (feature == 'Age' and value > 50) or (feature != 'Age' and value > 0) else 'normal'
        explanation['clinical_details'].append(f"{feature}: {clinical_explanations[feature][status]}")
        if status == 'high':
            explanation['suggestions'].append(clinical_explanations[feature]['suggestion'])

    # Environmental factors with diabetes-specific impact
    no2_status = 'high' if env_data['no2'] and env_data['no2'] > 0.0001 else 'normal'  # Example threshold
    explanation['environmental_details'].append(
        f"NO2: {clinical_explanations['NO2'][no2_status]} (Value: {env_data['no2']:.4e} mol/m²)"
    )
    if no2_status == 'high':
        explanation['suggestions'].append(clinical_explanations['NO2']['suggestion'])

    lst_status = 'high' if env_data['lst'] and env_data['lst'] > 30 else 'normal'  # 30°C threshold
    explanation['environmental_details'].append(
        f"Heat: {clinical_explanations['Heat'][lst_status]} (Temperature: {env_data['lst']:.2f} °C)"
    )
    if lst_status == 'high':
        explanation['suggestions'].append(clinical_explanations['Heat']['suggestion'])

    aerosol_status = 'high' if env_data['aerosol_index'] and env_data['aerosol_index'] > 1 else 'normal'  # Aerosol threshold
    explanation['environmental_details'].append(
        f"Air Pollution: {clinical_explanations['AirPollution'][aerosol_status]} (Aerosol Index: {env_data['aerosol_index']:.2f})"
    )
    if aerosol_status == 'high':
        explanation['suggestions'].append(clinical_explanations['AirPollution']['suggestion'])

    ndvi_status = 'normal' if env_data['ndvi'] and env_data['ndvi'] < 0.2 else 'high'  # Low greenness
    explanation['environmental_details'].append(
        f"Green Space: {clinical_explanations['GreenSpace'][ndvi_status]} (NDVI: {env_data['ndvi']:.2f})"
    )
    if ndvi_status == 'normal':
        explanation['suggestions'].append(clinical_explanations['GreenSpace']['suggestion'])

    # Summary
    explanation['summary'] = f"Your diabetes risk is {risk_level.lower()} ({probability:.0%}). "
    key_factors = [f for f in top_features if patient_data[f] > 0 or (f == 'Age' and patient_data[f] > 50)]
    if risk_level == 'High':
        if key_factors:
            explanation['summary'] += f"Key clinical factors include: {', '.join(key_factors)}. "
        if no2_status == 'high' or lst_status == 'high' or aerosol_status == 'high':
            explanation['summary'] += "Environmental stressors like poor air quality or heat may also be contributing."
    elif risk_level == 'Moderate':
        explanation['summary'] += "Monitor symptoms and environmental conditions closely for potential changes."
    else:
        explanation['summary'] += "Your risk is low—keep up your healthy habits!"

    return explanation

# Example usage (unchanged from your code)
explanation = generate_explanation(sample.iloc[0], tflite_pred_proba, top_features, env_data)

# Updated output
print("\nExplanation for Sample:")
print(f"Risk Level: {explanation['risk_level']}")
print(f"Summary: {explanation['summary']}")
print("\nClinical Details:")
for detail in explanation['clinical_details']:
    print(f"- {detail}")
print("\nEnvironmental Impact on Diabetes Risk:")
for detail in explanation['environmental_details']:
    print(f"- {detail}")
print("\nSuggestions:")
for suggestion in explanation['suggestions']:
    print(f"- {suggestion}")
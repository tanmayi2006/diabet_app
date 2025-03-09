import 'package:tflite/tflite.dart';

void main() async {
  try {
    // Sample clinical data
    final clinicalData = {
      'Age': 55,
      'Gender': 1, // Male
      'Polyuria': 1,
      'Polydipsia': 1,
      'Sudden Weight Loss': 0,
      'Weakness': 0,
      'Polyphagia': 1,
      'Genital Thrush': 0,
      'Visual Blurring': 0,
      'Itching': 0,
      'Irritability': 0,
      'Delayed Healing': 0,
      'Partial Paresis': 0,
      'Muscle Stiffness': 0,
      'Alopecia': 0,
      'Obesity': 0,
      'symptom_index': 3, // Sum of Polyuria, Polydipsia, Polyphagia
      'risk_factor': 2,   // Example calculation
    };

    // Load model and run inference
    await loadModelAndRun(clinicalData);
  } catch (e) {
    print('Error: $e');
  } finally {
    Tflite.close(); // Clean up
  }
}

Future<void> loadModelAndRun(Map<String, dynamic> clinicalData) async {
  // Load the TensorFlow Lite model
  String? loadResult = await Tflite.loadModel(
    model: 'diabetes_model.tflite', // File should be in the same directory
    isAsset: false, // Not using Flutter assets
  );
  print('Model load result: $loadResult');

  if (loadResult != 'success') {
    throw Exception('Failed to load model');
  }

  // Normalize input features
  final normalizedFeatures = normalizeFeatures(clinicalData);
  print('Input shape: [1, ${normalizedFeatures.length}]');
  print('Normalized input data: $normalizedFeatures');

  // Prepare input for the model (batch size of 1)
  final input = [normalizedFeatures];

  // Run inference
  final output = await Tflite.runModel(inputs: input);

  if (output == null || output.isEmpty) {
    throw Exception('No output from model');
  }

  // Print results
  print('Raw output: $output');
  print('Output shape: [${output.length}, ${output[0].length}]');
  print('Prediction probability: ${output[0][0]}');
  print('Diabetes risk: ${(output[0][0] * 100).toStringAsFixed(1)}%');
}

List<double> normalizeFeatures(Map<String, dynamic> data) {
  print('Raw input data: $data');
  final features = <double>[
    ((data['Age'] as num) - 30.0) / 50.0, // Normalize age (30-80 range)
    (data['Gender'] as num).toDouble(),
    (data['Polyuria'] as num).toDouble(),
    (data['Polydipsia'] as num).toDouble(),
    (data['Sudden Weight Loss'] as num).toDouble(),
    (data['Weakness'] as num).toDouble(),
    (data['Polyphagia'] as num).toDouble(),
    (data['Genital Thrush'] as num).toDouble(),
    (data['Visual Blurring'] as num).toDouble(),
    (data['Itching'] as num).toDouble(),
    (data['Irritability'] as num).toDouble(),
    (data['Delayed Healing'] as num).toDouble(),
    (data['Partial Paresis'] as num).toDouble(),
    (data['Muscle Stiffness'] as num).toDouble(),
    (data['Alopecia'] as num).toDouble(),
    (data['Obesity'] as num).toDouble(),
    (data['symptom_index'] as num).toDouble() / 3.0, // Normalize symptom index (0-3)
    (data['risk_factor'] as num).toDouble() / 4.0,   // Normalize risk factor (0-4)
  ];
  if (features.length != 18) {
    throw Exception('Feature length mismatch: expected 18, got ${features.length}');
  }
  return features;
}
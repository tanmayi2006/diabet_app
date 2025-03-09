import 'dart:typed_data';
import 'package:tflite_flutter/tflite_flutter.dart' as tfl;

class MlService {
  tfl.Interpreter? _interpreter;
  double? _yProba;
  Map<String, double>? _envData;
  Map<String, dynamic>? _explanation;

  Future<void> loadModelAndRun(Map<String, dynamic> clinicalData,
      {double lat = 37.7749, double lon = -122.4194}) async {
    try {
      await _loadModel();
      // Check if sufficient data is provided
      if (_hasSufficientData(clinicalData)) {
        await _runInference(clinicalData);
        await _fetchEnvironmentalData(lat, lon);
      } else {
        _yProba = null; // Set prediction to null if data is insufficient
      }
      _generateExplanation(clinicalData);
    } catch (e) {
      print("Error in loadModelAndRun: $e");
      throw e;
    }
  }

  // Helper method to check if sufficient data is provided
  bool _hasSufficientData(Map<String, dynamic> clinicalData) {
    // List of critical symptomatic features (beyond Age and Gender)
    final criticalFeatures = [
      'Polyuria',
      'Polydipsia',
      'Sudden Weight Loss',
      'Weakness',
      'Polyphagia',
      'Genital Thrush',
      'Visual Blurring',
      'Itching',
      'Irritability',
      'Delayed Healing',
      'Partial Paresis',
      'Muscle Stiffness',
      'Alopecia',
      'Obesity',
      'symptom_index',
      'risk_factor'
    ];

    // Check if any critical feature is present and non-zero
    return criticalFeatures.any((feature) => 
        clinicalData.containsKey(feature) && (clinicalData[feature] as num?)?.toDouble() != null && clinicalData[feature] != 0);
  }

  Future<void> _loadModel() async {
    // ... (unchanged)
    try {
      _interpreter = await tfl.Interpreter.fromAsset('diabetes_model.tflite');
      print("Model loaded successfully");

      try {
        final inputTensors = _interpreter!.getInputTensors();
        final outputTensors = _interpreter!.getOutputTensors();
        print("Input tensors count: ${inputTensors.length}");
        print("Output tensors count: ${outputTensors.length}");
        if (inputTensors.isNotEmpty && outputTensors.isNotEmpty) {
          print("Input shape: ${inputTensors[0].shape}");
          print("Output shape: ${outputTensors[0].shape}");
        }
      } catch (tensorError) {
        print("Non-critical error accessing tensor info: $tensorError");
      }
    } catch (e) {
      print("Error loading model: $e");
      throw Exception("Failed to load TensorFlow Lite model: $e");
    }
  }

  Future<void> _runInference(Map<String, dynamic> clinicalData) async {
    // ... (unchanged)
    if (_interpreter == null) {
      print("Model not loaded yet");
      return;
    }

    try {
      final normalizedFeatures = _normalizeFeatures(clinicalData);
      final List<double> paddedFeatures = normalizedFeatures.length >= 18
          ? normalizedFeatures.sublist(0, 18)
          : normalizedFeatures + List.filled(18 - normalizedFeatures.length, 0.0);

      List<List<double>> input = [paddedFeatures];
      var output = [<double>[0.0]];

      try {
        final inputTensors = _interpreter!.getInputTensors();
        final outputTensors = _interpreter!.getOutputTensors();
        if (inputTensors.isNotEmpty && outputTensors.isNotEmpty) {
          final outputShape = outputTensors[0].shape;
          if (outputShape.length >= 2) {
            output = List.generate(
              outputShape[0],
              (_) => List<double>.filled(outputShape[1], 0.0),
              growable: false,
            );
          }
        }
        _interpreter!.run(input, output);
        print("Raw output: $output");

        _yProba = output[0][0];
        print("Prediction probability: $_yProba");
      } catch (runError) {
        print("Error running model: $runError");
        final flatInput = Float32List.fromList(paddedFeatures);
        final flatOutput = Float32List(1);
        _interpreter!.run(flatInput.buffer.asUint8List(), flatOutput.buffer.asUint8List());
        _yProba = flatOutput[0];
        print("Alternative prediction method succeeded: $_yProba");
      }
    } catch (e) {
      print("Prediction error: $e");
      throw e;
    }
  }

  List<double> _normalizeFeatures(Map<String, dynamic> data) {
    // ... (unchanged)
    print("Input data: $data");

    double safeToDouble(dynamic value) {
      if (value == null) return 0.0;
      if (value is num) return value.toDouble();
      if (value is String) return double.tryParse(value) ?? 0.0;
      return 0.0;
    }

    final features = <double>[
      (safeToDouble(data['Age']) - 30.0) / 50.0,
      safeToDouble(data['Gender']),
      safeToDouble(data['Polyuria']),
      safeToDouble(data['Polydipsia']),
      safeToDouble(data['Sudden Weight Loss']),
      safeToDouble(data['Weakness']),
      safeToDouble(data['Polyphagia']),
      safeToDouble(data['Genital Thrush']),
      safeToDouble(data['Visual Blurring']),
      safeToDouble(data['Itching']),
      safeToDouble(data['Irritability']),
      safeToDouble(data['Delayed Healing']),
      safeToDouble(data['Partial Paresis']),
      safeToDouble(data['Muscle Stiffness']),
      safeToDouble(data['Alopecia']),
      safeToDouble(data['Obesity']),
      safeToDouble(data['symptom_index']) / 3.0,
      safeToDouble(data['risk_factor']) / 4.0,
    ];

    print("Normalized features: $features");
    return features;
  }

  Future<void> _fetchEnvironmentalData(double lat, double lon) async {
    // ... (unchanged)
    _envData = {
      'no2': 0.00012,
      'lst': 28.5,
      'aerosol_index': 0.8,
      'ndvi': 0.25,
    };
    print("Environmental Data: $_envData");
  }

  void _generateExplanation(Map<String, dynamic> clinicalData) {
    _explanation = {
      "risk_level": "Unknown",
      "clinical_details": <String>[],
      "environmental_details": <String>[],
      "suggestions": <String>[],
      "summary": "",
    };

    if (_yProba == null) {
      _explanation!['summary'] = "Insufficient data provided. Please include more clinical symptoms (e.g., Polyuria, Polydipsia) for an accurate risk assessment.";
      if (clinicalData.containsKey('Age')) {
        final age = clinicalData['Age'];
        final status = age > 50 ? 'high' : 'normal';
        _explanation!['clinical_details']!.add("Age: ${clinicalExplanations['Age']![status]}");
        if (status == 'high') _explanation!['suggestions']!.add(clinicalExplanations['Age']!['suggestion']);
      }
      if (clinicalData.containsKey('Gender')) {
        final gender = clinicalData['Gender'];
        final status = gender == 1 ? 'high' : 'normal'; // Assuming 1 is male, adjust as needed
        _explanation!['clinical_details']!.add("Gender: ${clinicalExplanations['Gender']![status]}");
        if (status == 'high') _explanation!['suggestions']!.add(clinicalExplanations['Gender']!['suggestion']);
      }
      return;
    }

    final riskLevel = _assignRisk(_yProba!);
    final topFeatures = _getTopFeatures(clinicalData);

    _explanation!['risk_level'] = riskLevel;

    for (var feature in topFeatures) {
      final value = clinicalData[feature];
      final status = (feature == 'Age' && value > 50) || (feature != 'Age' && value > 0) ? 'high' : 'normal';
      _explanation!['clinical_details']!.add("$feature: ${clinicalExplanations[feature]![status]}");
      if (status == 'high') _explanation!['suggestions']!.add(clinicalExplanations[feature]!['suggestion']);
    }

    String? no2Status;
    String? lstStatus;
    String? aerosolStatus;
    String? ndviStatus;

    if (_envData != null) {
      no2Status = _envData!['no2']! > 0.0001 ? 'high' : 'normal';
      _explanation!['environmental_details']!.add(
          "NO2: ${clinicalExplanations['NO2']![no2Status]} (Value: ${_envData!['no2']!.toStringAsFixed(4)} mol/m²)");
      if (no2Status == 'high') _explanation!['suggestions']!.add(clinicalExplanations['NO2']!['suggestion']);

      lstStatus = _envData!['lst']! > 30 ? 'high' : 'normal';
      _explanation!['environmental_details']!.add(
          "Heat: ${clinicalExplanations['Heat']![lstStatus]} (Temperature: ${_envData!['lst']!.toStringAsFixed(2)} °C)");
      if (lstStatus == 'high') _explanation!['suggestions']!.add(clinicalExplanations['Heat']!['suggestion']);

      aerosolStatus = _envData!['aerosol_index']! > 1 ? 'high' : 'normal';
      _explanation!['environmental_details']!.add(
          "Air Pollution: ${clinicalExplanations['AirPollution']![aerosolStatus]} (Aerosol Index: ${_envData!['aerosol_index']!.toStringAsFixed(2)})");
      if (aerosolStatus == 'high') _explanation!['suggestions']!.add(clinicalExplanations['AirPollution']!['suggestion']);

      ndviStatus = _envData!['ndvi']! < 0.2 ? 'normal' : 'high';
      _explanation!['environmental_details']!.add(
          "Green Space: ${clinicalExplanations['GreenSpace']![ndviStatus]} (NDVI: ${_envData!['ndvi']!.toStringAsFixed(2)})");
      if (ndviStatus == 'normal') _explanation!['suggestions']!.add(clinicalExplanations['GreenSpace']!['suggestion']);
    }

    _explanation!['summary'] = "Your diabetes risk is ${riskLevel.toLowerCase()} (${(_yProba! * 100).toStringAsFixed(0)}%). ";
    final keyFactors = topFeatures.where((f) => clinicalData[f] > 0 || (f == 'Age' && clinicalData[f] > 50)).toList();
    if (riskLevel == 'High') {
      if (keyFactors.isNotEmpty) _explanation!['summary'] += "Key clinical factors include: ${keyFactors.join(', ')}. ";
      if (_envData != null && (no2Status == 'high' || lstStatus == 'high' || aerosolStatus == 'high')) {
        _explanation!['summary'] += "Environmental stressors like poor air quality or heat may also be contributing.";
      }
    } else if (riskLevel == 'Moderate') {
      _explanation!['summary'] += "Monitor symptoms and environmental conditions closely for potential changes.";
    } else {
      _explanation!['summary'] += "Your risk is low—keep up your healthy habits!";
    }
  }

  String _assignRisk(double probability) {
    // ... (unchanged)
    if (probability >= 0.7) return "High";
    if (probability >= 0.4) return "Moderate";
    return "Low";
  }

  List<String> _getTopFeatures(Map<String, dynamic> data) {
    // ... (unchanged)
    final features = data.entries.map((e) => MapEntry(e.key, (e.value as num?)?.toDouble() ?? 0.0)).toList();
    features.sort((a, b) => b.value.compareTo(a.value));
    return features.take(5).map((e) => e.key).toList();
  }

  double? getPrediction() => _yProba;
  Map<String, dynamic>? getExplanation() => _explanation;
  void dispose() {
    _interpreter?.close();
  }
}

const clinicalExplanations = {
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
  "Sudden Weight Loss": {
    "high": "Unexplained weight loss may indicate uncontrolled diabetes.",
    "normal": "Weight is stable.",
    "suggestion": "Consult a doctor for potential metabolic concerns."
  },
  "Weakness": {
    "high": "Weakness can be a symptom of high or low blood sugar.",
    "normal": "No unusual fatigue or weakness reported.",
    "suggestion": "Monitor energy levels and consider a balanced diet."
  },
  "Polyphagia": {
    "high": "Excessive hunger may indicate blood sugar fluctuations.",
    "normal": "Appetite is within a normal range.",
    "suggestion": "Monitor food intake and check glucose levels."
  },
  "Genital Thrush": {
    "high": "Frequent fungal infections can be associated with diabetes.",
    "normal": "No infections reported.",
    "suggestion": "Maintain good hygiene and monitor blood sugar levels."
  },
  "Visual Blurring": {
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
  "Delayed Healing": {
    "high": "Slow healing of wounds is a common sign of diabetes.",
    "normal": "Normal healing observed.",
    "suggestion": "Monitor for infections and maintain good wound care."
  },
  "Partial Paresis": {
    "high": "Weakness or partial paralysis can be a neurological complication.",
    "normal": "No muscle weakness reported.",
    "suggestion": "Consider neurological evaluation if persistent."
  },
  "Muscle Stiffness": {
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
  },
};
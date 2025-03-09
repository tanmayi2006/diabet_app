import 'package:flutter/material.dart';
import 'diabetes_prediction_page.dart';

class ClinicalDataPage extends StatefulWidget {
  @override
  _ClinicalDataPageState createState() => _ClinicalDataPageState();
}

class _ClinicalDataPageState extends State<ClinicalDataPage> {
  int? age;
  String? gender;
  Map<String, bool> symptoms = {
    "Polyuria": false,
    "Polydipsia": false,
    "Sudden Weight Loss": false,
    "Weakness": false,
    "Polyphagia": false,
    "Genital Thrush": false,
    "Visual Blurring": false,
    "Itching": false,
    "Irritability": false,
    "Delayed Healing": false,
    "Partial Paresis": false,
    "Muscle Stiffness": false,
    "Alopecia": false,
    "Obesity": false,
  };

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text("Clinical Data"),
        backgroundColor: Colors.deepPurpleAccent,
      ),
      body: Container(
        padding: EdgeInsets.all(16),
        color: Colors.white,
        child: SingleChildScrollView(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text("Enter Age:", style: _labelStyle()),
              TextField(
                keyboardType: TextInputType.number,
                style: TextStyle(color: Colors.black),
                decoration: _inputDecoration("Age"),
                onChanged: (value) {
                  setState(() {
                    age = int.tryParse(value);
                  });
                },
              ),
              SizedBox(height: 16),
              Text("Select Gender:", style: _labelStyle()),
              Row(
                children: [
                  _genderRadio("Male"),
                  SizedBox(width: 20),
                  _genderRadio("Female"),
                ],
              ),
              SizedBox(height: 16),
              Text("Select Symptoms:", style: _labelStyle()),
              Column(
                children: symptoms.keys.map((String key) {
                  return CheckboxListTile(
                    title: Row(
                      children: [
                        _getSymptomIcon(key),
                        SizedBox(width: 8),
                        Text(key, style: TextStyle(color: Colors.black)),
                      ],
                    ),
                    value: symptoms[key],
                    onChanged: (bool? value) {
                      setState(() {
                        symptoms[key] = value!;
                      });
                    },
                    activeColor: Colors.greenAccent,
                  );
                }).toList(),
              ),
              SizedBox(height: 20),
              Center(
                child: ElevatedButton(
                  onPressed: () {
                    if (age == null || gender == null) {
                      ScaffoldMessenger.of(context).showSnackBar(
                        SnackBar(content: Text('Please enter age and select gender')),
                      );
                      return;
                    }
                    // Prepare data to pass
                    Map<String, dynamic> clinicalData = {
                      'Age': age!,
                      'Gender': gender == 'Male' ? 1 : 0,
                      ...symptoms.map((key, value) => MapEntry(key, value ? 1 : 0)),
                      'symptom_index': (symptoms['Polyuria']! ? 1 : 0) +
                          (symptoms['Polydipsia']! ? 1 : 0) +
                          (symptoms['Polyphagia']! ? 1 : 0),
                      'risk_factor': (symptoms['Polyuria']! ? 1 : 0) +
                          (symptoms['Polydipsia']! ? 1 : 0) +
                          (symptoms['Obesity']! ? 1 : 0) +
                          (age! > 50 ? 1 : 0),
                    };
                    Navigator.push(
                      context,
                      MaterialPageRoute(
                        builder: (context) => DiabetesPredictionPage(clinicalData: clinicalData),
                      ),
                    );
                  },
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.greenAccent,
                    padding: EdgeInsets.symmetric(horizontal: 40, vertical: 15),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12),
                    ),
                  ),
                  child: Text("Submit", style: TextStyle(fontSize: 18, color: Colors.black)),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _genderRadio(String value) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Radio(
          value: value,
          groupValue: gender,
          onChanged: (String? newValue) {
            setState(() {
              gender = newValue;
            });
          },
          activeColor: Colors.greenAccent,
        ),
        Text(value, style: TextStyle(color: Colors.black)),
      ],
    );
  }

  InputDecoration _inputDecoration(String hint) {
    return InputDecoration(
      hintText: hint,
      hintStyle: TextStyle(color: Colors.grey),
      enabledBorder: OutlineInputBorder(
        borderSide: BorderSide(color: Colors.black),
        borderRadius: BorderRadius.circular(12),
      ),
      focusedBorder: OutlineInputBorder(
        borderSide: BorderSide(color: Colors.greenAccent, width: 2),
        borderRadius: BorderRadius.circular(12),
      ),
      filled: true,
      fillColor: Colors.white,
    );
  }

  TextStyle _labelStyle() {
    return TextStyle(
      color: Colors.black,
      fontSize: 18,
      fontWeight: FontWeight.bold,
    );
  }

  Widget _getSymptomIcon(String symptom) {
    switch (symptom) {
      case "Polyuria":
        return Icon(Icons.water_drop, color: Colors.blueAccent);
      case "Polydipsia":
        return Icon(Icons.local_drink, color: Colors.blueAccent);
      case "Sudden Weight Loss":
        return Icon(Icons.scale, color: Colors.orangeAccent);
      case "Weakness":
        return Icon(Icons.battery_alert, color: Colors.redAccent);
      case "Polyphagia":
        return Icon(Icons.fastfood, color: Colors.greenAccent);
      case "Genital Thrush":
        return Icon(Icons.medical_services, color: Colors.purpleAccent);
      case "Visual Blurring":
        return Icon(Icons.visibility_off, color: Colors.grey);
      case "Itching":
        return Icon(Icons.pest_control, color: Colors.greenAccent);
      case "Irritability":
        return Icon(Icons.sentiment_very_dissatisfied, color: Colors.yellowAccent);
      case "Delayed Healing":
        return Icon(Icons.healing, color: Colors.teal);
      case "Partial Paresis":
        return Icon(Icons.accessibility, color: Colors.blueAccent);
      case "Muscle Stiffness":
        return Icon(Icons.sports_handball, color: Colors.purpleAccent);
      case "Alopecia":
        return Icon(Icons.face, color: Colors.black);
      case "Obesity":
        return Icon(Icons.fitness_center, color: Colors.redAccent);
      default:
        return Icon(Icons.help, color: Colors.grey);
    }
  }
}

void main() {
  runApp(MaterialApp(
    debugShowCheckedModeBanner: false,
    home: ClinicalDataPage(),
  ));
}
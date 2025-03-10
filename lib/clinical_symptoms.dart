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
                  return GestureDetector(
                    onTap: () {
                      setState(() {
                        symptoms[key] = !symptoms[key]!;
                      });
                    },
                    child: Container(
                      margin: EdgeInsets.symmetric(vertical: 5),
                      decoration: BoxDecoration(
                        border: Border.all(
                          color: symptoms[key]! ? Colors.green : Colors.grey,
                          width: 2,
                        ),
                        borderRadius: BorderRadius.circular(10),
                      ),
                      child: CheckboxListTile(
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
                      ),
                    ),
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
        return Image.asset(
          'urine.jpg',
          width: 80,
          height: 80,
          fit: BoxFit.contain,
        );
      case "Polydipsia":
        return Image.asset(
          'thirst.jpg',
          width: 80,
          height: 80,
          fit: BoxFit.contain,
        );
      case "Sudden Weight Loss":
        return Image.asset(
          'weightloss.jpg',
          width: 65,
          height: 65,
          fit: BoxFit.contain,
        );
      case "Weakness":
        return Image.asset(
          'weakness.jpg',
          width: 80,
          height: 80,
          fit: BoxFit.contain,
        );
      case "Polyphagia":
        return Image.asset(
          'hungry.jpg',
          width: 80,
          height: 80,
          fit: BoxFit.contain,
        );
      case "Genital Thrush":
        return Image.asset(
          'genital.jpg',
          width: 70,
          height: 70,
          fit: BoxFit.contain,
        );
      case "Visual Blurring":
        return Image.asset(
          'visualblur.jpg',
          width: 80,
          height: 80,
          fit: BoxFit.contain,
        );
      case "Itching":
        return Image.asset(
          'itching.jpg',
          width: 80,
          height: 80,
          fit: BoxFit.contain,
        );
      case "Irritability":
        return Image.asset(
          'irritate.jpg',
          width: 60,
          height: 60,
          fit: BoxFit.contain,
        );
      case "Delayed Healing":
        return Image.asset(
          'delayed_healing.jpg',
          width: 80,
          height: 80,
          fit: BoxFit.contain,
        );
      case "Partial Paresis":
        return Image.asset(
          'partialparesis.jpg',
          width: 80,
          height: 80,
          fit: BoxFit.contain,
        );
      case "Muscle Stiffness":
        return Image.asset(
          'muscle_stiffness.jpg',
          width: 80,
          height: 80,
          fit: BoxFit.contain,
        );
      case "Alopecia":
        return Image.asset(
          'hairloss.jpg',
          width: 80,
          height: 80,
          fit: BoxFit.contain,
        );
      case "Obesity":
        return Image.asset(
          'obesity.jpg',
          width: 80,
          height: 80,
          fit: BoxFit.contain,
        );
      default:
        return Icon(
        Icons.help,
        color: Colors.grey.withOpacity(0.05),  // Adjust opacity here
);

    }
  }
}

void main() {
  runApp(MaterialApp(
    debugShowCheckedModeBanner: false,
    home: ClinicalDataPage(),
  ));
}

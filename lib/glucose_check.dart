import 'package:flutter/material.dart';

class GlucoseCheckPage extends StatefulWidget {
  @override
  _GlucoseCheckPageState createState() => _GlucoseCheckPageState();
}

class _GlucoseCheckPageState extends State<GlucoseCheckPage> {
  final TextEditingController _glucoseController = TextEditingController();
  bool _smoking = false;
  bool _alcohol = false;
  bool _stress = false;
  bool _poorDiet = false;
  String _result = "";
  String _suggestion = "";

  void _checkGlucose() {
    double glucose = double.tryParse(_glucoseController.text) ?? 0;
    
    if (glucose == 0) {
      setState(() {
        _result = "Please enter a valid glucose level.";
        _suggestion = "";
      });
      return;
    }

    if (glucose < 70) {
      _result = "Low Glucose Level";
      _suggestion = "Eat a balanced meal and monitor your sugar intake.";
    } else if (glucose > 140) {
      _result = "High Glucose Level";
      _suggestion = "Reduce sugar intake, exercise, and consult a doctor.";
    } else {
      _result = "Normal Glucose Level";
      _suggestion = "Maintain a healthy diet and lifestyle.";
    }

    if (_smoking || _alcohol || _stress || _poorDiet) {
      _suggestion += "\nReduce bad habits for better health.";
    }

    setState(() {});
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text("Check Glucose Level")),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            TextField(
              controller: _glucoseController,
              keyboardType: TextInputType.number,
              decoration: InputDecoration(
                labelText: "Enter Glucose Level",
                border: OutlineInputBorder(),
              ),
            ),
            SizedBox(height: 20),
            Text("Select Habits:"),
            CheckboxListTile(
              title: Text("Smoking"),
              value: _smoking,
              onChanged: (val) => setState(() => _smoking = val!),
            ),
            CheckboxListTile(
              title: Text("Excessive Alcohol"),
              value: _alcohol,
              onChanged: (val) => setState(() => _alcohol = val!),
            ),
            CheckboxListTile(
              title: Text("High Stress"),
              value: _stress,
              onChanged: (val) => setState(() => _stress = val!),
            ),
            CheckboxListTile(
              title: Text("Poor Diet"),
              value: _poorDiet,
              onChanged: (val) => setState(() => _poorDiet = val!),
            ),
            SizedBox(height: 20),
            ElevatedButton(
              onPressed: _checkGlucose,
              child: Text("Check Glucose"),
            ),
            SizedBox(height: 20),
            if (_result.isNotEmpty)
              Text(
                _result,
                style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: Colors.red),
              ),
            if (_suggestion.isNotEmpty)
              Text(
                _suggestion,
                style: TextStyle(fontSize: 16, color: Colors.blue),
              ),
          ],
        ),
      ),
    );
  }
}

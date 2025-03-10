import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import 'qr_code.dart';
import 'clinical_symptoms.dart';

import 'services/backend_service.dart';
import 'user_details.dart';

class AddDetailsPage extends StatefulWidget {
  @override
  _AddDetailsPageState createState() => _AddDetailsPageState();
}

class _AddDetailsPageState extends State<AddDetailsPage> {
  final TextEditingController _nameController = TextEditingController();
  final TextEditingController _dobController = TextEditingController();
  final TextEditingController _aadharController = TextEditingController();
  final TextEditingController _locationController = TextEditingController();

  bool isMale = false;
  bool isFemale = false;

  final BackendService _backendService = BackendService();

  Future<void> _selectDate(BuildContext context) async {
    DateTime? pickedDate = await showDatePicker(
      context: context,
      initialDate: DateTime.now(),
      firstDate: DateTime(1900),
      lastDate: DateTime.now(),
    );
    if (pickedDate != null) {
      setState(() {
        _dobController.text = DateFormat('yyyy-MM-dd').format(pickedDate);
      });
    }
  }

  Future<void> _submitForm() async {
    if (_nameController.text.isEmpty ||
        _dobController.text.isEmpty ||
        _aadharController.text.isEmpty ||
        _locationController.text.isEmpty ||
        (!isMale && !isFemale)) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Please fill all fields')),
      );
      return;
    }

    if (_aadharController.text.length != 12) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Aadhar number must be 12 digits')),
      );
      return;
    }

    try {
      Map<String, dynamic> userData = {
        'name': _nameController.text,
        'date_of_birth': _dobController.text,
        'gender': isMale ? 'Male' : 'Female',
        'aadhaar_number': _aadharController.text,
        'location': _locationController.text,
      };

      Map<String, dynamic> savedUser = await _backendService.addUser(userData);

      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('User details saved successfully')),
      );

      _nameController.clear();
      _dobController.clear();
      _aadharController.clear();
      _locationController.clear();
      setState(() {
        isMale = false;
        isFemale = false;
      });

      Navigator.push(
        context,
        MaterialPageRoute(
          builder: (context) => QRCodePage(uniqueCode: savedUser['unique_code']),
        ),
      );
    } on ExistingUserException catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('User already exists')),
      );
      Navigator.push(
        context,
        MaterialPageRoute(
          builder: (context) => UserDetailsPage(user: e.existingUser),
        ),
      );
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error saving data: $e')),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Colors.deepPurpleAccent,
        title: Text('Add Details'),
        leading: IconButton(
          icon: Icon(Icons.arrow_back),
          onPressed: () => Navigator.pop(context),
        ),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(20.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            TextField(
              controller: _nameController,
              decoration: InputDecoration(
                labelText: 'Enter Name',
                labelStyle: TextStyle(color: Colors.black),
                filled: true,
                fillColor: Colors.white.withOpacity(0.7),
                border: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(10),
                ),
              ),
            ),
            SizedBox(height: 15),
            TextField(
              controller: _dobController,
              decoration: InputDecoration(
                labelText: 'Date of Birth',
                labelStyle: TextStyle(color: Colors.black),
                suffixIcon: IconButton(
                  icon: Icon(Icons.calendar_today),
                  onPressed: () => _selectDate(context),
                  color: Colors.black,
                ),
                filled: true,
                fillColor: Colors.white.withOpacity(0.7),
                border: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(10),
                ),
              ),
              readOnly: true,
            ),
            SizedBox(height: 15),
            Row(
              children: [
                Text('Gender: ', style: TextStyle(color: Colors.black)),
                Checkbox(
                  value: isMale,
                  onChanged: (value) {
                    setState(() {
                      isMale = value!;
                      isFemale = !value;
                    });
                  },
                ),
                Text('Male', style: TextStyle(color: Colors.black)),
                Checkbox(
                  value: isFemale,
                  onChanged: (value) {
                    setState(() {
                      isFemale = value!;
                      isMale = !value;
                    });
                  },
                ),
                Text('Female', style: TextStyle(color: Colors.black)),
              ],
            ),
            SizedBox(height: 15),
            TextField(
              controller: _aadharController,
              decoration: InputDecoration(
                labelText: 'Aadhar Number',
                labelStyle: TextStyle(color: Colors.black),
                filled: true,
                fillColor: Colors.white.withOpacity(0.7),
                border: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(10),
                ),
              ),
              keyboardType: TextInputType.number,
              maxLength: 12,
            ),
            SizedBox(height: 15),
            TextField(
              controller: _locationController,
              decoration: InputDecoration(
                labelText: 'Location',
                labelStyle: TextStyle(color: Colors.black),
                filled: true,
                fillColor: Colors.white.withOpacity(0.7),
                border: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(10),
                ),
              ),
            ),
            SizedBox(height: 20),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                _buildActionButton('Reset', Colors.red, () {
                  _nameController.clear();
                  _dobController.clear();
                  _aadharController.clear();
                  _locationController.clear();
                  setState(() {
                    isMale = false;
                    isFemale = false;
                  });
                }),
                _buildActionButton('Submit', Colors.green, _submitForm),
              ],
            ),
            SizedBox(height: 15),
            Center(
              child: _buildActionButton('Enter Clinical Symptoms', Colors.deepPurple, () {
                Navigator.push(
  context,
  MaterialPageRoute(builder: (context) => ClinicalDataPage()),
);
              }),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildActionButton(String text, Color color, VoidCallback onTap) {
    return ElevatedButton(
      onPressed: onTap,
      style: ElevatedButton.styleFrom(
        backgroundColor: color,
        foregroundColor: Colors.white,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(15),
        ),
        minimumSize: Size(150, 50),
        elevation: 5,
      ),
      child: Text(
        text,
        style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold, color: Colors.white),
      ),
    );
  }
}
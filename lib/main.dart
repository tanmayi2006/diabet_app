import 'package:flutter/material.dart';
import 'aadhar_details.dart';
import 'add_details.dart';
import 'qr_code.dart';
import 'qr_scan.dart';
import 'clinical_symptoms.dart';
import 'profile.dart';

void main() => runApp(DiabetesApp());

class DiabetesApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      home: HomePage(),
    );
  }
}

class HomePage extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: Container(
          decoration: BoxDecoration(
            gradient: LinearGradient(
              colors: [Colors.blue.shade900, Colors.purple.shade700],
              begin: Alignment.topCenter,
              end: Alignment.bottomCenter,
            ),
          ),
          child: Column(
            children: [
              SizedBox(height: 20),
              Stack(
                children: [
                  Align(
                    alignment: Alignment.topCenter,
                    child: Text(
                      'Diabetes Prediction',
                      style: TextStyle(
                        color: Colors.white,
                        fontSize: 32,
                        fontWeight: FontWeight.bold,
                      ),
                      textAlign: TextAlign.center,
                    ),
                  ),
                  Align(
                    alignment: Alignment.topRight,
                    child: Padding(
                      padding: const EdgeInsets.only(right: 20.0),
                      child: IconButton(
                        icon: Icon(Icons.person, color: Colors.white, size: 30),
                        onPressed: () {
                          Navigator.push(
                            context,
                            MaterialPageRoute(builder: (context) => ProfilePage()),
                          );
                        },
                      ),
                    ),
                  ),
                ],
              ),
              Spacer(),
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 20.0),
                child: GridView.count(
                  shrinkWrap: true,
                  crossAxisCount: MediaQuery.of(context).size.width < 600 ? 2 : 3,
                  crossAxisSpacing: 15,
                  mainAxisSpacing: 15,
                  children: [
                    CustomButton(
                      text: 'Add Details',
                      icon: Icons.add,
                      onTap: () => Navigator.push(
                        context,
                        MaterialPageRoute(builder: (context) => AddDetailsPage()),
                      ),
                    ),
                    CustomButton(
                      text: 'Risk Checking',
                      icon: Icons.health_and_safety,
                      onTap: () => Navigator.push(
                        context,
                        MaterialPageRoute(builder: (context) => ClinicalDataPage()),
                      ),
                    ),
                    CustomButton(
                      text: 'View Users',
                      icon: Icons.people,
                      onTap: () => Navigator.push(
                        context,
                        MaterialPageRoute(builder: (context) => AadharSearchPage()),
                      ),
                    ),
                    CustomButton(
                      text: 'Scan QR',
                      icon: Icons.qr_code_scanner,
                      onTap: () => Navigator.push(
                        context,
                        MaterialPageRoute(builder: (context) => QrScanPage()),
                      ),
                    ),
                  ],
                ),
              ),
              Spacer(),
            ],
          ),
        ),
      ),
    );
  }
}

class CustomButton extends StatelessWidget {
  final String text;
  final IconData icon;
  final VoidCallback onTap;

  CustomButton({required this.text, required this.icon, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return ElevatedButton(
      onPressed: onTap,
      style: ElevatedButton.styleFrom(
        backgroundColor: Colors.white,
        foregroundColor: Colors.black,
        minimumSize: Size(double.infinity, 100),
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(12),
        ),
        elevation: 5,
        shadowColor: Colors.black.withOpacity(0.2),
        padding: EdgeInsets.symmetric(vertical: 20),
      ),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(icon, size: 60, color: Colors.black),
          SizedBox(height: 10),
          Text(
            text,
            style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            textAlign: TextAlign.center,
          ),
        ],
      ),
    );
  }
}
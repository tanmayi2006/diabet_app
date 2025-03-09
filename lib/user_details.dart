import 'package:flutter/material.dart';
import 'package:qr_flutter/qr_flutter.dart'; // Import qr_flutter for QR code generation

class UserDetailsPage extends StatelessWidget {
  final Map<String, dynamic> user;

  const UserDetailsPage({Key? key, required this.user}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Colors.deepPurple.shade800,
        title: Text('User Details'),
        leading: IconButton(
          icon: Icon(Icons.arrow_back),
          onPressed: () => Navigator.pop(context),
        ),
      ),
      body: Container(
        padding: EdgeInsets.all(20.0),
        decoration: BoxDecoration(
          gradient: LinearGradient(
            colors: [Colors.purple.shade700, Colors.blue.shade900],
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
          ),
        ),
        child: SingleChildScrollView( // Added to handle overflow with QR code
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.center,
            children: [
              Text(
                "Aadhar Number: ${user['original_aadhaar_number'] as String? ?? 'N/A'}",
                style: TextStyle(fontSize: 24, color: Colors.white),
              ),
              SizedBox(height: 20),
              Text(
                "Name: ${user['name'] as String? ?? 'N/A'}",
                style: TextStyle(fontSize: 20, color: Colors.white),
              ),
              SizedBox(height: 10),
              Text(
                "Date of Birth: ${user['date_of_birth'] as String? ?? 'N/A'}",
                style: TextStyle(fontSize: 20, color: Colors.white),
              ),
              SizedBox(height: 10),
              Text(
                "Gender: ${user['gender'] as String? ?? 'N/A'}",
                style: TextStyle(fontSize: 20, color: Colors.white),
              ),
              SizedBox(height: 10),
              Text(
                "Location: ${user['location'] as String? ?? 'N/A'}",
                style: TextStyle(fontSize: 20, color: Colors.white),
              ),
              SizedBox(height: 10),
              Text(
                "Age: ${user['age'] != null ? user['age'].toString() : 'N/A'}",
                style: TextStyle(fontSize: 20, color: Colors.white),
              ),
              SizedBox(height: 20),
              Text(
                "QR Code:",
                style: TextStyle(fontSize: 20, color: Colors.white, fontWeight: FontWeight.bold),
              ),
              SizedBox(height: 10),
              Container(
                padding: EdgeInsets.all(10),
                decoration: BoxDecoration(
                  color: Colors.white, // Background for QR code visibility
                  borderRadius: BorderRadius.circular(10),
                ),
                child: QrImageView(
                  data: user['qr_code'] as String? ?? 'N/A', // Use qr_code field
                  version: QrVersions.auto,
                  size: 200.0, // Size of the QR code
                  errorStateBuilder: (context, error) {
                    return Text(
                      "Error generating QR code: $error",
                      style: TextStyle(color: Colors.red),
                    );
                  },
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
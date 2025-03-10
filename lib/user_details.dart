import 'package:flutter/material.dart';
import 'package:qr_flutter/qr_flutter.dart'; // Import qr_flutter for QR code generation

class UserDetailsPage extends StatelessWidget {
  final Map<String, dynamic> user;

  const UserDetailsPage({Key? key, required this.user}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Colors.deepPurpleAccent,
        title: Text(
          'User Details',
          style: TextStyle(
            fontSize: 24,
            fontWeight: FontWeight.bold,
            letterSpacing: 1.5,
            color: Colors.white,
          ),
        ),
        leading: IconButton(
          icon: Icon(Icons.arrow_back, size: 28),
          onPressed: () => Navigator.pop(context),
        ),
        elevation: 10,
      ),
      body: Container(
        padding: EdgeInsets.all(20.0),
        decoration: BoxDecoration(
          gradient: LinearGradient(
            colors: [Colors.white, Colors.grey[100]!],
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
          ),
          borderRadius: BorderRadius.circular(20), // Rounded corners
          boxShadow: [
            BoxShadow(
              color: Colors.black.withOpacity(0.1),
              spreadRadius: 3,
              blurRadius: 10,
              offset: Offset(0, 5),
            ),
          ], // Subtle shadow effect
        ),
        child: ListView(
          children: [
            _buildUserInfo("Aadhar Number: ", user['original_aadhaar_number'] as String?),
            _buildUserInfo("Name: ", user['name'] as String?),
            _buildUserInfo("Date of Birth: ", user['date_of_birth'] as String?),
            _buildUserInfo("Gender: ", user['gender'] as String?),
            _buildUserInfo("Location: ", user['location'] as String?),
            _buildUserInfo("Age: ", user['age'] != null ? user['age'].toString() : null),
            SizedBox(height: 20),

            // QR Code section
            Text(
              "QR Code:",
              style: TextStyle(
                fontSize: 20,
                color: Colors.black,
                fontWeight: FontWeight.bold,
                letterSpacing: 1.2,
              ),
            ),
            SizedBox(height: 10),
            Container(
              padding: EdgeInsets.all(15),
              // decoration: BoxDecoration(
              //   color: Colors.deepPurpleAccent, // QR code container background
              //   borderRadius: BorderRadius.circular(15),
              //   boxShadow: [
              //     BoxShadow(
              //       color: Colors.black.withOpacity(0.1),
              //       spreadRadius: 2,
              //       blurRadius: 7,
              //       offset: Offset(0, 3),
              //     ),
              //   ],
              // ),
              child: QrImageView(
                data: user['qr_code'] as String? ?? 'N/A',
                version: QrVersions.auto,
                size: 220.0, // Slightly larger QR code for better visibility
                errorStateBuilder: (context, error) {
                  return Center(
                    child: Text(
                      "Error generating QR code: $error",
                      style: TextStyle(
                        color: Colors.white,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  );
                },
              ),
            ),
            SizedBox(height: 20),

            // Expanded widget to fill remaining space
            Expanded(child: SizedBox()),
          ],
        ),
      ),
    );
  }

  // Helper widget to build user info sections
  Widget _buildUserInfo(String label, String? value) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 16.0),
      child: Container(
        padding: EdgeInsets.symmetric(vertical: 12.0, horizontal: 16.0),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(12),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withOpacity(0.05),
              spreadRadius: 2,
              blurRadius: 5,
              offset: Offset(0, 3),
            ),
          ],
        ),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              label,
              style: TextStyle(
                fontSize: 18,
                color: Colors.black.withOpacity(0.8),
                fontWeight: FontWeight.w600,
              ),
            ),
            SizedBox(width: 8),
            Expanded(
              child: Text(
                value ?? 'N/A',
                style: TextStyle(
                  fontSize: 18,
                  color: Colors.black,
                  fontWeight: FontWeight.w500,
                ),
                overflow: TextOverflow.ellipsis, // To handle long text
              ),
            ),
          ],
        ),
      ),
    );
  }
}

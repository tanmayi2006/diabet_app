import 'package:flutter/material.dart';

class ProfilePage extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Profile'),
        backgroundColor: Colors.deepPurpleAccent,
      ),
      body: ListView(
        padding: EdgeInsets.all(20),
        children: [
          // Frontline Worker Section
          _buildSectionHeader('Frontline Worker Details'),
          _buildInfoRow('Name:', 'John Doe'),
          _buildInfoRow('Location:', 'New York City'),
          _buildInfoRow('No. of Patients in Guidance:', '150'),
          _buildInfoRow('No. of Patients with High Risk:', '30'),

          // Environmental Section
          _buildSectionHeader('Environmental Details'),
          _buildInfoRow('Air Pollution:', 'Moderate'),
          _buildInfoRow('Heat:', 'High'),
        ],
      ),
    );
  }

  Widget _buildSectionHeader(String title) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 10),
      child: Text(
        title,
        style: TextStyle(
          fontSize: 18,
          fontWeight: FontWeight.bold,
          color: Colors.deepPurpleAccent,
        ),
      ),
    );
  }

  Widget _buildInfoRow(String label, String value) {
    return Card(
      margin: EdgeInsets.symmetric(vertical: 8),
      elevation: 4,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
      child: ListTile(
        title: Text(
          label,
          style: TextStyle(fontWeight: FontWeight.bold),
        ),
        subtitle: Text(value),
      ),
    );
  }
}

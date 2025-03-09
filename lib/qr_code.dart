import 'package:flutter/material.dart';
import 'package:qr_flutter/qr_flutter.dart';
import '../services/backend_service.dart'; // Import BackendService

class QRCodePage extends StatelessWidget {
  final String? uniqueCode; // Accept uniqueCode from AddDetailsPage

  const QRCodePage({Key? key, this.uniqueCode}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    final BackendService backendService = BackendService(); // Instantiate if needed

    return Scaffold(
      appBar: AppBar(
        backgroundColor: Colors.lightBlue.shade400,
        title: Text('QR Code', style: TextStyle(color: Colors.white)),
        leading: IconButton(
          icon: Icon(Icons.arrow_back),
          onPressed: () => Navigator.pop(context),
        ),
      ),
      body: Container(
        color: Colors.lightBlue.shade50,
        padding: EdgeInsets.all(20),
        child: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              if (uniqueCode != null) ...[
                QrImageView(
                  data: uniqueCode!,
                  version: QrVersions.auto,
                  size: 200.0,
                  backgroundColor: Colors.white,
                  padding: EdgeInsets.all(10),
                ),
                SizedBox(height: 20),
                Text(
                  'Your Unique Code: $uniqueCode',
                  style: TextStyle(
                    fontSize: 16,
                    color: Colors.blue.shade600,
                    fontWeight: FontWeight.bold,
                  ),
                  textAlign: TextAlign.center,
                ),
              ] else ...[
                Icon(
                  Icons.qr_code_scanner,
                  size: 150,
                  color: Colors.blue.shade600,
                ),
                SizedBox(height: 20),
                Text(
                  'No QR code available. Please submit details first.',
                  style: TextStyle(
                    fontSize: 16,
                    color: Colors.red.shade600,
                  ),
                  textAlign: TextAlign.center,
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }
}
import 'package:flutter/material.dart';
import 'package:mobile_scanner/mobile_scanner.dart';
import 'user_details.dart';
import 'services/backend_service.dart';

class QrScanPage extends StatefulWidget {
  @override
  _QrScanPageState createState() => _QrScanPageState();
}

class _QrScanPageState extends State<QrScanPage> {
  String qrText = "Scan a QR Code";
  final BackendService _backendService = BackendService();

  void _validateAndRedirect(String scannedCode) async {
    try {
      Map<String, dynamic>? user = await _backendService.getUserByUniqueCode(scannedCode);
      if (user != null) {
        Navigator.push(
          context,
          MaterialPageRoute(
            builder: (context) => UserDetailsPage(user: user), // Pass full user data
          ),
        );
      } else {
        setState(() {
          qrText = "Invalid QR Code";
        });
      }
    } catch (e) {
      setState(() {
        qrText = "Error: $e";
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            colors: [Colors.purple.shade700, Colors.blue.shade900],
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
          ),
        ),
        child: Column(
          children: [
            SizedBox(height: 40),
            Row(
              children: [
                IconButton(
                  icon: Icon(Icons.arrow_back, color: Colors.white, size: 30),
                  onPressed: () {
                    Navigator.pop(context);
                  },
                ),
                Expanded(
                  child: Center(
                    child: Text(
                      'QR Scan',
                      style: TextStyle(
                        color: Colors.white,
                        fontSize: 30,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ),
                ),
              ],
            ),
            SizedBox(height: 20),
            Center(
              child: Container(
                width: 280,
                height: 280,
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.circular(20),
                  border: Border.all(color: Colors.white, width: 3),
                ),
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(20),
                  child: MobileScanner(
                    onDetect: (barcodeCapture) {
                      final List<Barcode> barcodes = barcodeCapture.barcodes;
                      if (barcodes.isNotEmpty && barcodes.first.rawValue != null) {
                        final String scannedCode = barcodes.first.rawValue!;
                        setState(() {
                          qrText = scannedCode;
                        });
                        _validateAndRedirect(scannedCode);
                      }
                    },
                  ),
                ),
              ),
            ),
            SizedBox(height: 20),
            Text(
              qrText,
              style: TextStyle(color: Colors.white, fontSize: 18),
              textAlign: TextAlign.center,
            ),
            SizedBox(height: 40),
          ],
        ),
      ),
    );
  }
}
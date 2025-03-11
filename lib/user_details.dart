import 'package:flutter/material.dart';
import 'package:qr_flutter/qr_flutter.dart';
import 'package:pie_chart/pie_chart.dart';
import 'services/database_helper.dart';
import 'dart:convert'; // For parsing explanation string if needed

class UserDetailsPage extends StatefulWidget {
  final Map<String, dynamic> user;

  const UserDetailsPage({Key? key, required this.user}) : super(key: key);

  @override
  _UserDetailsPageState createState() => _UserDetailsPageState();
}

class _UserDetailsPageState extends State<UserDetailsPage> {
  final DatabaseHelper _dbHelper = DatabaseHelper();
  List<Map<String, dynamic>> _reports = [];
  bool _isLoading = true;

  @override
  void initState() {
    super.initState();
    _loadReports();
  }

  Future<void> _loadReports() async {
    try {
      String aadhaar = widget.user['original_aadhaar_number'] as String;
      _reports = await _dbHelper.getReportsByAadhaar(aadhaar);
      setState(() => _isLoading = false);
    } catch (e) {
      print("Error loading reports: $e");
      setState(() => _isLoading = false);
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("Error loading reports: $e")),
      );
    }
  }

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
          borderRadius: BorderRadius.circular(20),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withOpacity(0.1),
              spreadRadius: 3,
              blurRadius: 10,
              offset: Offset(0, 5),
            ),
          ],
        ),
        child: _isLoading
            ? Center(child: CircularProgressIndicator())
            : ListView(
                children: [
                  _buildUserInfo("Aadhaar Number: ", widget.user['original_aadhaar_number'] as String?),
                  _buildUserInfo("Name: ", widget.user['name'] as String?),
                  _buildUserInfo("Date of Birth: ", widget.user['date_of_birth'] as String?),
                  _buildUserInfo("Gender: ", widget.user['gender'] as String?),
                  _buildUserInfo("Location: ", widget.user['location'] as String?),
                  _buildUserInfo("Age: ", widget.user['age'] != null ? widget.user['age'].toString() : null),
                  SizedBox(height: 20),
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
                    child: QrImageView(
                      data: widget.user['qr_code'] as String? ?? 'N/A',
                      version: QrVersions.auto,
                      size: 220.0,
                      errorStateBuilder: (context, error) {
                        return Center(
                          child: Text(
                            "Error generating QR code: $error",
                            style: TextStyle(
                              color: Colors.red,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                        );
                      },
                    ),
                  ),
                  SizedBox(height: 20),
                  Text(
                    "Diabetes Prediction Reports:",
                    style: TextStyle(
                      fontSize: 20,
                      color: Colors.black,
                      fontWeight: FontWeight.bold,
                      letterSpacing: 1.2,
                    ),
                  ),
                  SizedBox(height: 10),
                  _reports.isEmpty
                      ? Padding(
                          padding: const EdgeInsets.all(8.0),
                          child: Text(
                            "No reports available for this Aadhaar.",
                            style: TextStyle(
                              fontSize: 16,
                              color: Colors.grey,
                              fontStyle: FontStyle.italic,
                            ),
                          ),
                        )
                      : ListView.builder(
                          shrinkWrap: true,
                          physics: NeverScrollableScrollPhysics(),
                          itemCount: _reports.length,
                          itemBuilder: (context, index) {
                            final report = _reports[index];
                            return InkWell(
                              onTap: () {
                                Navigator.push(
                                  context,
                                  MaterialPageRoute(
                                    builder: (context) => ReportDetailsPage(report: report),
                                  ),
                                );
                              },
                              child: Card(
                                elevation: 2,
                                margin: EdgeInsets.symmetric(vertical: 8.0),
                                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                                child: Padding(
                                  padding: const EdgeInsets.all(12.0),
                                  child: Column(
                                    crossAxisAlignment: CrossAxisAlignment.start,
                                    children: [
                                      Text(
                                        "Timestamp: ${report['timestamp']}",
                                        style: TextStyle(
                                          fontSize: 16,
                                          fontWeight: FontWeight.bold,
                                          color: Colors.deepPurpleAccent,
                                        ),
                                      ),
                                      SizedBox(height: 8),
                                      Text(
                                        "Prediction: ${report['prediction'] != null ? '${(report['prediction'] * 100).toStringAsFixed(1)}%' : 'N/A'}",
                                        style: TextStyle(fontSize: 14, color: Colors.black87),
                                      ),
                                      SizedBox(height: 8),
                                      Text(
                                        "Explanation: ${report['explanation']}",
                                        style: TextStyle(fontSize: 14, color: Colors.black54),
                                        maxLines: 3,
                                        overflow: TextOverflow.ellipsis,
                                      ),
                                    ],
                                  ),
                                ),
                              ),
                            );
                          },
                        ),
                ],
              ),
      ),
    );
  }

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
                overflow: TextOverflow.ellipsis,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class ReportDetailsPage extends StatelessWidget {
  final Map<String, dynamic> report;

  const ReportDetailsPage({Key? key, required this.report}) : super(key: key);

  // Pie chart data preparation (copied from DiabetesPredictionPage)
  Map<String, double> _getPieChartData(double? probability) {
    if (probability == null) {
      return {"Risk": 0.0, "No Risk": 100.0};
    }
    return {
      "Risk": probability * 100,
      "No Risk": (1 - probability) * 100,
    };
  }

  // Parse the explanation string back into a Map
  Map<String, dynamic> _parseExplanation(String explanationString) {
    try {
      // Remove curly braces and split by commas to mimic a simple map parse
      String cleaned = explanationString.replaceAll('{', '').replaceAll('}', '');
      Map<String, dynamic> parsed = {};
      List<String> pairs = cleaned.split(', ');
      for (String pair in pairs) {
        List<String> keyValue = pair.split(': ');
        if (keyValue.length == 2) {
          parsed[keyValue[0].trim()] = keyValue[1].trim();
        }
      }
      // Handle nested lists (clinical_details, suggestions, environmental_details)
      // This is a basic parse; adjust based on actual string format
      if (explanationString.contains('clinical_details')) {
        parsed['clinical_details'] = explanationString.contains('clinical_details: [')
            ? RegExp(r'clinical_details: \[(.*?)\]').firstMatch(explanationString)?.group(1)?.split(', ') ?? []
            : [];
      }
      if (explanationString.contains('suggestions')) {
        parsed['suggestions'] = explanationString.contains('suggestions: [')
            ? RegExp(r'suggestions: \[(.*?)\]').firstMatch(explanationString)?.group(1)?.split(', ') ?? []
            : [];
      }
      if (explanationString.contains('environmental_details')) {
        parsed['environmental_details'] = explanationString.contains('environmental_details: [')
            ? RegExp(r'environmental_details: \[(.*?)\]').firstMatch(explanationString)?.group(1)?.split(', ') ?? []
            : [];
      }
      return parsed;
    } catch (e) {
      print("Error parsing explanation: $e");
      return {'summary': explanationString}; // Fallback to raw string
    }
  }

  @override
  Widget build(BuildContext context) {
    double? prediction = report['prediction'] as double?;
    Map<String, dynamic> explanation = _parseExplanation(report['explanation'] as String);

    return Scaffold(
      appBar: AppBar(
        backgroundColor: Colors.deepPurpleAccent,
        title: Text(
          'Report Details',
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
        ),
        child: SingleChildScrollView(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Timestamp (unchanged)
              _buildReportField("Timestamp:", report['timestamp'] ?? 'N/A'),
              SizedBox(height: 16),

              // Prediction (unchanged)
              _buildReportField(
                "Prediction:",
                prediction != null ? '${(prediction * 100).toStringAsFixed(1)}%' : 'N/A',
              ),
              SizedBox(height: 16),

              // Explanation Section (styled like DiabetesPredictionPage.dart)
              Card(
                elevation: 4,
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                child: Padding(
                  padding: const EdgeInsets.all(16.0),
                  child: Column(
                    children: [
                      const Text(
                        "Risk Prediction",
                        style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
                      ),
                      const SizedBox(height: 10),
                      PieChart(
                        dataMap: _getPieChartData(prediction),
                        chartRadius: MediaQuery.of(context).size.width / 3,
                        colorList: [Colors.redAccent, Colors.greenAccent],
                        chartType: ChartType.ring,
                        ringStrokeWidth: 20,
                        centerText: prediction != null
                            ? "${(prediction * 100).toStringAsFixed(1)}%"
                            : "N/A",
                        centerTextStyle: const TextStyle(
                          fontSize: 18,
                          fontWeight: FontWeight.bold,
                          color: Colors.black,
                        ),
                        legendOptions: const LegendOptions(
                          showLegendsInRow: true,
                          legendPosition: LegendPosition.bottom,
                        ),
                        chartValuesOptions: const ChartValuesOptions(
                          showChartValues: false,
                        ),
                      ),
                      const SizedBox(height: 10),
                      Text(
                        "Risk Level: ${explanation['risk_level'] ?? 'Unknown'}",
                        style: const TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
                      ),
                    ],
                  ),
                ),
              ),
              const SizedBox(height: 20),

              // Clinical Details Section
              Card(
                elevation: 2,
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                child: Padding(
                  padding: const EdgeInsets.all(16.0),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text(
                        "Clinical Details",
                        style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                      ),
                      const SizedBox(height: 10),
                      if (explanation.containsKey('clinical_details') && explanation['clinical_details'].isNotEmpty)
                        SizedBox(
                          height: 150,
                          child: ListView.builder(
                            shrinkWrap: true,
                            itemCount: (explanation['clinical_details'] as List).length,
                            itemBuilder: (context, index) {
                              final detail = (explanation['clinical_details'] as List)[index];
                              return Padding(
                                padding: const EdgeInsets.symmetric(vertical: 4.0),
                                child: Text(
                                  "- $detail",
                                  style: const TextStyle(fontSize: 14),
                                ),
                              );
                            },
                          ),
                        )
                      else
                        const Text("No clinical details available."),
                    ],
                  ),
                ),
              ),
              const SizedBox(height: 20),

              // Suggestions Section
              Card(
                elevation: 4,
                color: Colors.blue[50],
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                child: Padding(
                  padding: const EdgeInsets.all(16.0),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text(
                        "Suggestions",
                        style: TextStyle(
                          fontSize: 18,
                          fontWeight: FontWeight.bold,
                          color: Colors.blueAccent,
                        ),
                      ),
                      const SizedBox(height: 10),
                      if (explanation.containsKey('suggestions') && explanation['suggestions'].isNotEmpty)
                        ListView.builder(
                          shrinkWrap: true,
                          physics: const NeverScrollableScrollPhysics(),
                          itemCount: (explanation['suggestions'] as List).length,
                          itemBuilder: (context, index) {
                            final suggestion = (explanation['suggestions'] as List)[index];
                            return Padding(
                              padding: const EdgeInsets.symmetric(vertical: 4.0),
                              child: Row(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  const Icon(Icons.check_circle, size: 16, color: Colors.blueAccent),
                                  const SizedBox(width: 8),
                                  Expanded(
                                    child: Text(
                                      suggestion,
                                      style: const TextStyle(fontSize: 15),
                                    ),
                                  ),
                                ],
                              ),
                            );
                          },
                        )
                      else
                        const Text("No suggestions available."),
                    ],
                  ),
                ),
              ),
              const SizedBox(height: 20),

              // Environmental Impact Section
              if (explanation.containsKey('environmental_details') && explanation['environmental_details'].isNotEmpty)
                ExpansionTile(
                  title: const Text(
                    "Environmental Impact (Optional)",
                    style: TextStyle(fontSize: 16, fontWeight: FontWeight.w500),
                  ),
                  initiallyExpanded: false,
                  children: [
                    Padding(
                      padding: const EdgeInsets.all(8.0),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: (explanation['environmental_details'] as List)
                            .map((detail) => Padding(
                                  padding: const EdgeInsets.symmetric(vertical: 4.0),
                                  child: Text(
                                    "- $detail",
                                    style: const TextStyle(fontSize: 13),
                                  ),
                                ))
                            .toList(),
                      ),
                    ),
                  ],
                ),
              const SizedBox(height: 20),

              // Summary
              if (explanation.containsKey('summary'))
                Text(
                  "Summary: ${explanation['summary']}",
                  style: const TextStyle(fontSize: 16, fontStyle: FontStyle.italic),
                ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildReportField(String label, String value) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          label,
          style: TextStyle(
            fontSize: 20,
            fontWeight: FontWeight.bold,
            color: Colors.deepPurpleAccent,
          ),
        ),
        SizedBox(height: 8),
        Container(
          padding: EdgeInsets.all(12.0),
          width: double.infinity,
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
          child: Text(
            value,
            style: TextStyle(
              fontSize: 16,
              color: Colors.black87,
            ),
          ),
        ),
      ],
    );
  }
}
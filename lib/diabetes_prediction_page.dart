import 'package:flutter/material.dart';
import 'package:pie_chart/pie_chart.dart';
import 'services/ml_service.dart';
import 'services/database_helper.dart'; // Import DatabaseHelper
import 'package:pdf/pdf.dart';
import 'package:pdf/widgets.dart' as pw;
import 'package:path_provider/path_provider.dart';
import 'dart:io';
import 'glucose_check.dart';
class DiabetesPredictionPage extends StatefulWidget {
  final Map<String, dynamic> clinicalData;

  const DiabetesPredictionPage({Key? key, required this.clinicalData}) : super(key: key);

  @override
  _DiabetesPredictionPageState createState() => _DiabetesPredictionPageState();
}

class _DiabetesPredictionPageState extends State<DiabetesPredictionPage> {
  bool _isLoading = true;
  final MlService _mlService = MlService();
  final DatabaseHelper _dbHelper = DatabaseHelper(); // Instantiate DatabaseHelper

  @override
  void initState() {
    super.initState();
    _loadData();
  }

  Future<void> _loadData() async {
    try {
      // Extract Aadhaar separately
      String aadhaar = widget.clinicalData['Aadhaar'] as String;
      Map<String, dynamic> clinicalDataWithoutAadhaar = Map.from(widget.clinicalData)..remove('Aadhaar');

      // Pass only clinical data to MlService
      await _mlService.loadModelAndRun(clinicalDataWithoutAadhaar);
      double? prediction = _mlService.getPrediction();
      Map<String, dynamic>? explanation = _mlService.getExplanation();

      // Save to database with Aadhaar and timestamp
      await _dbHelper.insertReport(aadhaar, prediction, explanation ?? {});
      setState(() => _isLoading = false);
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("Error loading prediction: $e")),
      );
      setState(() => _isLoading = false);
    }
  }
Future<void> _downloadReport() async {
    final pdf = pw.Document();
    final explanation = _mlService.getExplanation() ?? {};

    pdf.addPage(
      pw.Page(
        build: (pw.Context context) {
          return pw.Column(
            crossAxisAlignment: pw.CrossAxisAlignment.start,
            children: [
              pw.Text("Diabetes Risk Report", style: pw.TextStyle(fontSize: 24, fontWeight: pw.FontWeight.bold)),
              pw.SizedBox(height: 10),
              pw.Text("Risk Level: ${explanation['risk_level'] ?? 'Unknown'}", style: pw.TextStyle(fontSize: 18)),
              pw.SizedBox(height: 10),
              pw.Text("Suggestions:", style: pw.TextStyle(fontSize: 18, fontWeight: pw.FontWeight.bold)),
              pw.Column(
                children: (explanation['suggestions'] as List?)?.map((s) => pw.Text("- $s"))?.toList() ?? [],
              ),
              pw.SizedBox(height: 10),
              pw.Text("Clinical Details:", style: pw.TextStyle(fontSize: 18, fontWeight: pw.FontWeight.bold)),
              pw.Column(
                children: (explanation['clinical_details'] as List?)?.map((c) => pw.Text("- $c"))?.toList() ?? [],
              ),
            ],
          );
        },
      ),
    );

    final directory = await getApplicationDocumentsDirectory();
    final file = File("${directory.path}/Diabetes_Report.pdf");
    await file.writeAsBytes(await pdf.save());

    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text("Report saved: ${file.path}")),
    );
  }

  @override
  void dispose() {
    _mlService.dispose();
    super.dispose();
  }

  // Pie chart data preparation
  Map<String, double> _getPieChartData() {
    final probability = _mlService.getPrediction();
    if (probability == null) {
      return {"Risk": 0.0, "No Risk": 100.0};
    }
    return {
      "Risk": probability * 100,
      "No Risk": (1 - probability) * 100,
    };
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("Diabetes Risk Assessment"),
        backgroundColor: Colors.purple,
          actions: [
    IconButton(
      icon: const Icon(Icons.flag, color: Colors.red),
      tooltip: "Report False Prediction",
      onPressed: () {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text("False prediction reported."),
            backgroundColor: Colors.red,
          ),
        );
      },
    ),
  ],
      ),
      body: _isLoading
          ? const Center(child: CircularProgressIndicator())
          : SingleChildScrollView(
              padding: const EdgeInsets.all(16.0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // Risk Prediction Section (Pie Chart)
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
                            dataMap: _getPieChartData(),
                            chartRadius: MediaQuery.of(context).size.width / 3,
                            colorList: [Colors.redAccent, Colors.greenAccent],
                            chartType: ChartType.ring,
                            ringStrokeWidth: 20,
                            centerText: _mlService.getPrediction() != null
                                ? "${(_mlService.getPrediction()! * 100).toStringAsFixed(1)}%"
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
                            "Risk Level: ${_mlService.getExplanation()?['risk_level'] ?? 'Unknown'}",
                            style: const TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
                          ),
                          if (_mlService.getExplanation()?['risk_level'] == 'Unknown')
                            const Text(
                              "Please provide more clinical data for an accurate assessment.",
                              style: TextStyle(color: Colors.grey, fontStyle: FontStyle.italic),
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
                          if (_mlService.getExplanation() != null)
                            SizedBox(
                              height: 150, // Fixed height for scrollable list
                              child: ListView.builder(
                                shrinkWrap: true,
                                itemCount: (_mlService.getExplanation()!['clinical_details'] as List).length,
                                itemBuilder: (context, index) {
                                  final detail = (_mlService.getExplanation()!['clinical_details'] as List)[index];
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

                  // Suggestions Section (Emphasized)
                  Card(
                    elevation: 4,
                    color: Colors.blue[50], // Light background to draw attention
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
                          if (_mlService.getExplanation() != null)
                            ListView.builder(
                              shrinkWrap: true,
                              physics: const NeverScrollableScrollPhysics(),
                              itemCount: (_mlService.getExplanation()!['suggestions'] as List).length,
                              itemBuilder: (context, index) {
                                final suggestion = (_mlService.getExplanation()!['suggestions'] as List)[index];
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

                  // Environmental Impact Section (Minimized)
                  if (_mlService.getExplanation()?['environmental_details']?.isNotEmpty ?? false)
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
                            children: (_mlService.getExplanation()!['environmental_details'] as List)
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

                  // Summary (Brief)
                  if (_mlService.getExplanation() != null)
                    Text(
                      "Summary: ${_mlService.getExplanation()!['summary']}",
                      style: const TextStyle(fontSize: 16, fontStyle: FontStyle.italic),
                    ),
                    const SizedBox(height: 20),
                  ElevatedButton(
  onPressed: _downloadReport,
  style: ElevatedButton.styleFrom(
    backgroundColor: Colors.purple, // Purple background
    foregroundColor: Colors.white, // White text
    padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12), // Padding
    textStyle: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold), // Text styling
  ),
  child: const Text("Download Report"),
),
ElevatedButton(
      onPressed: () {
        Navigator.push(
          context,
          MaterialPageRoute(builder: (context) => GlucoseCheckPage()),
        );
      },
      style: ElevatedButton.styleFrom(
        backgroundColor: Colors.blue, // Blue background
        foregroundColor: Colors.white, // White text
      ),
      child: const Text("Check Glucose"),
    ),

                ],
              ),
            ),
            
    );
   
  }
}
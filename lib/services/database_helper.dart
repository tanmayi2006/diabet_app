import 'package:sqflite/sqflite.dart';
import 'package:path/path.dart';
import 'package:path_provider/path_provider.dart';
import 'dart:io';

class DatabaseHelper {
  // Singleton pattern for single database instance
  static final DatabaseHelper _instance = DatabaseHelper._internal();
  factory DatabaseHelper() => _instance;
  DatabaseHelper._internal();

  static Database? _database;

  // Getter for database instance
  Future<Database> get database async {
    if (_database != null) return _database!;
    _database = await _initDatabase();
    return _database!;
  }

  // Initialize the database
  Future<Database> _initDatabase() async {
    try {
      Directory documentsDirectory = await getApplicationDocumentsDirectory();
      String path = join(documentsDirectory.path, 'diabetes_reports.db');
      print("Database path: $path"); // Debug log for path verification
      return await openDatabase(
        path,
        version: 1,
        onCreate: (db, version) async {
          await db.execute('''
            CREATE TABLE reports (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              aadhaar TEXT NOT NULL,
              prediction REAL,              -- Stores the prediction probability (nullable)
              explanation TEXT,             -- Stores the explanation as a string
              timestamp TEXT NOT NULL       -- ISO 8601 timestamp
            )
          ''');
          print("Database table 'reports' created");
        },
        onOpen: (db) {
          print("Database opened successfully");
        },
      );
    } catch (e) {
      print("Error initializing database: $e");
      rethrow; // Rethrow to allow caller to handle the error
    }
  }

  // Insert a prediction report into the database
  Future<void> insertReport(String aadhaar, double? prediction, Map<String, dynamic> explanation) async {
    try {
      final db = await database;
      await db.insert(
        'reports',
        {
          'aadhaar': aadhaar,
          'prediction': prediction, // Nullable REAL field
          'explanation': explanation.toString(), // Convert map to string for simplicity
          'timestamp': DateTime.now().toIso8601String(), // ISO 8601 timestamp
        },
        conflictAlgorithm: ConflictAlgorithm.replace, // Replace if duplicate (e.g., same Aadhaar with same timestamp)
      );
      print("Report saved for Aadhaar: $aadhaar at ${DateTime.now()}");
    } catch (e) {
      print("Error inserting report: $e");
      rethrow; // Allow caller to handle the error
    }
  }

  // Optional: Retrieve all reports for a specific Aadhaar (for future use)
  Future<List<Map<String, dynamic>>> getReportsByAadhaar(String aadhaar) async {
    try {
      final db = await database;
      final result = await db.query(
        'reports',
        where: 'aadhaar = ?',
        whereArgs: [aadhaar],
        orderBy: 'timestamp DESC', // Latest reports first
      );
      print("Retrieved ${result.length} reports for Aadhaar: $aadhaar");
      return result;
    } catch (e) {
      print("Error retrieving reports: $e");
      return [];
    }
  }

  // Optional: Retrieve all reports (for debugging or admin use)
  Future<List<Map<String, dynamic>>> getAllReports() async {
    try {
      final db = await database;
      final result = await db.query('reports', orderBy: 'timestamp DESC');
      print("Retrieved ${result.length} total reports");
      return result;
    } catch (e) {
      print("Error retrieving all reports: $e");
      return [];
    }
  }
}
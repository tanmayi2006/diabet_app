import 'package:sqflite/sqflite.dart';
import 'package:path/path.dart';
import 'package:path_provider/path_provider.dart';
import 'dart:io';

class DatabaseHelper {
  static final DatabaseHelper _instance = DatabaseHelper._internal();
  factory DatabaseHelper() => _instance;
  DatabaseHelper._internal();

  static Database? _database;

  Future<Database> get database async {
    if (_database != null) return _database!;
    _database = await _initDatabase();
    return _database!;
  }

  Future<Database> _initDatabase() async {
    Directory documentsDirectory = await getApplicationDocumentsDirectory();
    String path = join(documentsDirectory.path, 'diabetes_reports.db');
    return await openDatabase(
      path,
      version: 1,
      onCreate: (db, version) async {
        await db.execute('''
          CREATE TABLE reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            aadhaar TEXT NOT NULL,
            prediction REAL,
            explanation TEXT,
            timestamp TEXT NOT NULL
          )
        ''');
      },
    );
  }

  Future<void> insertReport(String aadhaar, double? prediction, Map<String, dynamic> explanation) async {
    final db = await database;
    await db.insert(
      'reports',
      {
        'aadhaar': aadhaar,
        'prediction': prediction,
        'explanation': explanation.toString(), // Convert map to string for simplicity
        'timestamp': DateTime.now().toIso8601String(),
      },
      conflictAlgorithm: ConflictAlgorithm.replace,
    );
    print("Report saved for Aadhaar: $aadhaar at ${DateTime.now()}");
  }
}
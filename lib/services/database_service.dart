import 'package:sqflite/sqflite.dart';
import 'package:path/path.dart';

class DatabaseService {
  static Database? _database;

  Future<Database> get database async {
    if (_database != null) return _database!;
    _database = await _initDatabase();
    return _database!;
  }

  Future<Database> _initDatabase() async {
    String path = join(await getDatabasesPath(), 'users.db');
    return await openDatabase(
      path,
      version: 2, // Increment version for migration
      onCreate: (db, version) async {
        await db.execute('''
          CREATE TABLE users (
            aadhaar_number TEXT PRIMARY KEY,
            original_aadhaar_number TEXT NOT NULL, -- New field for original Aadhar
            name TEXT NOT NULL,
            date_of_birth TEXT NOT NULL,
            gender TEXT NOT NULL,
            location TEXT NOT NULL,
            age INTEGER NOT NULL,
            unique_code TEXT NOT NULL UNIQUE,
            qr_code TEXT NOT NULL
          )
        ''');
        await db.execute('''
          CREATE TABLE clinical_data (
            aadhaar_number TEXT PRIMARY KEY,
            encrypted_data TEXT NOT NULL,
            hmac TEXT NOT NULL,
            FOREIGN KEY (aadhaar_number) REFERENCES users(aadhaar_number) ON DELETE CASCADE
          )
        ''');
      },
      onUpgrade: (db, oldVersion, newVersion) async {
        if (oldVersion < 2) {
          // Migration: Add original_aadhaar_number column
          await db.execute('ALTER TABLE users ADD COLUMN original_aadhaar_number TEXT NOT NULL DEFAULT ""');
        }
      },
    );
  }

  Future<void> insertUser(Map<String, dynamic> user) async {
    final db = await database;
    await db.insert('users', user, conflictAlgorithm: ConflictAlgorithm.replace);
  }

  Future<void> insertClinicalData(Map<String, dynamic> data) async {
    final db = await database;
    await db.insert('clinical_data', data, conflictAlgorithm: ConflictAlgorithm.replace);
  }

  Future<Map<String, dynamic>?> getUser(String aadhaarNumber) async {
    final db = await database;
    List<Map<String, dynamic>> result = await db.query(
      'users',
      where: 'aadhaar_number = ?',
      whereArgs: [aadhaarNumber],
    );
    return result.isNotEmpty ? Map<String, dynamic>.from(result.first) : null;
  }

  Future<Map<String, dynamic>?> getClinicalData(String aadhaarNumber) async {
    final db = await database;
    List<Map<String, dynamic>> result = await db.query(
      'clinical_data',
      where: 'aadhaar_number = ?',
      whereArgs: [aadhaarNumber],
    );
    return result.isNotEmpty ? Map<String, dynamic>.from(result.first) : null;
  }

  Future<Map<String, dynamic>?> getUserByUniqueCode(String uniqueCode) async {
    final db = await database;
    List<Map<String, dynamic>> result = await db.query(
      'users',
      where: 'unique_code = ?',
      whereArgs: [uniqueCode],
    );
    return result.isNotEmpty ? Map<String, dynamic>.from(result.first) : null;
  }

  Future<List<Map<String, dynamic>>> getAllUsers() async {
    final db = await database;
    List<Map<String, dynamic>> result = await db.query('users');
    return result.map((map) => Map<String, dynamic>.from(map)).toList();
  }
}
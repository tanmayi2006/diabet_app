import 'dart:convert';
import 'package:uuid/uuid.dart';
import 'package:qr_flutter/qr_flutter.dart';
import 'package:intl/intl.dart';
import 'database_service.dart';
import 'crypto_service.dart';


class BackendService {
  final DatabaseService dbService = DatabaseService();
  final CryptoService cryptoService = CryptoService();
  

  Future<int> calculateAge(String dob) async {
    DateTime birthDate = DateFormat('yyyy-MM-dd').parse(dob);
    DateTime today = DateTime.now();
    int age = today.year - birthDate.year;
    if (today.month < birthDate.month || (today.month == birthDate.month && today.day < birthDate.day)) {
      age--;
    }
    return age;
  }

  Future<String> generateUniqueCode() async => Uuid().v4();

  Future<String> generateQrCode(String uniqueCode) async {
    final qrValidationResult = QrValidator.validate(
      data: uniqueCode,
      version: QrVersions.auto,
      errorCorrectionLevel: QrErrorCorrectLevel.L,
    );
    if (qrValidationResult.status == QrValidationStatus.valid) {
      return uniqueCode;
    }
    throw Exception('QR code generation failed');
  }

  Future<Map<String, dynamic>> addUser(Map<String, dynamic> user) async {
    String hashedAadhaar = cryptoService.hashData(user['aadhaar_number']);
    
    // Check if user already exists
    Map<String, dynamic>? existingUser = await dbService.getUser(hashedAadhaar);
    if (existingUser != null) {
      throw ExistingUserException(existingUser);
    }

    int age = await calculateAge(user['date_of_birth']);
    String uniqueCode = await generateUniqueCode();
    String qrCode = await generateQrCode(uniqueCode);

    final userData = {
      'aadhaar_number': hashedAadhaar,
      'original_aadhaar_number': user['aadhaar_number'],
      'name': user['name'],
      'date_of_birth': user['date_of_birth'],
      'gender': user['gender'],
      'location': user['location'],
      'age': age,
      'unique_code': uniqueCode,
      'qr_code': qrCode,
    };

    await dbService.insertUser(userData);
    return userData;
  }

  Future<Map<String, dynamic>> addClinicalData(Map<String, dynamic> clinicalData) async {
    String hashedAadhaar = cryptoService.hashData(clinicalData['aadhaar_number']);
    if (await dbService.getUser(hashedAadhaar) == null) {
      throw Exception('User not found');
    }

    String jsonData = jsonEncode(clinicalData);
    String encryptedData = cryptoService.encryptData(jsonData);
    String hmac = cryptoService.generateHmac(jsonData);

    await dbService.insertClinicalData({
      'aadhaar_number': hashedAadhaar,
      'encrypted_data': encryptedData,
      'hmac': hmac,
    });

    return clinicalData;
  }

  Future<Map<String, dynamic>?> getUser(String aadhaarNumber) async {
    String hashedAadhaar = cryptoService.hashData(aadhaarNumber);
    var user = await dbService.getUser(hashedAadhaar);
    if (user == null) return null;

    var clinical = await dbService.getClinicalData(hashedAadhaar);
    if (clinical != null) {
      String decrypted = cryptoService.decryptData(clinical['encrypted_data']);
      if (cryptoService.generateHmac(decrypted) != clinical['hmac']) {
        throw Exception('Data integrity compromised');
      }
      user['clinical_data'] = jsonDecode(decrypted);
    }
    return user;
  }

  Future<Map<String, dynamic>?> getUserByUniqueCode(String uniqueCode) async {
    return await dbService.getUserByUniqueCode(uniqueCode);
  }

  Future<List<Map<String, dynamic>>> getAllUsers() async {
    return await dbService.getAllUsers();
  }

  
}

class ExistingUserException implements Exception {
  final Map<String, dynamic> existingUser;

  ExistingUserException(this.existingUser);

  @override
  String toString() => 'User already exists';
}
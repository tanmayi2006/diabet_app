import 'dart:convert';
import 'dart:typed_data';
import 'package:crypto/crypto.dart';
import 'package:pointycastle/api.dart';
import 'package:pointycastle/block/aes_fast.dart';
import 'package:pointycastle/block/modes/cfb.dart';

class CryptoService {
  static final Uint8List _encryptionKey = Uint8List.fromList(List.generate(32, (i) => i));
  static final Uint8List _hmacKey = Uint8List.fromList(List.generate(32, (i) => i + 32));

  String hashData(String data) {
    return sha256.convert(utf8.encode(data)).toString();
  }

  String generateHmac(String data) {
    var hmac = Hmac(sha256, _hmacKey);
    return hmac.convert(utf8.encode(data)).toString();
  }

  String encryptData(String data) {
    final iv = Uint8List(16)..setRange(0, 16, List.generate(16, (i) => i));
    final cipher = CFBBlockCipher(AESFastEngine(), 16);
    cipher.init(true, ParametersWithIV(KeyParameter(_encryptionKey), iv));
    final input = utf8.encode(data);
    final encrypted = cipher.process(Uint8List.fromList(input));
    return base64Encode(iv + encrypted);
  }

  String decryptData(String encryptedBase64) {
    final encryptedBytes = base64Decode(encryptedBase64);
    final iv = encryptedBytes.sublist(0, 16);
    final cipher = CFBBlockCipher(AESFastEngine(), 16);
    cipher.init(false, ParametersWithIV(KeyParameter(_encryptionKey), iv));
    final decrypted = cipher.process(encryptedBytes.sublist(16));
    return utf8.decode(decrypted);
  }
}
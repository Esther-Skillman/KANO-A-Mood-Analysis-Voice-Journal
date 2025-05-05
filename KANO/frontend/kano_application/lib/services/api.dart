import 'package:dio/dio.dart';
import 'dart:io';
import 'package:flutter/foundation.dart';

class ApiService {
  static final Dio _dio = Dio();
  static const String baseUrl =
      'https://kano-mood-analysis-870b6bb34a2d.herokuapp.com';
  static const String userId = 'anonymous';

  static Future<Map<String, dynamic>> processAudio(File file) async {
    final formData = FormData.fromMap({
      'file': await MultipartFile.fromFile(file.path, filename: 'audio.wav'),
      'user_id': userId,
    });
    final response = await _dio.post(
      '$baseUrl/process_audio/',
      data: formData,
    );
    return response.data;
  }

  static Future<Map<String, dynamic>> processAudioWeb(
      Uint8List fileBytes, String fileName) async {
    final formData = FormData.fromMap({
      'file': MultipartFile.fromBytes(fileBytes, filename: fileName),
      'user_id': userId,
    });
    final response = await _dio.post(
      '$baseUrl/process_audio/',
      data: formData,
    );
    return response.data;
  }

  static Future<List<Map<String, dynamic>>> getRecordings() async {
    final response = await _dio.get(
      '$baseUrl/user_recordings/',
      queryParameters: {'user_id': userId},
    );
    return List<Map<String, dynamic>>.from(response.data['recordings']);
  }

  static Future<Map<String, dynamic>> deleteAll() async {
    final response = await _dio.delete(
      '$baseUrl/delete_all/',
      queryParameters: {'user_id': userId},
    );
    return response.data;
  }
}

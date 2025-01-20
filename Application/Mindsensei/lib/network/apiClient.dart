import 'dart:io';
import 'package:dio/dio.dart';
import 'package:dio/io.dart';

import '../constants/appString.dart';

class ApiClient {
  final Dio dioClient;

  ApiClient({BaseOptions? baseOptions})
      : dioClient = Dio(
    baseOptions ??
        BaseOptions(
          baseUrl: AppStrings.apiEndpoints.baseURL,
          connectTimeout: const Duration(milliseconds: 60000),
          receiveTimeout: const Duration(milliseconds: 60000),
        ),
  ) {
    dioClient.httpClientAdapter = DefaultHttpClientAdapter()
      ..onHttpClientCreate = (HttpClient client) {
        client.badCertificateCallback =
            (X509Certificate cert, String host, int port) => true;
        return client;
      };

    dioClient.interceptors.add(LogInterceptor());
  }

  /// Factory method to create a default instance of `ApiClient`.
  static ApiClient defaultClient() {
    return ApiClient();
  }
}

class CustomException implements Exception {
  final String errorCode;
  final String errorMessage;

  CustomException(this.errorCode, this.errorMessage);

  @override
  String toString() => 'Error $errorCode: $errorMessage';
}

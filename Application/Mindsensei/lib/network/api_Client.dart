import 'dart:io';

// import 'package:dio/adapter.dart';
import 'package:dio/dio.dart';
import 'package:dio/io.dart';

import '../constants/appString.dart';


enum HTTPContentType { json, raw, binary, pdf }

class HTTPStatus {
  static const int HTTP_OK = 200;
}

class ApiClient {
  late Dio dioClient;

  ApiClient.defaultClient({ BaseOptions? baseOptions}) {
    if (baseOptions == null) {
      baseOptions = BaseOptions(
        baseUrl: AppStrings.apiEndpoints.baseURL,
        connectTimeout: Duration(milliseconds: 60000),
        receiveTimeout: Duration(milliseconds: 60000),
      );
    }
    dioClient = Dio(baseOptions);
    (dioClient.httpClientAdapter as DefaultHttpClientAdapter)
        .onHttpClientCreate = (HttpClient client) {
      //bypass SSL pinning
      client.badCertificateCallback =
          (X509Certificate cert, String host, int port) => true;
      return client;
    };

    //log api calls
    dioClient.interceptors.add(LogInterceptor());
  }
}

class CustomException implements Exception {
  late String? errorCode;
  late String? errorMessage;
  late String? humanizedMessage;

  CustomException(
      { this.errorCode,  this.errorMessage,  this.humanizedMessage});

  CustomException.fromJson(Map<String, dynamic> json) {

    errorCode = json['errorCode'] == null ? null : json['errorCode'];
    errorMessage = json['errorMessage'] == null ? null : json['errorMessage'];
    humanizedMessage =
    json['errorMessage'] == null ? null : json['errorMessage'];
  }

  String toString() {
    return humanizedMessage!;
  }
}

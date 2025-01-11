import 'dart:convert';
import 'package:dio/dio.dart';
import 'package:get/get.dart';
import 'package:get/get_core/src/get_main.dart';
import 'package:get_storage/get_storage.dart';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';
import '../../constants/appString.dart';
import '../../constants/app_Colors.dart';
import '../../network/api_Client.dart';
import '../../network/request_options_builder.dart';

const baseUrl = "http://192.168.0.105:3000/";


const loginUser = 'token';

class MyApiClient {
  final http.Client httpClient;

  MyApiClient({required this.httpClient});

  GetStorage box = GetStorage();
  late SharedPreferences prefs;

  registerUser(Map<String, dynamic> mapData) async {
    try {
      print('login mapData : ' + mapData.toString());
      ApiClient apiClient = ApiClient.defaultClient();
      var response = await apiClient.dioClient.post(
        AppStrings.apiEndpoints.registerUser,
        data: jsonEncode(mapData),
        options: (await AppRequestOptionsBuilder().defaultHeader()).build(),
      );
      if (response.statusCode == 201) {
        dynamic jsonResponse = response.data;
        print("registerUser response: " + jsonEncode(jsonResponse));
        return jsonResponse;
      } else {
        print('Error in registerUser response: Status Code ' + response.statusCode.toString());
        print('Response data: ' + response.data.toString());

        return null;
      }
    }
    catch (e) {
      if (e is DioException) {
        print('DioException in registerJudge: ' + e.toString());
        print('Response status code: ${e.response?.statusCode}');
        print('Response data: ${e.response?.data}');
        Get.snackbar("REGISTER FAILED ",
            "${e.response?.data['error'].toString()}",
            duration: const Duration(seconds: 3),
            backgroundColor: AppColors.red);
      } else {
        print('Exception in registerJudge: ' + e.toString());
      }
      return null;
    }
  }

  loginUser(userName, password) async {

    dynamic mapData = {
      "email": userName,
      "password": password
    };
    try {
      print('login mapData : ' + mapData.toString());
      ApiClient apiClient = ApiClient.defaultClient();
      var response = await apiClient.dioClient.post(
        AppStrings.apiEndpoints.login,
        data: jsonEncode(mapData),
        options: (await AppRequestOptionsBuilder().defaultHeader()).build(),
      );
      if (response.statusCode == 200) {
        dynamic jsonResponse = response.data;
        print("loginUser response: " + jsonEncode(jsonResponse));
        return jsonResponse;
      } else {
        print('Error in loginUser response: Status Code ' + response.statusCode.toString());
        print('Response data: ' + response.data.toString());

        return null;
      }
    }
    catch (e) {
      if (e is DioException) {
        print('DioException in loginUser: ' + e.toString());
        print('Response status code: ${e.response?.statusCode}');
        print('Response data: ${e.response?.data}');
        Get.snackbar("loginUser FAILED",
            "${e.response?.data['error'].toString()}",
            duration: const Duration(seconds: 3),
            backgroundColor: AppColors.red);
      } else {
        print('Exception in registerJudge: ' + e.toString());
      }
      return null;
    }

  }




}

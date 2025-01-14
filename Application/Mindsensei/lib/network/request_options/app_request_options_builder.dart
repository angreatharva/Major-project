import 'package:dio/dio.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../constants/app_string.dart';
import 'request_options/request_options_builder.dart';

class AppRequestOptionsBuilder extends RequestOptionsBuilder {
  late SharedPreferences? sharedPreferences;

  AppRequestOptionsBuilder({Options? options}) {
    _options = options ?? Options();
  }

  late Options _options;

  Future<AppRequestOptionsBuilder> defaultHeader() async {
    _options.headers = {
      // "Content-Type": "application/x-www-form-urlencoded",
      'Content-type': 'application/json',
      // "Accept" : "*/*",
    };
    return this;
  }

  Future<AppRequestOptionsBuilder> defaultHeaderWithToken() async {
    sharedPreferences = await SharedPreferences.getInstance();
    String? token = sharedPreferences?.getString(AppStrings.token)!;

    if (token != null) {
      _options.headers = {
        "Authorization": "Bearer $token",
        "Content-Type": "application/json",
        "Accept" : "*/*",
      };
    } else {
      // Handle the case where 'token' is null or not found.
    }


    return this;
  }

  Options build() {
    return _options;
  }
}

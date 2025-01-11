import 'package:flutter/material.dart';
import 'package:flutter/widgets.dart';
import 'package:get/get.dart';
import 'package:get_storage/get_storage.dart';
import 'package:flutter_easyloading/flutter_easyloading.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../constants/api_Endpoints.dart';
import '../constants/app_Colors.dart';
import '../data/repository/post_repository.dart';
import '../data_model/dropDown_model.dart';
import '../routes/app_routes.dart';

class LoginController extends GetxController with SingleGetTickerProviderMixin {
  final MyRepository repository;

  LoginController({required this.repository}) : assert(repository != null);

  //if status is false then add other then remove..........

  var isLoading = false.obs;
  GetStorage box = GetStorage();
  Rx<TextEditingController> emailLogin = TextEditingController().obs;
  Rx<TextEditingController> passwordLogin = TextEditingController().obs;

  Rx<TextEditingController> userNameRegister = TextEditingController().obs;
  Rx<TextEditingController> phoneRegister = TextEditingController().obs;
  Rx<TextEditingController> ageRegister = TextEditingController().obs;
  var selectedGender = 'Male'.obs;
  Rx<TextEditingController> emailRegister = TextEditingController().obs;
  Rx<TextEditingController> passwordRegister = TextEditingController().obs;

  var url = ''.obs;
  final scaffoldKey = GlobalKey<ScaffoldState>();
  var isShowPass = true.obs;
  var isSuperior = false.obs;
  var dummyText = ''.obs;
  late SharedPreferences prefs;
  // var selectedJudge;
  var selectedJudgeType = 'Superior'.obs;
  RxList<DropdownList> ageGroupList = <DropdownList>[
    DropdownList('under12', 'Under 12'),
    DropdownList('under14', 'Under 14'),
    DropdownList('under16', 'Under 16'),
    DropdownList('under18', 'Under 18'),
    DropdownList('above16', 'Above 16'),
    DropdownList('above18', 'Above 18'),
  ].obs;
  var selectedAgeGroup = "".obs;

  @override
  void onClose() {
    // flutterWebViewPlugin.dispose();
    super.onClose();
    EasyLoading.dismiss();
  }

  @override
  void onInit() {
    super.onInit();
    print("LoginController init");
    box = GetStorage();

    // EasyLoading.init();
    EasyLoading.dismiss();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.paused) {
      // App is being minimized
      // EasyLoading.dismiss();
    }
  }

  registerUser(userName, phone, age, gender, email,password) async {
    EasyLoading.show();
    try {
      print('auth : ' + userName + "-" + password);
      Map<String, dynamic> mapData = {
        "userName":userName,
        "phone":phone,
        "age":age,
        "gender":gender,
        "email":email,
        "password":password
      };
      if (email.isNotEmpty && password.isNotEmpty) {
        var data = await repository.registerUser(mapData);
            // .then((data) {ATH
          print("Register data: $data");
          if (data != null) {
            if (data['status'] != 'Failure') {
              print('Register Success');
              Get.toNamed(Routes.DASHBOARD);
              EasyLoading.dismiss();

            }
          }
          else {
            print("Register Failed");

            EasyLoading.dismiss();
          }
      } else {
        Get.snackbar(
            "Email or Password not Entered", "Please Enter Email and Password");
      }
    } catch (e) {
      EasyLoading.dismiss();
      print("exception: " + e.toString());
    }
  }

  registerDoctor(userName, phone, age, gender, email,password) async {
    EasyLoading.show();
    try {
      print('auth : ' + userName + "-" + password);
      Map<String, dynamic> mapData = {
        "userName":userName,
        "phone":phone,
        "age":age,
        "gender":gender,
        "email":email,
        "password":password
      };
      if (email.isNotEmpty && password.isNotEmpty) {
        var data = await repository.registerUser(mapData);
        // .then((data) {ATH
        print("Register data: $data");
        if (data != null) {
          if (data['status'] != 'Failure') {
            print('Register Success');

            Get.toNamed(Routes.DASHBOARD);

            EasyLoading.dismiss();

          }
        }
        else {
          print("Register Failed");

          EasyLoading.dismiss();
        }
      } else {
        Get.snackbar(
            "Email or Password not Entered", "Please Enter Email and Password");
      }
    } catch (e) {
      EasyLoading.dismiss();
      print("exception: " + e.toString());
    }
  }

  void loginUser(email, password) async {
    try {
      if (email.isNotEmpty && password.isNotEmpty) {
        repository.loginUser(email, password).then((data) {
          if (data != null) {
            if (data["success"] == true) {
              print("Login Success"+data.toString());
              Get.toNamed(Routes.DASHBOARD);
            } else {
              print("Login Failed");
              Get.snackbar("Login failed", "Please check Email and Password");
            }
          }
        });
      }
    } catch (e) {
      EasyLoading.dismiss();
      print("exception: " + e.toString());
    }
  }
}

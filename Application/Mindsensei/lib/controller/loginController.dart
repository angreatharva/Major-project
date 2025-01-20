import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter/widgets.dart';
import 'package:get/get.dart';
import 'package:get_storage/get_storage.dart';
import 'package:flutter_easyloading/flutter_easyloading.dart';
import 'package:image_picker/image_picker.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../data/model/dropDown_model.dart';
import '../data/repository/postRepository.dart';
import '../routes/appRoutes.dart';

class LoginController extends GetxController with SingleGetTickerProviderMixin {
  final MyRepository repository;

  LoginController({required this.repository}) : assert(repository != null);

  //if status is false then add other then remove..........

  var isLoading = false.obs;
  GetStorage box = GetStorage();
  Rx<TextEditingController> emailLogin = TextEditingController().obs;
  Rx<TextEditingController> passwordLogin = TextEditingController().obs;

  Rx<TextEditingController> userNameRegister = TextEditingController().obs;
  Rx<TextEditingController> doctorNameRegister = TextEditingController().obs;

  Rx<TextEditingController> userPhoneRegister = TextEditingController().obs;
  Rx<TextEditingController> doctorPhoneRegister = TextEditingController().obs;

  Rx<TextEditingController> userAgeRegister = TextEditingController().obs;
  Rx<TextEditingController> doctorAgeRegister = TextEditingController().obs;

  Rx<TextEditingController> userEmailRegister = TextEditingController().obs;
  Rx<TextEditingController> doctorEmailRegister = TextEditingController().obs;

  Rx<TextEditingController> doctorQualificationRegister = TextEditingController().obs;
  Rx<TextEditingController> doctorSpecializationRegister = TextEditingController().obs;
  Rx<TextEditingController> doctorLicenseNumberRegister = TextEditingController().obs;

  Rx<TextEditingController> userPasswordRegister = TextEditingController().obs;
  Rx<TextEditingController> doctorPasswordRegister = TextEditingController().obs;

  var selectedGenderUser = 'Male'.obs;
  var selectedGenderDoctor = 'Male'.obs;

  var userType = Rxn<String>();
  var userSelection = false.obs;

  Rx<File?> imageFile = Rx<File?>(null);
  RxString base64String = "".obs;


  final scaffoldKey = GlobalKey<ScaffoldState>();
  var isShowPass = true.obs;
  var dummyText = ''.obs;
  late SharedPreferences prefs;


  @override
  void onClose() {
    super.onClose();

    EasyLoading.dismiss();
  }

  @override
  void onInit() {
    super.onInit();
    print("LoginController init");
    box = GetStorage();
    requestPermission();
    var userType = box.write('userType', null);
    EasyLoading.dismiss();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.paused) {
      // App is being minimized
      // EasyLoading.dismiss();
    }
  }

  // Request storage permission
  Future<void> requestPermission() async {
    PermissionStatus status = await Permission.photos.request();
    if (status.isGranted) {
      print('Permission granted');
    } else if (status.isDenied) {
      print('Permission denied');
    } else if (status.isPermanentlyDenied) {
      print('Permission permanently denied');
      openAppSettings();
    }
  }

  // Pick an image from the gallery
  Future<void> pickImage() async {
    final picker = ImagePicker();
    final XFile? pickedFile = await picker.pickImage(source: ImageSource.gallery);

    if (pickedFile != null) {
      imageFile.value = File(pickedFile.path);

      // Convert the picked image to Base64
      String base64Str = convertToBase64(imageFile.value!);
      base64String.value = base64Str;

      print("Base64 String: ${base64String.value}");
    }
  }

  // Convert image to Base64
  String convertToBase64(File file) {
    List<int> imageBytes = file.readAsBytesSync();
    String base64Image = base64Encode(imageBytes);
    return 'data:image/jpeg;base64,$base64Image';
  }

  registerUser(userName, phone, age, gender, email,password) async {
    EasyLoading.show();
    try {
      print('auth : ' + userName + "-" + password);
      Map<String, dynamic> mapData = {
        "userName":userName,
        "phone":phone,
        "age":int.parse(age),
        "gender":gender,
        "email":email,
        "password":password
      };
      if (email.isNotEmpty && password.isNotEmpty) {
        var data = await repository.registerUser(mapData);
            // .then((data) {ATH
          print("Register data: $data");
          box.write("Username",data['data']['userName']);

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

  registerDoctor(doctorName, phone, age, gender, email, qualification, specialization, licenseNumber, password,image) async {
    EasyLoading.show();

    try {
      print('auth : ' + doctorName + "-" + password);
      Map<String, dynamic> mapData = {
        "doctorName":doctorName,
        "phone":phone,
        "age":int.parse(age),
        "gender":gender,
        "email":email,
        "qualification":qualification,
        "specialization":specialization,
        "licenseNumber":licenseNumber,
        "image":image,
        "password":password
      };
      if (email.isNotEmpty && password.isNotEmpty) {
        var data = await repository.registerDoctor(mapData);
        // .then((data) {ATH
        print("Register data: $data");
        print("Register data doctorName: ${data['data']['doctorName']}");
        box.write("Username",data['data']['doctorName']);
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
              print("Register data: $data");
              box.write("Username",data['userName']);
              box.write("loggedInRole",data['role']);
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

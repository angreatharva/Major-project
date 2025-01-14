import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:get/get_state_manager/src/rx_flutter/rx_getx_widget.dart';
import 'package:get/get_state_manager/src/simple/get_view.dart';

import '../../constants/appString.dart';
import '../../constants/app_Colors.dart';
import '../../controller/loginController.dart';
import '../../routes/app_routes.dart';

class Register extends GetView<LoginController> {
  Register(this.navigatorKey);
  GlobalKey<NavigatorState> navigatorKey;

  final FocusNode _focusNode = FocusNode();
  final formGlobalKey = GlobalKey<FormState>();
  TextEditingController scanEditingController = TextEditingController();

  init() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      // executes after build
      // controller.checkForAlreadyLogin();
    });
  }

  @override
  Widget build(BuildContext context) {
    return GetX<LoginController>(initState: (_) {
      init();
    }, builder: (controller) {
      return Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Hero(
            tag: 'logo',
            child: Row(
              children: [
                Container(
                  margin: EdgeInsets.symmetric(horizontal: Get.width * 0.02),
                  child: GestureDetector(
                    onDoubleTap: () {
                      Get.toNamed(Routes.LOGIN);
                    },
                    child: ClipRRect(
                      borderRadius: BorderRadius.circular(20.0),
                      child: Image.asset(
                        "assets/images/mindsensei1.png",
                        height: Get.height * 0.15,
                      ),
                    ),
                  ),
                ),
                SizedBox(
                  width: Get.width * 0.01,
                ),
                Container(
                  // width: Get.width * 0.6,
                  child: Text(
                    AppStrings.appName,
                    style: TextStyle(
                      fontSize: 20,
                      fontWeight: FontWeight.bold,
                    ),
                    textAlign: TextAlign.center,
                  ),
                ),
              ],
            ),
          ),
          SizedBox(height: Get.height * 0.02),

          controller.userType.value == 'patient' ?
          //doctor
          Column(
            children: [
              // Username TextField
              Container(
                margin: EdgeInsets.symmetric(horizontal: 20.0),
                child: TextFormField(
                  controller: controller.doctorNameRegister.value,
                  decoration: InputDecoration(
                    labelText: "Enter Doctor Name",
                    contentPadding: EdgeInsets.symmetric(horizontal: 20.0),
                    border: OutlineInputBorder(
                      borderSide: BorderSide(color: Colors.green),
                      borderRadius: BorderRadius.circular(8.0),
                    ),
                  ),
                ),
              ),
              SizedBox(height: Get.height * 0.02),

              //Phone
              Container(
                margin: EdgeInsets.symmetric(horizontal: 20.0),
                child: TextFormField(
                  controller: controller.phoneRegister.value,
                  decoration: InputDecoration(
                    labelText: "Enter Phone Number",
                    contentPadding: EdgeInsets.symmetric(horizontal: 20.0),
                    border: OutlineInputBorder(
                      borderSide: BorderSide(color: Colors.red),
                      borderRadius: BorderRadius.circular(8.0),
                    ),
                  ),
                ),
              ),
              SizedBox(height: Get.height * 0.02),

              //age
              Container(
                margin: EdgeInsets.symmetric(horizontal: 20.0),
                child: TextFormField(
                  controller: controller.ageRegister.value,
                  decoration: InputDecoration(
                    labelText: "Enter Age",
                    contentPadding: EdgeInsets.symmetric(horizontal: 20.0),
                    border: OutlineInputBorder(
                      borderSide: BorderSide(color: Colors.red),
                      borderRadius: BorderRadius.circular(8.0),
                    ),
                  ),
                ),
              ),
              SizedBox(height: Get.height * 0.02),

              //age
              Row(
                children: [
                  Expanded(
                    child: RadioListTile(
                      title: Text('Male'),
                      value: 'Male',
                      groupValue: controller.selectedGender.value,
                      onChanged: (value) =>
                      controller.selectedGender.value = value!,
                    ),
                  ),
                  Expanded(
                    child: RadioListTile(
                      title: Text('Female'),
                      value: 'Female',
                      groupValue: controller.selectedGender.value,
                      onChanged: (value) =>
                      controller.selectedGender.value = value!,
                    ),
                  ),
                ],
              ),
              SizedBox(height: Get.height * 0.02),

              //email
              Container(
                margin: EdgeInsets.symmetric(horizontal: 20.0),
                child: TextFormField(
                  controller: controller.emailRegister.value,
                  decoration: InputDecoration(
                    labelText: "Enter Email",
                    contentPadding: EdgeInsets.symmetric(horizontal: 20.0),
                    border: OutlineInputBorder(
                      borderSide: BorderSide(color: Colors.red),
                      borderRadius: BorderRadius.circular(8.0),
                    ),
                  ),
                ),
              ),
              SizedBox(height: Get.height * 0.02),

              // Password TextField
              Container(
                margin: EdgeInsets.symmetric(horizontal: 20.0),
                child: TextFormField(
                  controller: controller.passwordRegister.value,
                  decoration: InputDecoration(
                    labelText: "Password",
                    contentPadding: EdgeInsets.symmetric(horizontal: 20.0),
                    border: OutlineInputBorder(
                      borderSide: BorderSide(color: Colors.red),
                      borderRadius: BorderRadius.circular(8.0),
                    ),
                  ),
                ),
              ),
              SizedBox(height: Get.height * 0.02),
            ],
          ):
          //patient
          Column(
            children: [
              // Username TextField
              Container(
                margin: EdgeInsets.symmetric(horizontal: 20.0),
                child: TextFormField(
                  controller: controller.userNameRegister.value,
                  decoration: InputDecoration(
                    labelText: "Enter User Name",
                    contentPadding: EdgeInsets.symmetric(horizontal: 20.0),
                    border: OutlineInputBorder(
                      borderSide: BorderSide(color: Colors.green),
                      borderRadius: BorderRadius.circular(8.0),
                    ),
                  ),
                ),
              ),
              SizedBox(height: Get.height * 0.02),

              //Phone
              Container(
                margin: EdgeInsets.symmetric(horizontal: 20.0),
                child: TextFormField(
                  controller: controller.phoneRegister.value,
                  decoration: InputDecoration(
                    labelText: "Enter Phone Number",
                    contentPadding: EdgeInsets.symmetric(horizontal: 20.0),
                    border: OutlineInputBorder(
                      borderSide: BorderSide(color: Colors.red),
                      borderRadius: BorderRadius.circular(8.0),
                    ),
                  ),
                ),
              ),
              SizedBox(height: Get.height * 0.02),

              //age
              Container(
                margin: EdgeInsets.symmetric(horizontal: 20.0),
                child: TextFormField(
                  controller: controller.ageRegister.value,
                  decoration: InputDecoration(
                    labelText: "Enter Age",
                    contentPadding: EdgeInsets.symmetric(horizontal: 20.0),
                    border: OutlineInputBorder(
                      borderSide: BorderSide(color: Colors.red),
                      borderRadius: BorderRadius.circular(8.0),
                    ),
                  ),
                ),
              ),
              SizedBox(height: Get.height * 0.02),

              //age
              Row(
                children: [
                  Expanded(
                    child: RadioListTile(
                      title: Text('Male'),
                      value: 'Male',
                      groupValue: controller.selectedGender.value,
                      onChanged: (value) =>
                      controller.selectedGender.value = value!,
                    ),
                  ),
                  Expanded(
                    child: RadioListTile(
                      title: Text('Female'),
                      value: 'Female',
                      groupValue: controller.selectedGender.value,
                      onChanged: (value) =>
                      controller.selectedGender.value = value!,
                    ),
                  ),
                ],
              ),
              SizedBox(height: Get.height * 0.02),

              //email
              Container(
                margin: EdgeInsets.symmetric(horizontal: 20.0),
                child: TextFormField(
                  controller: controller.emailRegister.value,
                  decoration: InputDecoration(
                    labelText: "Enter Email",
                    contentPadding: EdgeInsets.symmetric(horizontal: 20.0),
                    border: OutlineInputBorder(
                      borderSide: BorderSide(color: Colors.red),
                      borderRadius: BorderRadius.circular(8.0),
                    ),
                  ),
                ),
              ),
              SizedBox(height: Get.height * 0.02),

              // Password TextField
              Container(
                margin: EdgeInsets.symmetric(horizontal: 20.0),
                child: TextFormField(
                  controller: controller.passwordRegister.value,
                  decoration: InputDecoration(
                    labelText: "Password",
                    contentPadding: EdgeInsets.symmetric(horizontal: 20.0),
                    border: OutlineInputBorder(
                      borderSide: BorderSide(color: Colors.red),
                      borderRadius: BorderRadius.circular(8.0),
                    ),
                  ),
                ),
              ),
              SizedBox(height: Get.height * 0.02),
            ],
          ),

          //buttons
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              ElevatedButton(
                style: ButtonStyle(
                  backgroundColor: MaterialStateProperty.all(AppColors.black),
                  shape: MaterialStateProperty.all(RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(10.0))),
                ),
                onPressed: () {

                  Get.toNamed(Routes.LOGIN);
                },
                child: Text(
                  "Login",
                  style: TextStyle(color: AppColors.white),
                ),
              ),

              SizedBox(width: Get.width * 0.05),

              ElevatedButton(
                  style: ButtonStyle(
                    backgroundColor: MaterialStateProperty.all(AppColors.black),
                    shape: MaterialStateProperty.all(RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(10.0))),
                  ),
                  onPressed: () {
                    print("Register Details");
                    print("judgeName :${controller.userNameRegister.value.text}");
                    print("password :${controller.phoneRegister.value.text}");
                    print("judgeName :${controller.ageRegister.value.text}");
                    print("password :${controller.selectedGender.value}");
                    print("judgeName :${controller.emailRegister.value.text}");
                    print("password :${controller.passwordRegister.value.text}");
                    controller.registerUser(
                        controller.userNameRegister.value.text,
                        controller.phoneRegister.value.text,
                        controller.ageRegister.value.text,
                        controller.selectedGender.value,
                        controller.emailRegister.value.text,
                        controller.passwordRegister.value.text,
                    );
                  },
                  child: Text(
                    "Register",
                    style: TextStyle(color: AppColors.white),
                  ))
            ],
          )

        ],
      );
    });
  }

}

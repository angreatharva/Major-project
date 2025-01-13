import 'package:flutter/material.dart';
import 'package:get/get.dart';
import '../../constants/appString.dart';
import '../../constants/app_Colors.dart';
import '../../controller/loginController.dart';
import '../../routes/app_routes.dart';

class Login extends StatelessWidget {
  final GlobalKey<NavigatorState> navigatorKey;

  Login(this.navigatorKey);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      // backgroundColor: Colors.orange,
      body: GetX<LoginController>(
        initState: (_) => init(),
        builder: (controller) {
          return Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              // Logo and title
              Hero(
                tag: 'logo',
                child: Row(
                  children: [
                    Container(
                      margin: EdgeInsets.symmetric(horizontal: 20.0),
                      child: GestureDetector(
                        onDoubleTap: () {
                          Get.toNamed(Routes.REGISTER);
                        },
                        child: ClipRRect(
                          borderRadius: BorderRadius.circular(20.0),
                          child: Image.asset(
                            "assets/images/mindsensei1.png",
                            height: Get.height * 0.18,
                          ),
                        ),
                      ),
                    ),
                    SizedBox(width: Get.width * 0.01),
                    Expanded(
                      child: Container(
                        child: Text(
                          AppStrings.appName,
                          style: TextStyle(
                              fontSize: 20, fontWeight: FontWeight.bold),
                          textAlign: TextAlign.center,
                        ),
                      ),
                    ),
                  ],
                ),
              ),

              SizedBox(height: Get.height * 0.1),

              // Username TextField
              Container(
                margin: EdgeInsets.symmetric(horizontal: 20.0),
                child: TextFormField(
                  controller: controller.emailLogin.value,
                  decoration: InputDecoration(
                    labelText: "Email",
                    contentPadding: EdgeInsets.symmetric(horizontal: 20.0),
                    border: OutlineInputBorder(
                      borderSide: BorderSide(color: Colors.green),
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
                  controller: controller.passwordLogin.value,
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
                      print("userName :${controller.emailLogin.value.text}");
                      print("password :${controller.passwordLogin.value.text}");
                      controller.loginUser(controller.emailLogin.value.text,
                          controller.passwordLogin.value.text);
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
                      print("userName :${controller.emailLogin.value.text}");
                      print("password :${controller.passwordLogin.value.text}");
                      // controller.loginUser(controller.userNameLogin.value.text,
                      //     controller.passwordLogin.value.text);
                      Get.toNamed(Routes.REGISTER);
                    },
                    child: Text(
                      "Register",
                      style: TextStyle(color: AppColors.white),
                    ),
                  ),
                ],
              )

            ],
          );
        },
      ),
    );
  }


  void init() {
    WidgetsBinding.instance?.addPostFrameCallback((_) {
      // executes after build
      // controller.checkForAlreadyLogin();
    });
  }
}

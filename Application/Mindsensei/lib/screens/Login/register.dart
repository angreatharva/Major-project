import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:get/get_state_manager/src/rx_flutter/rx_getx_widget.dart';
import 'package:get/get_state_manager/src/simple/get_view.dart';

import '../../constants/app_string.dart';
import '../../constants/app_colors.dart';
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
      controller.requestPermission();
      // controller.checkForAlreadyLogin();
    });
  }

  @override
  Widget build(BuildContext context) {
    return GetX<LoginController>(initState: (_) {
      init();
    }, builder: (controller) {
      return Scaffold(
        body: SingleChildScrollView(
          child: Column(
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
                          // Get.toNamed(Routes.LOGIN);
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
          
              SingleChildScrollView(
                scrollDirection: Axis.vertical ,
                child: Column(
                  children: [
                    controller.userType.value == 'patient' ?
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
                            controller: controller.userPhoneRegister.value,
                            keyboardType:TextInputType.number,
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
                            controller: controller.userAgeRegister.value,
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
                                groupValue: controller.selectedGenderUser.value,
                                onChanged: (value) =>
                                controller.selectedGenderUser.value = value!,
                              ),
                            ),
                            Expanded(
                              child: RadioListTile(
                                title: Text('Female'),
                                value: 'Female',
                                groupValue: controller.selectedGenderUser.value,
                                onChanged: (value) =>
                                controller.selectedGenderUser.value = value!,
                              ),
                            ),
                          ],
                        ),
                        SizedBox(height: Get.height * 0.02),

                        //email
                        Container(
                          margin: EdgeInsets.symmetric(horizontal: 20.0),
                          child: TextFormField(
                            controller: controller.userEmailRegister.value,
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
                            controller: controller.userPasswordRegister.value,
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
                            controller: controller.doctorPhoneRegister.value,
                            keyboardType:TextInputType.number,
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
                            controller: controller.doctorAgeRegister.value,
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
          
                        //selectedGender
                        Row(
                          children: [
                            Expanded(
                              child: RadioListTile(
                                title: Text('Male'),
                                value: 'Male',
                                groupValue: controller.selectedGenderDoctor.value,
                                onChanged: (value) =>
                                controller.selectedGenderDoctor.value = value!,
                              ),
                            ),
                            Expanded(
                              child: RadioListTile(
                                title: Text('Female'),
                                value: 'Female',
                                groupValue: controller.selectedGenderDoctor.value,
                                onChanged: (value) =>
                                controller.selectedGenderDoctor.value = value!,
                              ),
                            ),
                          ],
                        ),
                        SizedBox(height: Get.height * 0.02),
          
                        //email
                        Container(
                          margin: EdgeInsets.symmetric(horizontal: 20.0),
                          child: TextFormField(
                            controller: controller.doctorEmailRegister.value,
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
          
                        //qualification
                        Container(
                          margin: EdgeInsets.symmetric(horizontal: 20.0),
                          child: TextFormField(
                            controller: controller.doctorQualificationRegister.value,
                            decoration: InputDecoration(
                              labelText: "Enter Qualification",
                              contentPadding: EdgeInsets.symmetric(horizontal: 20.0),
                              border: OutlineInputBorder(
                                borderSide: BorderSide(color: Colors.red),
                                borderRadius: BorderRadius.circular(8.0),
                              ),
                            ),
                          ),
                        ),
                        SizedBox(height: Get.height * 0.02),
          
                        //specialization
                        Container(
                          margin: EdgeInsets.symmetric(horizontal: 20.0),
                          child: TextFormField(
                            controller: controller.doctorSpecializationRegister.value,
                            decoration: InputDecoration(
                              labelText: "Enter Specialization",
                              contentPadding: EdgeInsets.symmetric(horizontal: 20.0),
                              border: OutlineInputBorder(
                                borderSide: BorderSide(color: Colors.red),
                                borderRadius: BorderRadius.circular(8.0),
                              ),
                            ),
                          ),
                        ),
                        SizedBox(height: Get.height * 0.02),
          
                        //licenseNumber
                        Container(
                          margin: EdgeInsets.symmetric(horizontal: 20.0),
                          child: TextFormField(
                            controller: controller.doctorLicenseNumberRegister.value,
                            decoration: InputDecoration(
                              labelText: "Enter LicenseNumber",
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
                            controller: controller.doctorPasswordRegister.value,
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
          
                        Container(
                          child: Center(
                            child: Obx(() {
                              return Column(
                                mainAxisAlignment: MainAxisAlignment.center,
                                children: <Widget>[
                                  controller.imageFile.value == null
                                      ? Text('No image selected.')
                                      : Image.file(
                                    controller.imageFile.value!,
                                    height: 150,
                                  ),
                                  SizedBox(height: 20),
                                  ElevatedButton(
                                    onPressed: controller.pickImage,
                                    child: Text('Pick Image'),
                                  ),
                                  SizedBox(height: 20),
                                  // controller.base64String.value.isEmpty
                                  //     ? Container()
                                  //     : Text(
                                  //   'Base64 String: \n${controller.base64String.value}',
                                  //   style: TextStyle(fontSize: 12),
                                  // ),
                                ],
                              );
                            }),
                          ),
                        )
          
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
                              if(controller.userType.value == 'patient'){
                                print("Register Details Patient");
                                print("userName :${controller.userNameRegister.value.text}");
                                print("userPhone :${controller.userPhoneRegister.value.text}");
                                print("userAge :${controller.userAgeRegister.value.text}");
                                print("Gender :${controller.selectedGenderUser.value}");
                                print("userEmail :${controller.userEmailRegister.value.text}");
                                print("userPassword :${controller.userPasswordRegister.value.text}");
                                controller.registerUser(
                                  controller.userNameRegister.value.text,
                                  controller.userPhoneRegister.value.text,
                                  controller.userAgeRegister.value.text,
                                  controller.selectedGenderUser.value,
                                  controller.userEmailRegister.value.text,
                                  controller.userPasswordRegister.value.text,
                                );
                              }
                              else{
                                print("Register Details Doctor");
                                print("doctorName :${controller.doctorNameRegister.value.text}");
                                print("doctorPhone :${controller.doctorPhoneRegister.value.text}");
                                print("doctorAge :${controller.doctorAgeRegister.value.text}");
                                print("Gender :${controller.selectedGenderDoctor.value}");
                                print("doctorEmail :${controller.doctorEmailRegister.value.text}");
                                print("doctorQualification :${controller.doctorQualificationRegister.value.text}");
                                print("doctorSpecialization :${controller.doctorSpecializationRegister.value.text}");
                                print("doctorLicenseNumber :${controller.doctorLicenseNumberRegister.value.text}");
                                print("doctorPassword :${controller.doctorPasswordRegister.value.text}");
                                controller.registerDoctor(
                                    controller.doctorNameRegister.value.text,
                                    controller.doctorPhoneRegister.value.text,
                                    controller.doctorAgeRegister.value.text,
                                    controller.selectedGenderDoctor.value,
                                    controller.doctorEmailRegister.value.text,
                                    controller.doctorQualificationRegister.value.text,
                                    controller.doctorSpecializationRegister.value.text,
                                    controller.doctorLicenseNumberRegister.value.text,
                                    controller.doctorPasswordRegister.value.text,
                                    controller.base64String.value
                                );
          
                              }
          
          
          
                            },
                            child: Text(
                              "Register",
                              style: TextStyle(color: AppColors.white),
                            ))
                      ],
                    ),
                    SizedBox(height: Get.height * 0.02),

                  ],
                ),
              )
          
          
            ],
          ),
        ),
      );
    });
  }

}

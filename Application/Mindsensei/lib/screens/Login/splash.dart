import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:get/get_core/src/get_main.dart';
import 'package:get/get_state_manager/src/simple/get_state.dart';
import 'package:get/get_state_manager/src/simple/get_view.dart';

import '../../constants/app_Colors.dart';
import '../../controller/loginController.dart';
import '../../routes/app_routes.dart';

class Splash extends GetView<LoginController> {

  @override
  Widget build(BuildContext context) {

    // Timer(Duration(seconds: 3), () {
    //   print("Yeah, this line is printed after 3 seconds");
    //   Get.offAllNamed(Routes.FILTERPAGE);
    // });
    return Scaffold(
      backgroundColor: AppColors.colorPrimary,
      body: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        crossAxisAlignment: CrossAxisAlignment.center,
        children:[
          Container(
            margin: EdgeInsets.symmetric(vertical: Get.height * 0.3),
            child: Image.asset(
            "assets/images/mindsensei1.png",
            ),
          ),
        ]
      ),
    );
  }
}

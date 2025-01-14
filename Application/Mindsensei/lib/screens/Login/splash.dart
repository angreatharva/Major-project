import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:get/get_core/src/get_main.dart';
import 'package:get/get_state_manager/src/simple/get_state.dart';
import 'package:get/get_state_manager/src/simple/get_view.dart';

import '../../constants/app_colors.dart';
import '../../controller/loginController.dart';
import '../../routes/app_routes.dart';

class Splash extends GetView<LoginController> {

  double _initialSize = 100.0; // Initial size of the logo

  @override
  Widget build(BuildContext context) {

    // Timer(Duration(seconds: 3), () {
    //   print("Yeah, this line is printed after 3 seconds");
    //   Get.offAllNamed(Routes.FILTERPAGE);
    // });
    return Scaffold(
      backgroundColor: AppColors.colorPrimary,
      body: Center(
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 500),
          width: _initialSize,
          height: _initialSize,
          child: Hero(
            tag: 'logo',
            child: Image.asset(
              "assets/images/mindsensei1.png",
              fit: BoxFit.contain,
            ),
          ),
        ),
      ),
    );
  }
}

import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:get/get_state_manager/src/simple/get_view.dart';
import 'package:mindsensei/controller/doctorsController.dart';
import '../../commonWidgets/bottomNavigation/commonBottomBarController.dart';
import '../../commonWidgets/bottomNavigation/commonBottomBar.dart';
import '../../constants/appColors.dart';
import '../../controller/dashboardController.dart';
import '../../data/provider/apiProvider.dart';
import '../../data/repository/postRepository.dart';
import '../../routes/appRoutes.dart';
import 'package:http/http.dart' as http;


class Doctors extends GetView<DoctorsController> {
  const Doctors({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    if (!Get.isRegistered<DoctorsController>()) {
      Get.put(DoctorsController(repository: MyRepository(apiClient: MyApiClient(httpClient: http.Client()))));
    }
    return GetX<DoctorsController>(initState: (state) {
      print("Doctors page");
    }, builder: (context) {
      return Center(child: Container(child: Text(controller.dummyText.value + "Doctor Page"),));
    });
  }



}
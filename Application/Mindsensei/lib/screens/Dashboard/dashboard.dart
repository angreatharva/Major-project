import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:get/get_state_manager/src/simple/get_view.dart';
import 'package:mindsensei/screens/Blogs/blogs.dart';
import 'package:mindsensei/screens/Monitor/monitor.dart';
import '../../commonWidgets/bottomNavigation/commonBottomBarController.dart';
import '../../commonWidgets/bottomNavigation/commonBottomBar.dart';
import '../../constants/appColors.dart';
import '../../controller/dashboardController.dart';
import '../../routes/appRoutes.dart';
import '../Doctors/doctors.dart';
import '../Wellness/wellness.dart';

class Dashboard extends GetView<DashboardController> {
  const Dashboard({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return GetX<DashboardController>(initState: (state) {
      print("Dashboard page");
    }, builder: (context) {
      return Scaffold(
        appBar: AppBar(
          automaticallyImplyLeading: false,
          backgroundColor: AppColors.colorPrimary,
          title: Text(
            "Welcome, ${controller.box.read("Username")}${controller.dummyText.value}",
            style: TextStyle(color: AppColors.white),
          ),
          actions: [
            IconButton(
                onPressed: () {
                  Get.toNamed(Routes.LOGIN);
                },
                icon: Icon(Icons.power_settings_new_sharp))
          ],
        ),
        body: Container(
          height: Get.height * 0.894,
          child: _main(),
        ),
        bottomNavigationBar: CommonBottomNav(),
      );
    });
  }

  _main() {
    return Obx(() {
      switch (Get.find<BottomNavigationController>().selectedIndex.value) {
        case 0:
          print("Wellness SC");
          print(Get.find<BottomNavigationController>().selectedIndex.value);
          return Wellness();
        case 1:
          print("Doctors SC");
          print(Get.find<BottomNavigationController>().selectedIndex.value);
          return Doctors();
        case 2:
          print("Blogs SC");
          print(Get.find<BottomNavigationController>().selectedIndex.value);
          return Doctors();
          case 3:
          print("Blogs SC");
          print(Get.find<BottomNavigationController>().selectedIndex.value);
          return Monitor();
          case 4:
          print("Blogs SC");
          print(Get.find<BottomNavigationController>().selectedIndex.value);
          return Blogs();
        default:
          return Wellness();
      }
    });
  }



}
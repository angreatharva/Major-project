import 'dart:io';
import 'package:flutter/material.dart';
import 'package:get/get.dart';
import '../../constants/appColors.dart';
import '../../data/provider/apiProvider.dart';
import '../../data/repository/postRepository.dart';
import 'commonBottomBarController.dart';
import 'package:http/http.dart' as http;

class CommonBottomNav extends GetView<BottomNavigationController> {
  const CommonBottomNav({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    // Making sure controller is initialized
    if (!Get.isRegistered<BottomNavigationController>()) {
      Get.put(BottomNavigationController(repository: MyRepository(apiClient: MyApiClient(httpClient: http.Client()))));
    }

    return SizedBox(
      height: Platform.isIOS ? Get.height * 0.11 : Get.height * 0.09,
      child: Obx(
            () => BottomNavigationBar(
          iconSize: 22.0,
          items: [
            BottomNavigationBarItem(
              icon: Icon(
                Icons.home,
                color: AppColors.colorPrimary,
              ),
              label: 'Wellness'.tr,
              backgroundColor: Colors.grey[100],
            ),
            BottomNavigationBarItem(
              icon: Icon(
                Icons.home,
                color: AppColors.colorPrimary,
              ),
              label: 'Doctors'.tr,
              backgroundColor: Colors.grey[100],
            ),
            BottomNavigationBarItem(
              icon: Icon(
                Icons.person,
                color: AppColors.colorPrimary,
              ),
              label: 'Blogs'.tr,
              backgroundColor: Colors.grey[100],
            ),
          ],
          type: BottomNavigationBarType.fixed,
          selectedItemColor: AppColors.colorPrimary,
          currentIndex: controller.selectedIndex.value,
          onTap: controller.changeIndex,
          elevation: 5,
        ),
      ),
    );
  }
}
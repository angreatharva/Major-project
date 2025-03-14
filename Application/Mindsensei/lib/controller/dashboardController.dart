import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:flutter/widgets.dart';
import 'package:get/get.dart';
import 'package:get_storage/get_storage.dart';
import 'package:flutter_easyloading/flutter_easyloading.dart';
import 'package:mindsensei/controller/blogsController.dart';
import 'package:mindsensei/controller/doctorsController.dart';
import 'package:mindsensei/controller/wellnessController.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../data/model/teamsListModel.dart';
import '../data/repository/postRepository.dart';

class DashboardController extends GetxController with SingleGetTickerProviderMixin {
  final MyRepository repository;

  DashboardController({required this.repository}) : assert(repository != null);

  //if status is false then add other then remove..........

  var dummyText = ''.obs;
  late SharedPreferences prefs;
  GetStorage box = GetStorage();

  RxList<TeamsListModel> teamListMain = <TeamsListModel>[].obs;
  RxList<TeamsListModel> teamListTemp = <TeamsListModel>[].obs;
  Rx<TextEditingController> searchTeamsEditingController = TextEditingController().obs;
  ScrollController scrollController = new ScrollController();

  @override
  void onClose() {
    super.onClose();
    EasyLoading.dismiss();
  }

  @override
  void onInit() {
    super.onInit();
    print("DashboardController init");
    box = GetStorage();
    Get.lazyPut(() => WellnessController(repository: repository));
    Get.lazyPut(() => DoctorsController(repository: repository));
    Get.lazyPut(() => BlogsController(repository: repository));

    EasyLoading.dismiss();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.paused) {
      // App is being minimized
      // EasyLoading.dismiss();
    }
  }

}

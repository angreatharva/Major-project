import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:flutter/widgets.dart';
import 'package:get/get.dart';
import 'package:get_storage/get_storage.dart';
import 'package:flutter_easyloading/flutter_easyloading.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../constants/api_endpoints.dart';
import '../data/model/teamsListModel.dart';
import '../data/repository/post_repository.dart';
import '../routes/app_routes.dart';

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
    EasyLoading.dismiss();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.paused) {
      // App is being minimized
      // EasyLoading.dismiss();
    }
  }


  List<dynamic> convertAgeGroup(String ageGroupString) {
    // Decode the input JSON string
    List<dynamic> ageGroupData = json.decode(ageGroupString);

    // Modify the data as per your requirements
    for (var item in ageGroupData) {
      item['Age'] = int.parse(item['Age']);
    }

    return ageGroupData;
  }


  searchTeamList(String searchQuery) {
    print('searchTeamList: ${searchQuery}');
    List<TeamsListModel> filteredList = [];
    if (searchQuery.isEmpty) {
      // If the search query is empty, return the original full list
      filteredList = teamListMain.value;
    } else {
      // Convert searchQuery to lower case for case-insensitive search
      String query = searchQuery.toLowerCase();
      // Filter the list based on the search query
      filteredList = teamListMain.value.where((item) {
        return
          item.teamName.toString().toLowerCase().contains(query) ||
          item.coachName.toString().toLowerCase().contains(query)

        ;
      }).toList();
    }
    // Update the list to display in the UI
    teamListTemp.value = filteredList;
  }

}

import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:get/get_state_manager/src/simple/get_view.dart';
import '../../controller/wellnessController.dart';
import '../../data/provider/apiProvider.dart';
import '../../data/repository/postRepository.dart';
import 'package:http/http.dart' as http;


class Wellness extends GetView<WellnessController> {
  const Wellness({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    if (!Get.isRegistered<WellnessController>()) {
      Get.put(WellnessController(repository: MyRepository(apiClient: MyApiClient(httpClient: http.Client()))));
    }
    return GetX<WellnessController>(initState: (state) {
      print("Wellness page");
    }, builder: (context) {
      return Center(child: Container(child: Text(controller.dummyText.value + "Wellness Page"),));
    });
  }


}
import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:get/get_state_manager/src/simple/get_view.dart';
import '../../controller/blogsController.dart';
import '../../data/provider/apiProvider.dart';
import '../../data/repository/postRepository.dart';
import 'package:http/http.dart' as http;


class Blogs extends GetView<BlogsController> {
  const Blogs({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    if (!Get.isRegistered<BlogsController>()) {
      Get.put(BlogsController(repository: MyRepository(apiClient: MyApiClient(httpClient: http.Client()))));
    }
    return GetX<BlogsController>(initState: (state) {
      print("Blogs page");
    }, builder: (context) {
      return Center(child: Container(child: Text(controller.dummyText.value + "Blogs Page"),));
    });
  }


}
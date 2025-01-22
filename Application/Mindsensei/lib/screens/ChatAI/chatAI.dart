import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:get/get_state_manager/src/simple/get_view.dart';
import 'package:mindsensei/controller/chatAIController.dart';
import '../../controller/blogsController.dart';
import '../../data/provider/apiProvider.dart';
import '../../data/repository/postRepository.dart';
import 'package:http/http.dart' as http;


class ChatAI extends GetView<ChatAIController> {
  const ChatAI({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    if (!Get.isRegistered<ChatAIController>()) {
      Get.put(ChatAIController(repository: MyRepository(apiClient: MyApiClient(httpClient: http.Client()))));
    }
    return GetX<ChatAIController>(initState: (state) {
      print("ChatAI page");
    }, builder: (context) {
      return Center(child: Container(child: Text(controller.dummyText.value + "ChatAI Page"),));
    });
  }


}
import 'package:get/get.dart';

import '../../data/repository/postRepository.dart';

class BottomNavigationController extends GetxController {

  final MyRepository repository;

  BottomNavigationController({required this.repository}) : assert(repository != null);
  var selectedIndex = 0.obs;

  void changeIndex(int index) {
    selectedIndex.value = index;
  }
}
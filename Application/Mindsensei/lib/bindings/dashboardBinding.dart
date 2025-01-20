import 'package:get/get.dart';
import '../commonWidgets/bottomNavigation/commonBottomBarController.dart';
import '../controller/dashboardController.dart';
import '../data/provider/apiProvider.dart';
import '../data/repository/postRepository.dart';
import 'package:http/http.dart' as http;

class DashboardBinding implements Bindings {
  @override
  void dependencies() {
    Get.lazyPut(() => DashboardController(
        repository:
        MyRepository(apiClient: MyApiClient(httpClient: http.Client())))
    );
    Get.lazyPut(() => BottomNavigationController(
        repository:
        MyRepository(apiClient: MyApiClient(httpClient: http.Client())))
    );
  }
}

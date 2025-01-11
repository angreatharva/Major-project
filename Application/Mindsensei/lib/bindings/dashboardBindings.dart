import 'package:get/get.dart';
import '../controller/dashboardController.dart';
import '../data/provider/api.dart';
import '../data/repository/post_repository.dart';
import 'package:http/http.dart' as http;

class DashboardBinding implements Bindings {
  @override
  void dependencies() {
    Get.lazyPut(() => DashboardController(
        repository:
        MyRepository(apiClient: MyApiClient(httpClient: http.Client()))));
  }
}

import 'package:get/get.dart';
import '../controller/loginController.dart';
import '../data/provider/apiProvider.dart';
import '../data/repository/postRepository.dart';
import 'package:http/http.dart' as http;

class LoginBinding implements Bindings {
  @override
  void dependencies() {
    Get.lazyPut(() => LoginController(
        repository:
        MyRepository(apiClient: MyApiClient(httpClient: http.Client()))));
  }
}

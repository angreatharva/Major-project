import 'package:get/get.dart';
import '../controller/doctorsController.dart';
import '../data/provider/apiProvider.dart';
import '../data/repository/postRepository.dart';
import 'package:http/http.dart' as http;

class DoctorsBinding implements Bindings {
  @override
  void dependencies() {
    Get.lazyPut(() => DoctorsController(
        repository:
        MyRepository(apiClient: MyApiClient(httpClient: http.Client()))));
  }
}

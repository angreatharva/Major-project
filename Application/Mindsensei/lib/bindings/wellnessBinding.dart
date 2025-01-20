import 'package:get/get.dart';
import '../controller/wellnessController.dart';
import '../data/provider/apiProvider.dart';
import '../data/repository/postRepository.dart';
import 'package:http/http.dart' as http;

class WellnessBinding implements Bindings {
  @override
  void dependencies() {
    Get.lazyPut(() => WellnessController(
        repository:
        MyRepository(apiClient: MyApiClient(httpClient: http.Client())))
    );

  }
}

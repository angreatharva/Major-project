import 'package:get/get.dart';
import '../controller/blogsController.dart';
import '../data/provider/apiProvider.dart';
import '../data/repository/postRepository.dart';
import 'package:http/http.dart' as http;

class BlogsBinding implements Bindings {
  @override
  void dependencies() {
    Get.lazyPut(() => BlogsController(
        repository:
        MyRepository(apiClient: MyApiClient(httpClient: http.Client())))
    );

  }
}

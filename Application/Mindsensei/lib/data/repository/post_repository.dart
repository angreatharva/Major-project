import '../provider/api_provider.dart';

class MyRepository {
  final MyApiClient apiClient;

  MyRepository({required this.apiClient}) : assert(apiClient != null);

  registerUser(mapData) {
    return apiClient.registerUser(mapData);
  }
  registerDoctor(mapData) {
    return apiClient.registerDoctor(mapData);
  }

  loginUser(userName, password) {
    return apiClient.loginUser(userName, password);
  }



}
import '../../network/utils/network_error.dart';
import 'package:dio/dio.dart';

enum NetworkCallConnectionStatus {
  inProgress,
  completedSuccessfully,
  failed,
  aborted
}

typedef OnNetworkCallProgress = Function();
typedef OnNetworkCallSuccess = Function<T>(T);
typedef OnNetworkCallCancelled = Function();
typedef OnNetworkCallFailed = Function(NetworkError);
typedef OnValueChanged = Function();
typedef MockDataHandler = Future<Response> Function(RequestOptions options);


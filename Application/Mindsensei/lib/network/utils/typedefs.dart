import 'package:dio/dio.dart';
import 'network_error.dart';

/// Represents the current network call status.
enum NetworkCallConnectionStatus {
  inProgress,
  completedSuccessfully,
  failed,
  aborted,
}

/// Typedefs for callbacks during network operations.
typedef OnNetworkCallProgress = void Function();
typedef OnNetworkCallSuccess<T> = void Function(T data);
typedef OnNetworkCallCancelled = void Function();
typedef OnNetworkCallFailed = void Function(NetworkError error);
typedef OnValueChanged = void Function();
typedef MockDataHandler = Future<Response> Function(RequestOptions options);

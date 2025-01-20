class NetworkError {
  final String message;
  final int? errorCode;

  NetworkError(this.message, {this.errorCode});

  @override
  String toString() => 'NetworkError: $message (code: $errorCode)';
}

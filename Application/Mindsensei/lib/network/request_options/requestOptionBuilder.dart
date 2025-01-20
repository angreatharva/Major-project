import 'package:dio/dio.dart';

class RequestOptionsBuilder {
  late Options? _options;

  RequestOptionsBuilder({Options? options}) {
    _options = options ?? Options();
  }

  Options? build() {
    return _options;
  }
}

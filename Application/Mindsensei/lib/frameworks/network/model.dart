import 'package:flutter/material.dart';

import 'index.dart';

class Model {
  ValueNotifier<NetworkCallConnectionStatus?> _connectionStatusLiveData =
      ValueNotifier(null);

  ValueNotifier<NetworkCallConnectionStatus?> get connectionStatusLiveData =>
      _connectionStatusLiveData;

  NetworkCallConnectionStatus get connectionStatus =>
      _connectionStatusLiveData.value!;

  set connectionStatus(NetworkCallConnectionStatus status) {
    _connectionStatusLiveData.value = status;
  }
}

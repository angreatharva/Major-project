
import 'package:flutter/cupertino.dart';
import 'package:get/get.dart';

import '../bindings/dashboardBindings.dart';
import '../bindings/loginBinding.dart';
import '../screens/Dashboard/dashboard.dart';
import '../screens/Login/login.dart';
import '../screens/Login/register.dart';
import '../screens/Login/splash.dart';
import 'app_routes.dart';


class AppPages {
  static final pages = [
    GetPage(
      name: Routes.SPLASH,
      page: () => Splash(),
    ),
    GetPage(
        name: Routes.LOGIN,
        page: () => Login(GlobalKey<NavigatorState>()),
        binding: LoginBinding(),
    ),
    GetPage(
      name: Routes.REGISTER,
      page: () => Register(GlobalKey<NavigatorState>()),
      binding: LoginBinding(),
    ),
    GetPage(
      name: Routes.DASHBOARD,
      page: () => const Dashboard(),
      binding: DashboardBinding(),
    ),
  ];
}

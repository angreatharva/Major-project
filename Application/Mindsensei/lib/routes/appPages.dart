
import 'package:flutter/cupertino.dart';
import 'package:get/get.dart';
import 'package:mindsensei/bindings/doctorsBinding.dart';
import 'package:mindsensei/bindings/monitorBinding.dart';
import 'package:mindsensei/bindings/wellnessBinding.dart';
import 'package:mindsensei/screens/Blogs/blogs.dart';
import 'package:mindsensei/screens/Doctors/doctors.dart';
import 'package:mindsensei/screens/Monitor/monitor.dart';
import 'package:mindsensei/screens/Wellness/wellness.dart';
import '../bindings/blogsBinding.dart';
import '../bindings/dashboardBinding.dart';
import '../bindings/loginBinding.dart';
import '../screens/ChatAI/chatAI.dart';
import '../screens/Dashboard/dashboard.dart';
import '../screens/Login/login.dart';
import '../screens/Login/register.dart';
import '../screens/Login/splash.dart';
import 'appRoutes.dart';


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
    GetPage(
      name: Routes.WELLNESS,
      page: () => const Wellness(),
      binding: WellnessBinding(),
    ),
    GetPage(
      name: Routes.DOCTORS,
      page: () => const Doctors(),
      binding: DoctorsBinding(),
    ),
    GetPage(
      name: Routes.CHATAI,
      page: () => const ChatAI(),
      binding: DashboardBinding(),
    ),
    GetPage(
      name: Routes.MONITOR,
      page: () => const Monitor(),
      binding: MonitorBinding(),
    ),
    GetPage(
      name: Routes.BLOGS,
      page: () => const Blogs(),
      binding: BlogsBinding(),
    ),
  ];
}

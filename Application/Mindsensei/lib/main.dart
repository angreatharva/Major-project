import 'package:flutter/material.dart';
import 'package:flutter_easyloading/flutter_easyloading.dart';
import 'package:get/get.dart';
import 'package:get/get_navigation/src/root/get_material_app.dart';
import 'package:get/get_navigation/src/routes/transitions_type.dart';
import 'package:mindsensei/routes/appPages.dart';
import 'package:mindsensei/routes/appRoutes.dart';
import 'package:mindsensei/translation/appTranslation.dart';

void main() {
  // Initialize EasyLoading
  EasyLoading.init();

  runApp(GetMaterialApp(
    debugShowCheckedModeBanner: false,
    initialRoute: Routes.SPLASH, // Set the initial route to SPLASH
    defaultTransition: Transition.fade,
    getPages: AppPages.pages,
    builder: EasyLoading.init(), // Initialize EasyLoading builder
    home: SplashScreen(), // Set the home to your splash screen
    locale: Get.deviceLocale,
    fallbackLocale: const Locale('en', 'US'),
    translationsKeys: AppTranslation.translations,
    theme: ThemeData(fontFamily: 'SedanSC-Regular'),
  ));
}

class SplashScreen extends StatefulWidget {
  @override
  _SplashScreenState createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen> {
  bool _imagePrecached = false;
  final GlobalKey<NavigatorState> _navigatorKey = GlobalKey<NavigatorState>();

  double _initialSize = 100.0; // Initial size of the logo
  double _targetSize = 200.0; // Target size of the logo

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    if (!_imagePrecached) {
      precacheImage(const AssetImage("assets/images/mindsensei1.png"), context);
      _imagePrecached = true;
    }
  }

  @override
  void initState() {
    super.initState();
    // Start the animation after a short delay
    Future.delayed(const Duration(milliseconds: 100), () {
      setState(() {
        _initialSize = 200.0; // Change the initial size to start small
      });
    });
    // Navigate to the login screen after a longer delay
    Future.delayed(const Duration(milliseconds: 1800), () {
      Get.offNamed(Routes.LOGIN, arguments: _navigatorKey);
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: Center(
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 500),
          width: _initialSize,
          height: _initialSize,
          child: Hero(
            tag: 'logo',
            child: Image.asset(
              "assets/images/mindsensei1.png",
              fit: BoxFit.contain,
            ),
          ),
        ),
      ),
    );
  }
}

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:get/get.dart';
import 'package:flutter_easyloading/flutter_easyloading.dart';

import 'package:mindsensei/routes/app_pages.dart';
import 'package:mindsensei/routes/app_routes.dart';
import 'package:mindsensei/translation/app_Translation.dart';
import 'package:mindsensei/main.dart'; // Import your SplashScreen if needed.

void main() {
  testWidgets('Splash screen to login screen transition test', (WidgetTester tester) async {
    // Build the app with the GetMaterialApp configuration.
    await tester.pumpWidget(GetMaterialApp(
      debugShowCheckedModeBanner: false,
      initialRoute: Routes.SPLASH,
      defaultTransition: Transition.fade,
      getPages: AppPages.pages,
      builder: EasyLoading.init(),
      locale: Get.deviceLocale,
      fallbackLocale: const Locale('en', 'US'),
      translationsKeys: AppTranslation.translations,
      theme: ThemeData(fontFamily: 'SedanSC-Regular'),
    ));

    // Verify that the splash screen is displayed.
    expect(find.byType(SplashScreen), findsOneWidget);

    // Simulate the delay for the splash screen animation and transition.
    await tester.pumpAndSettle(const Duration(milliseconds: 1800));

    // Verify that the app navigates to the login screen.
    // Replace 'Login' with an identifier from your actual Login screen widget.
    expect(find.text('Login'), findsOneWidget);
  });
}

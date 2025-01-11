import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:get/get_state_manager/src/simple/get_view.dart';
import '../../constants/app_Colors.dart';
import '../../controller/dashboardController.dart';
import '../../routes/app_routes.dart';

class Dashboard extends GetView<DashboardController> {
  const Dashboard({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return GetX<DashboardController>(initState: (state) {
      print("Dashboard page");
    }, builder: (context) {
      return Scaffold(
        appBar: AppBar(
          automaticallyImplyLeading: false,
          backgroundColor: AppColors.orange,
          title: Text(
            "Welcome, ${controller.box.read("Username")}",
            style: TextStyle(color: AppColors.black),
          ),
          actions: [
            IconButton(
                onPressed: () {
                  Get.toNamed(Routes.LOGIN);
                },
                icon: Icon(Icons.power_settings_new_sharp))
          ],
        ),
        body: Container(
          height: Get.height * 0.894,
          child: _main(controller.teamListTemp.value),
        ),
      );
    });
  }

  _main(data) {
    return Stack(
      children: [
        _search(controller, data),
        Align(
          alignment: Alignment.center,
          child: controller.teamListTemp.isNotEmpty == true
              ? _cardList(controller, data)
              : Center(
            child: Container(
              color: Colors.white,
              child: Text(
                "Data_not_found".tr,
              ),
            ),
          ),
        ),
      ],
    );
  }

  //  Search
  _search(controller, data) {
    return Container(
      height: Get.height * 0.045,
      margin: EdgeInsets.symmetric(
          vertical: Get.height * 0.012, horizontal: Get.width * 0.02),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(8.0),
      ),
      child: Obx(
            () => TextFormField(
          controller: controller.searchTeamsEditingController.value,
          decoration: InputDecoration(
            suffixIcon: controller.searchTeamsEditingController.value.text.length >
                0
                ? GestureDetector(
              onTap: () {
                controller.searchTeamsEditingController.value.text = '';
                controller.getTeamList();
              },
              child: Icon(Icons.cancel, color: AppColors.grey),
            )
                : GestureDetector(
              onTap: () {
                controller.searchTeamList(
                    controller.searchTeamsEditingController.value.text);
              },
              child: Icon(Icons.search, color: AppColors.grey),
            ),
            hintText: 'Search'.tr,
            hintStyle: TextStyle(color: Colors.grey),
            contentPadding: EdgeInsets.symmetric(
                vertical: Get.height * 0.008, horizontal: Get.width * 0.03),
            border: OutlineInputBorder(
              borderRadius: BorderRadius.circular(5.0),
            ),
          ),
          onChanged: (value) {
            if (value.length > 0) {
              controller.searchTeamList(value);
            } else {
              controller.getTeamList();
            }
          },
          keyboardType: TextInputType.text,
          textInputAction: TextInputAction.done,
        ),
      ),
    );
  }

  _cardList(controller, list) {
    return Container(
      margin: EdgeInsets.only(top: Get.height * 0.07),
      width: double.infinity,
      child: ListView.builder(
        padding: EdgeInsets.zero,
        controller: controller.scrollController,
        itemCount: list.length,
        itemBuilder: (context, index) {
          return _unitCard(controller, index, list[index]);
        },
      ),
    );
  }

  _unitCard(controller, index, data) {
    final team = controller.teamListTemp[index];
    return GestureDetector(
      onTap: () {
        controller.box.write("teamId", data.teamId);
        controller.box.write("ageGroup", controller.box.read("ageGroup"));
        controller.box.write("gender", data.gender);
        controller.box.write("teamName", data.teamName);
      },
      child: Card(
        elevation: 5,
        child: Container(
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(5),
            border: Border.all(color: Colors.grey),
          ),
          padding: EdgeInsets.only(
              left: Get.width * 0.02,
              right: Get.width * 0.02,
              top: Get.height * 0.008,
              bottom: Get.height * 0.008),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  Container(
                    child: Text(
                      data.teamName,
                      style: TextStyle(
                        fontWeight: FontWeight.bold,
                        fontSize: 30,
                      ),
                    ),
                  ),
                  Container(
                    margin: EdgeInsets.symmetric(horizontal: Get.width * 0.2),
                    child: Text(
                      data.coachName + " (" + "Coach".tr + ")",
                      style: TextStyle(
                        fontSize: 30,
                      ),
                    ),
                  ),
                ],
              ),

              Container(
                  height: Get.height * 0.065,
                  child: GridView.builder(
                      gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(
                        crossAxisSpacing: Get.width * 0.01,
                        crossAxisCount: 6,
                      ),
                      itemCount: 6,
                      itemBuilder: (context, index) {
                        return Column(
                          mainAxisAlignment: MainAxisAlignment.start,
                          children: [
                            Container(
                              height: 50,
                                width: Get.width * 0.2,
                                color: Colors.red,
                                alignment: Alignment.center,
                                child: Text(
                                  'ATHARVA ANGRE',
                                  textAlign: TextAlign.center,
                                  style: TextStyle(
                                    fontSize: 20
                                  ),
                                  maxLines: 2,
                                )
                            ),
                            Container(
                              // height: 50,
                                width: Get.width * 0.2,
                                color: Colors.green,
                                alignment: Alignment.center,
                                child: Text('10.00',textAlign: TextAlign.center,)
                            ),
                          ],
                        );
                      }
                  )
              )

            ],
          ),
        ),
      ),
    );
  }
}
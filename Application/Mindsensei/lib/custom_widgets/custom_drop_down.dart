import 'package:flutter/material.dart';
import '../data_model/dropDown_model.dart';

typedef SelectedData = Function(String value);

class CustomDropDown extends StatefulWidget {
  late String hintText;
  late List<DropdownList> dropdownList;
  late SelectedData? selectedData;
  late String? value;
  late String? label;
  late String? disabledHint;
  late TextStyle? labelStyle;
  late bool? isRequired;
  late bool? isDisabled;
  late bool? isContainsImage;
  late VoidCallback? onTap;

  CustomDropDown(
      {required this.dropdownList,
        this.hintText = '',
        this.selectedData,
        this.value = '',
        this.label = '',
        this.labelStyle,
        this.isRequired = false,
        this.isContainsImage = false,
        this.isDisabled = false,
        this.onTap,
        this.disabledHint = ""});

  @override
  _CustomDropDownState createState() => _CustomDropDownState();
}

class _CustomDropDownState extends State<CustomDropDown> {
  @override
  Widget build(BuildContext context) {
    return Column(
      children: <Widget>[
        widget.label != ''
            ? Container(
          padding: const EdgeInsets.only(bottom: 0.0),
          height: MediaQuery.of(context).size.height * 0.018,
          child: Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: <Widget>[
              Row(
                children: <Widget>[
                  Container(
                    alignment: Alignment.bottomLeft,
//                        padding: EdgeInsets.only(bottom: 5),
                    child: Text(
                      widget.label!,
                      style: widget.labelStyle,
                    ),
                  ),
                  widget.isRequired!
                      ? Text(
                    ' * ',
                    style: TextStyle(
                      color: Colors.red,
                      fontFamily: 'Arial',
                    ),
                  )
                      : Container(),
                ],
              ),
              widget.isContainsImage!
                  ? GestureDetector(
                onTap: widget.onTap,
                child: Image.asset('images/add_blue.png'),
              )
                  : Container(),
            ],
          ),
        )
            : Container(),
        Container(
          decoration: BoxDecoration(
            border: Border.all(
              color: Colors.grey,
            ),
            borderRadius: BorderRadius.all(Radius.circular(3.0)),
          ),
          child: Container(
            height: MediaQuery.of(context).size.height * 0.048,
            padding: EdgeInsets.symmetric(horizontal: 7),
            child: DropdownButton<String>(
              menuMaxHeight: widget.dropdownList.isNotEmpty && widget.dropdownList.length > 10 ? MediaQuery.of(context).size.height/2 : null,
              disabledHint: Text(
                  style: TextStyle(fontFamily: 'Arial'),
                  widget.disabledHint! != ""
                      ? widget.disabledHint!
                      : widget.hintText),
              items: widget.dropdownList.isNotEmpty
                  ? widget.dropdownList.map<DropdownMenuItem<String>>((data) {
                return DropdownMenuItem<String>(
                  value: data.code,
                  child: Text(
                    data.description.toString(),
                    style: TextStyle(fontFamily: 'Arial',fontSize: 16),
                  ),
                );
              }).toList()
                  : [],
              onChanged:widget.isDisabled! ? null :  (String? value) {
                widget.selectedData!(value!);
                setState(() {
                  widget.value = value;
                });
              },
              hint: Text(
                widget.hintText != null ? widget.hintText : '',
                style: TextStyle(fontFamily: 'Arial'),
              ),
              value: widget.value,
              isExpanded: true,
              underline: Container(
                decoration: BoxDecoration(
                  border: Border(bottom: BorderSide.none),
                ),
              ),
              style: TextStyle(
                  fontSize: 14.0,
                  color: Colors.black,
                  fontFamily: 'Arial'
              ),
            ),
          ),
        ),
      ],
    );
  }
}

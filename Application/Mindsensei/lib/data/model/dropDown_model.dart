class DropdownList {
  DropdownList(this.code, this.description);

  late String? description = '';
  late String? code = '';

  @override
  String toString() {
    return 'DropdownList{description: $description, code: $code}';
  }

  DropdownList.fromJson(Map<String, dynamic> json) {
    if (json == null) return;
    description = json["description"];
    code = json["code"].toString();
  }

  Map<String, dynamic> toJson() {
    final Map<String, dynamic> data = new Map<String, dynamic>();
    data['description'] = this.description;
    data['code'] = this.code;
    return data;
  }


  static List<DropdownList> listFromJson(List<dynamic> json) {
    return json == null
        ? <DropdownList>[]
        : json.map((value) => DropdownList.fromJson(value)).toList();
  }
}


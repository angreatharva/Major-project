class TeamsListModel {
  int? teamId;
  String? teamName;
  String? coachName;
  String? contactNo;
  String? emailAddress;
  int? gender;
  List<PlayerList>? playerList;

  TeamsListModel(
      {this.teamId,
        this.teamName,
        this.coachName,
        this.contactNo,
        this.emailAddress,
        this.gender,
        this.playerList});

  TeamsListModel.fromJson(Map<String, dynamic> json) {
    teamId = json['teamId'];
    teamName = json['teamName'];
    coachName = json['coachName'];
    contactNo = json['contactNo'];
    emailAddress = json['emailAddress'];
    gender = json['gender'];
    if (json['playerList'] != null) {
      playerList = <PlayerList>[];
      json['playerList'].forEach((v) {
        playerList!.add(new PlayerList.fromJson(v));
      });
    }
  }

  Map<String, dynamic> toJson() {
    final Map<String, dynamic> data = new Map<String, dynamic>();
    data['teamId'] = this.teamId;
    data['teamName'] = this.teamName;
    data['coachName'] = this.coachName;
    data['contactNo'] = this.contactNo;
    data['emailAddress'] = this.emailAddress;
    data['gender'] = this.gender;
    if (this.playerList != null) {
      data['playerList'] = this.playerList!.map((v) => v.toJson()).toList();
    }
    return data;
  }
}

class PlayerList {
  String? playerName;
  String? playerDob;
  int? playerAge;
  int? totalMarks;
  int? avgMarks;

  PlayerList(
      {this.playerName,
        this.playerDob,
        this.playerAge,
        this.totalMarks,
        this.avgMarks});

  PlayerList.fromJson(Map<String, dynamic> json) {
    playerName = json['playerName'];
    playerDob = json['playerDob'];
    playerAge = json['playerAge'];
    totalMarks = json['totalMarks'];
    avgMarks = json['avgMarks'];
  }

  Map<String, dynamic> toJson() {
    final Map<String, dynamic> data = new Map<String, dynamic>();
    data['playerName'] = this.playerName;
    data['playerDob'] = this.playerDob;
    data['playerAge'] = this.playerAge;
    data['totalMarks'] = this.totalMarks;
    data['avgMarks'] = this.avgMarks;
    return data;
  }
}

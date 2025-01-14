class TeamPlayerList {
  int? playerId;
  String? playerName;
  int? teamId;
  String? teamName;
  String? ageGroup;
  String? gender;
  int? totalMarks;
  int? avgMarks;

  TeamPlayerList(
      {this.playerId,
        this.playerName,
        this.teamId,
        this.teamName,
        this.ageGroup,
        this.gender,
        this.totalMarks,
        this.avgMarks});

  TeamPlayerList.fromJson(Map<String, dynamic> json) {
    playerId = json['playerId'];
    playerName = json['playerName'];
    teamId = json['teamId'];
    teamName = json['teamName'];
    ageGroup = json['ageGroup'];
    gender = json['gender'];
    totalMarks = json['totalMarks'];
    avgMarks = json['avgMarks'];
  }

  Map<String, dynamic> toJson() {
    final Map<String, dynamic> data = new Map<String, dynamic>();
    data['playerId'] = this.playerId;
    data['playerName'] = this.playerName;
    data['teamId'] = this.teamId;
    data['teamName'] = this.teamName;
    data['ageGroup'] = this.ageGroup;
    data['gender'] = this.gender;
    data['totalMarks'] = this.totalMarks;
    data['avgMarks'] = this.avgMarks;
    return data;
  }
}

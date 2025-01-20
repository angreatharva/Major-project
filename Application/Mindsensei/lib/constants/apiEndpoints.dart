class APIEndpoints {
  APIEndpoints();

  String get baseURL {
    // return "http://localhost:9000/";
    return "http://192.168.0.103:3000/";
  }

  String get registerUser{
    return "registerUser";
  }
  String get registerDoctor{
    return "registerDoctor";
  }
  String get login{
    return "login";
  }

}
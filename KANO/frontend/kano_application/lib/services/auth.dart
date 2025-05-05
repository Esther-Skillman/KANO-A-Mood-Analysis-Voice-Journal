// import 'package:firebase_auth/firebase_auth.dart';
// import 'package:flutter/foundation.dart';

// class AuthService with ChangeNotifier {
//   final FirebaseAuth _auth = FirebaseAuth.instance;
//   User? _user;

//   AuthService() {
//     _auth.authStateChanges().listen((User? user) {
//       _user = user;
//       notifyListeners();
//     });
//   }
// //
//   bool get isSignedIn => _user != null;

//   Future<String?> getToken() async {
//     return await _user?.getIdToken();
//   }

//   Future<void> signIn(String email, String password) async {
//     await _auth.signInWithEmailAndPassword(email: email, password: password);
//   }

//   Future<void> signUp(String email, String password) async {
//     await _auth.createUserWithEmailAndPassword(
//         email: email, password: password);
//   }

//   Future<void> signOut() async {
//     await _auth.signOut();
//   }
// }

import 'package:flutter/material.dart';
import 'home.dart';

class LoginScreen extends StatefulWidget {
  const LoginScreen({super.key});

  @override
  _LoginScreenState createState() => _LoginScreenState();
}

class _LoginScreenState extends State<LoginScreen> {
  final TextEditingController _unController = TextEditingController();
  final TextEditingController _psController = TextEditingController();
  final FocusNode _unFocus = FocusNode();
  final FocusNode _psFocus = FocusNode();

  @override
  void dispose() {
    _unController.dispose();
    _psController.dispose();
    _unFocus.dispose();
    _psFocus.dispose();
    super.dispose();
  }

  Future<void> _signIn(BuildContext context) async {
    final username = _unController.text.trim();
    final password = _psController.text.trim();

    if (username.isEmpty || password.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Please enter username and password')),
      );
      return;
    }

    // Credentials placeholder
    Navigator.pushReplacement(
      // pushReplacement so that the user cannot go back to the login screen
      context,
      MaterialPageRoute(builder: (context) => const HomeScreen()),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 40.0),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              // Logo text
              Text(
                'KANO',
                style: TextStyle(
                  fontSize: 36,
                  fontWeight: FontWeight.bold,
                  color: Colors.purple[200],
                ),
              ),
              const SizedBox(height: 60),

              // Username field
              TextField(
                controller: _unController,
                focusNode: _unFocus,
                decoration: InputDecoration(
                  labelText: 'Username',
                  border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(8.0),
                  ),
                  contentPadding: const EdgeInsets.symmetric(
                    vertical: 12.0,
                    horizontal: 16.0,
                  ),
                ),
                onSubmitted: (value) {
                  _unFocus.unfocus();
                  FocusScope.of(context).requestFocus(_psFocus);
                },
              ),
              const SizedBox(height: 16),

              // Password field
              TextField(
                controller: _psController,
                focusNode: _psFocus,
                decoration: InputDecoration(
                  labelText: 'Password',
                  border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(8.0),
                  ),
                  contentPadding: const EdgeInsets.symmetric(
                    vertical: 12.0,
                    horizontal: 16.0,
                  ),
                ),
                obscureText: true,
                onSubmitted: (value) async {
                  _psFocus.unfocus();
                  await _signIn(context);
                },
              ),
              const SizedBox(height: 24),

              // Login button
              SizedBox(
                width: double.infinity,
                child: ElevatedButton(
                  onPressed: () => _signIn(context),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.purple[200],
                    foregroundColor: Colors.white,
                    padding: const EdgeInsets.symmetric(vertical: 12),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(8.0),
                    ),
                  ),
                  child: const Text(
                    'LOGIN',
                    style: TextStyle(fontWeight: FontWeight.bold),
                  ),
                ),
              ),
              const SizedBox(height: 12),

              // Sign up button
              TextButton(
                // Placeholder for sign up action
                onPressed: () => _signIn(context),
                child: Text(
                  'SIGN UP',
                  style: TextStyle(
                    color: Colors.purple[200],
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

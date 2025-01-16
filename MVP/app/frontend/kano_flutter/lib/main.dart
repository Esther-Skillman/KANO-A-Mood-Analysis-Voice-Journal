import 'package:english_words/english_words.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return ChangeNotifierProvider(
      create: (context) => MyAppState(),
      child: MaterialApp(
        title: 'Namer App',
        theme: ThemeData(
          useMaterial3: true,
          colorScheme: ColorScheme.fromSeed(seedColor: Colors.blueGrey),
        ),
        home: MyHomePage(),
      ),
    );
  }
}

class MyAppState extends ChangeNotifier {
  var current = WordPair.random();
}

class MyHomePage extends StatelessWidget {
  const MyHomePage({super.key});

  @override
  Widget build(BuildContext context) {
    // var appState = context.watch<MyAppState>();

    return Scaffold(
      body: Center(
        child: Column(
          children: [
            Text('KANO - Mood Analysis Voice Journal'),
            // Text(appState.current.asLowerCase),
            ElevatedButton(
                onPressed: () {
                  print('Record icon pressed');
                },
                child: Icon(Icons.voice_chat)),
            ElevatedButton(
                onPressed: () {
                  print('Record icon pressed');
                },
                child: Icon(Icons.voice_chat)),
            ElevatedButton(
                onPressed: () {
                  print('Record icon pressed');
                },
                child: Icon(Icons.voice_chat)),
            ElevatedButton(
                onPressed: () {
                  print('Record icon pressed');
                },
                child: Icon(Icons.voice_chat)),
            ElevatedButton(
                onPressed: () {
                  print('Record icon pressed');
                },
                child: Icon(Icons.voice_chat)),
            ElevatedButton(
                onPressed: () {
                  print('Record icon pressed');
                },
                child: Icon(Icons.mic)),
          ],
        ),
      ),
    );
  }
}

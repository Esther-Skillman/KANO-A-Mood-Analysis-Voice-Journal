import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';
import 'package:http/http.dart' as http;
import 'package:pie_chart/pie_chart.dart';
import 'package:flutter/foundation.dart';
import 'dart:convert';
import 'dart:io';
// import 'package:firebase_core/firebase_core.dart';
// import 'firebase_options.dart';

// // ...

// await Firebase.initializeApp(
//     options: DefaultFirebaseOptions.currentPlatform,
// );
void main() {
  runApp(const MaterialApp(home: Home()));
}

class Home extends StatefulWidget {
  const Home({super.key});

  @override
  State<Home> createState() => _HomeState();
}

class _HomeState extends State<Home> {
  bool isUploading = false;
  bool showChart = false; // Controls visibility of the PieChart
  Map<String, double> emotionData = {}; // Stores emotion percentages

  /// Function to pick and upload a file
  Future<void> pickFile() async {
    FilePickerResult? result = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: ['wav'],
    );

    if (result != null) {
      if (kIsWeb) {
        // Web: Use bytes
        Uint8List? fileBytes = result.files.first.bytes;
        String fileName = result.files.first.name;
        uploadFile(fileBytes, fileName);
        // Handle fileBytes (e.g., upload to server)
      } else {
        // Mobile/Desktop: Use path
        String? filePath = result.files.first.path;
        File file = File(filePath!);
        uploadFile(file);
        // Handle file operations
      }
    } else {
      print('No file selected');
    }

    // if (result != null && result.files.isNotEmpty) {
    //   File file = File(result.files.single.path!);
    //   uploadFile(file);
    // } else {
    //   print("No file selected");
    // }
  }

  /// Function to upload the file and get emotion data
  Future<void> uploadFile(dynamic file, [String? fileName]) async {
    setState(() {
      isUploading = true;
      showChart = false;
    });

    var url = Uri.parse(
        'https://kano-mood-analysis-870b6bb34a2d.herokuapp.com/predict_simple/');
    var request = http.MultipartRequest('POST', url);

    if (file is File) {
      // For mobile/desktop (File object)
      request.files.add(await http.MultipartFile.fromPath('file', file.path));
    } else if (file is Uint8List) {
      // For web (sending byte data)
      request.files
          .add(http.MultipartFile.fromBytes('file', file, filename: fileName));
    }

    try {
      var response = await request.send();
      if (response.statusCode == 200) {
        print("Request okay!");
        var responseBody = await response.stream.bytesToString();
        var jsonResponse = json.decode(responseBody);

        if (jsonResponse.containsKey("emotion_percentages")) {
          Map<String, dynamic> emotions = jsonResponse["emotion_percentages"];
          Map<String, double> parsedData = emotions
              .map((key, value) => MapEntry(key, (value as num).toDouble()));

          setState(() {
            emotionData = parsedData;
            showChart = true;
          });
        }
      } else {
        print("Failed to upload file. Status: ${response.statusCode}");
      }
    } catch (e) {
      print("Upload error: $e ");
    } finally {
      setState(() {
        isUploading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: isUploading
            ? CircularProgressIndicator()
            : Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  if (showChart && emotionData.isNotEmpty)
                    PieChart(
                      dataMap: emotionData,
                      animationDuration: Duration(milliseconds: 800),
                      chartLegendSpacing: 50,
                      chartRadius: MediaQuery.of(context).size.width / 3.2,
                      colorList: [
                        const Color.fromARGB(255, 116, 196, 233), //Fear
                        const Color.fromARGB(255, 212, 74, 137), //Anger
                        const Color.fromARGB(255, 248, 159, 242), //Happy
                        const Color.fromARGB(255, 185, 174, 237), //Sad
                        const Color.fromARGB(255, 248, 159, 242), //N/A
                        const Color.fromARGB(255, 248, 159, 242), //N/A
                        const Color.fromARGB(255, 248, 159, 242), //N/A
                      ],
                      initialAngleInDegree: 0,
                      chartType: ChartType.ring,
                      ringStrokeWidth: 25,
                      centerText: "Mood Analysis",
                      legendOptions: LegendOptions(
                        showLegendsInRow: true,
                        legendPosition: LegendPosition.bottom,
                        showLegends: true,
                        legendShape: BoxShape.circle,
                        legendTextStyle: TextStyle(
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      chartValuesOptions: ChartValuesOptions(
                        showChartValueBackground: true,
                        showChartValues: true,
                        showChartValuesInPercentage: true,
                        showChartValuesOutside: false,
                        decimalPlaces: 0,
                      ),
                    ),
                ],
              ),
      ),
      bottomNavigationBar: BottomAppBar(
        child: Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            ElevatedButton(
              onPressed: pickFile, // Trigger file picker and upload
              style: ElevatedButton.styleFrom(
                shape: CircleBorder(),
                padding: EdgeInsets.all(12),
              ),
              child: Icon(Icons.add, size: 36, color: Colors.black),
            ),
          ],
        ),
      ),
    );
  }
}

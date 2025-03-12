import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';
import 'package:http/http.dart' as http;
import 'package:pie_chart/pie_chart.dart';
import 'dart:convert';
import 'dart:io';

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

    if (result != null && result.files.single.path != null) {
      File file = File(result.files.single.path!);
      uploadFile(file);
    } else {
      print("No file selected");
    }
  }

  /// Function to upload the file and get emotion data
  Future<void> uploadFile(File file) async {
    setState(() {
      isUploading = true;
      showChart = false;
    });

    var url =
        Uri.parse('http://127.0.0.1:8000/predict_simple/'); // API endpoint
    var request = http.MultipartRequest('POST', url)
      ..files.add(await http.MultipartFile.fromPath('file', file.path));

    try {
      var response = await request.send();
      if (response.statusCode == 200) {
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
      print("Upload error: $e");
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
                      chartLegendSpacing: 32,
                      chartRadius: MediaQuery.of(context).size.width / 3.2,
                      colorList: [
                        Colors.blue,
                        Colors.green,
                        Colors.orange,
                        Colors.red,
                        Colors.purple
                      ],
                      initialAngleInDegree: 0,
                      chartType: ChartType.ring,
                      ringStrokeWidth: 32,
                      centerText: "Mood Analysis",
                      legendOptions: LegendOptions(
                        showLegendsInRow: false,
                        legendPosition: LegendPosition.right,
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
                        decimalPlaces: 1,
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

import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';
import 'package:http/http.dart' as http;
import 'package:pie_chart/pie_chart.dart';
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
  String? selectedFilePath;
  bool isUploading = false;
  bool showChart = false; // New variable to control PieChart visibility

  /// Function to pick a .wav file
  Future<void> pickFile() async {
    FilePickerResult? result = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: ['wav'], // Allow only .wav files
    );

    if (result != null && result.files.single.path != null) {
      setState(() {
        selectedFilePath = result.files.single.path!;
        showChart = false; // Reset chart visibility
      });

      // Show confirmation
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Selected: ${result.files.single.name}')),
      );

      // Upload the file
      uploadFile(File(selectedFilePath!));
    } else {
      print("No file selected");
    }
  }

  /// Function to upload a file to the endpoint
  Future<void> uploadFile(File file) async {
    setState(() {
      isUploading = true;
    });

    var url = Uri.parse('https://httpbin.org/post'); // Endpoint
    var request = http.MultipartRequest('POST', url)
      ..files.add(await http.MultipartFile.fromPath('file', file.path));

    try {
      var response = await request.send();
      if (response.statusCode == 200) {
        print("File uploaded successfully!");
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Upload successful!')),
        );

        setState(() {
          showChart = true; // Show PieChart after successful upload
        });
      } else {
        print("Failed to upload file. Status: ${response.statusCode}");
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
              content: Text('Upload failed! Error: ${response.statusCode}')),
        );
      }
    } catch (e) {
      print("Upload error: $e");
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Upload error: $e')),
      );
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
            : selectedFilePath == null
                ? Text("No file selected")
                : Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Text(
                        "Uploaded file: ${selectedFilePath!.split('/').last}",
                        style: TextStyle(
                            fontSize: 12, fontWeight: FontWeight.bold),
                      ),
                      SizedBox(height: 20),
                      if (showChart) // Only show PieChart after upload
                        PieChart(
                          dataMap: {
                            "Happiness": 40,
                            "Anger": 30,
                            "Disgust": 15,
                            "Surprise": 15,
                          },
                          animationDuration: Duration(milliseconds: 800),
                          chartLegendSpacing: 32,
                          chartRadius: MediaQuery.of(context).size.width / 3.2,
                          colorList: [
                            Colors.blue,
                            Colors.green,
                            Colors.orange,
                            Colors.red,
                          ],
                          initialAngleInDegree: 0,
                          chartType: ChartType.ring,
                          ringStrokeWidth: 32,
                          centerText: "MOOD ANALYSIS",
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
                            showChartValuesInPercentage: false,
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

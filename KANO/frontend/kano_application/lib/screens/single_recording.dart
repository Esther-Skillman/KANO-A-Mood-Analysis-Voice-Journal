import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:pie_chart/pie_chart.dart';
import 'package:audioplayers/audioplayers.dart';

class RecordingDetailsScreen extends StatefulWidget {
  final Map<String, dynamic> recording;

  const RecordingDetailsScreen({super.key, required this.recording});

  @override
  _RecordingDetailsScreenState createState() => _RecordingDetailsScreenState();
}

class _RecordingDetailsScreenState extends State<RecordingDetailsScreen> {
  final AudioPlayer _audioPlayer = AudioPlayer();
  bool isPlaying = false;

  @override
  void initState() {
    super.initState();
    // Set up audio player completion listener
    _audioPlayer.onPlayerComplete.listen((event) {
      setState(() {
        isPlaying = false;
      });
    });
  }

  @override
  void dispose() {
    _audioPlayer.dispose();
    super.dispose();
  }

  void _togglePlayback() async {
    if (isPlaying) {
      await _audioPlayer.pause();
      setState(() {
        isPlaying = false;
      });
    } else {
      await _audioPlayer.play(UrlSource(widget.recording["wav_url"]));
      setState(() {
        isPlaying = true;
      });
    }
  }

  void _confirmDelete() {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: Text("Delete Recording"),
        content: Text("Are you sure you want to delete this recording?"),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(),
            child: Text("Cancel"),
          ),
          TextButton(
            onPressed: () {
              Navigator.of(context).pop();
              _deleteRecording();
            },
            child: Text("Delete", style: TextStyle(color: Colors.red)),
          ),
        ],
      ),
    );
  }

  void _deleteRecording() async {
    final timestamp = widget.recording["timestamp"];
    final userId = "anonymous"; // Will be replaced with actua user ID in future

    final url = Uri.parse(
        "https://kano-mood-analysis-870b6bb34a2d.herokuapp.com/delete_recording/?user_id=$userId&timestamp=$timestamp");
    final response = await http.delete(url);

    if (response.statusCode == 200) {
      ScaffoldMessenger.of(context)
          .showSnackBar(SnackBar(content: Text("Recording deleted")));
      Navigator.of(context).pop();
    } else {
      ScaffoldMessenger.of(context)
          .showSnackBar(SnackBar(content: Text("Failed to delete recording")));
    }
  }

  @override
  Widget build(BuildContext context) {
    final emotionData = widget.recording["emotion_percentages"] != null
        ? Map<String, double>.from(widget.recording["emotion_percentages"]
            .map((k, v) => MapEntry(k, v.toDouble())))
        : {
            "Unkown": 100.0,
          };

    return Scaffold(
      backgroundColor: Colors.white,
      appBar: AppBar(
        backgroundColor: Colors.white,
        elevation: 0,
        title: Text(
          'KANO',
          style: TextStyle(
            color: Colors.purple[200],
            fontWeight: FontWeight.bold,
            fontSize: 24,
          ),
        ),
        actions: [
          Padding(
            padding: const EdgeInsets.only(right: 16.0),
            child: CircleAvatar(
              backgroundColor: Colors.purple[100],
              child: Icon(Icons.person, color: Colors.white),
            ),
          ),
        ],
        iconTheme: IconThemeData(color: Colors.black),
      ),
      body: SingleChildScrollView(
        padding: EdgeInsets.all(20.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.center,
          children: [
            // Static waveform image
            Container(
              height: 100,
              width: double.infinity,
              decoration: BoxDecoration(
                color: Colors.grey[200],
                borderRadius: BorderRadius.circular(15),
              ),
              padding: EdgeInsets.all(16),
              child: Image.asset(
                'assets/wavb.png',
                height: 80,
                fit: BoxFit.contain,
              ),
            ),
            SizedBox(height: 20),

            // Play button
            GestureDetector(
              onTap: _togglePlayback,
              child: Container(
                width: 60,
                height: 60,
                decoration: BoxDecoration(
                  color: Colors.grey[200],
                  shape: BoxShape.circle,
                ),
                child: Icon(
                  isPlaying ? Icons.pause : Icons.play_arrow,
                  size: 40,
                  color: Colors.purple[200],
                ),
              ),
            ),
            SizedBox(height: 30),

            // Pie chart for emotion analysis
            PieChart(
              dataMap: emotionData,
              animationDuration: Duration(milliseconds: 800),
              chartLegendSpacing: 20,
              chartRadius: MediaQuery.of(context).size.width / 2,
              colorList: [
                Color.fromARGB(255, 116, 196, 233), // Fear
                Color.fromARGB(255, 212, 74, 137), // Anger
                Color.fromARGB(255, 248, 159, 242), // Happy
                Color.fromARGB(255, 185, 174, 237), // Sad
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
                legendTextStyle: TextStyle(fontWeight: FontWeight.bold),
              ),
              chartValuesOptions: ChartValuesOptions(
                showChartValueBackground: false,
                showChartValues: true,
                showChartValuesInPercentage: true,
                showChartValuesOutside: false,
                decimalPlaces: 0,
              ),
            ),
            SizedBox(height: 30),

            // Transcript
            Container(
              width: double.infinity,
              padding: EdgeInsets.all(16),
              decoration: BoxDecoration(
                border: Border.all(color: Colors.grey[300]!),
                borderRadius: BorderRadius.circular(8),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    widget.recording["transcript"] ?? 'No transcript available',
                    style: TextStyle(fontSize: 16),
                  ),
                ],
              ),
            ),
            SizedBox(height: 30),
            GestureDetector(
              onTap: _confirmDelete,
              child: Container(
                width: 60,
                height: 60,
                decoration: BoxDecoration(
                  color: Colors.red[300],
                  shape: BoxShape.circle,
                ),
                child: Icon(
                  Icons.delete,
                  size: 30,
                  color: Colors.white,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

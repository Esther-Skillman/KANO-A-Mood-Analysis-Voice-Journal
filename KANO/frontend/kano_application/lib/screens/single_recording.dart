import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:pie_chart/pie_chart.dart';
import 'package:audioplayers/audioplayers.dart';
import 'package:fl_chart/fl_chart.dart' as fl;

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

  Widget _moodAudioGraph() {
    final segments = widget.recording['segments'] as List<dynamic>? ?? [];
    final double duration =
        widget.recording['duration']?.toDouble() ?? 1.0; // Fallback to 1s
    final emotions = [
      'Happy',
      'Sad',
      'Neutral',
      'Fear',
      'Anger',
      'Disgust',
      'Surprise'
    ];
    final Map<String, Color> emotionColors = {
      'Happy': Color(0xFF00ACC1).withOpacity(0.5),
      'Sad': Color(0xFF1976D2).withOpacity(0.5),
      'Neutral': Color.fromARGB(255, 65, 114, 135).withOpacity(0.5),
      'Fear': Color(0xFFC2185B).withOpacity(0.5),
      'Anger': Color(0xFF512DA8).withOpacity(0.5),
      'Disgust': Color(0xFFFFAB91).withOpacity(0.5),
      'Surprise': Color(0xFFFBC02D).withOpacity(0.5),
    };

    // Calculate segment positions and collect present emotions
    final List<Map<String, dynamic>> segmentPositions = [];
    final Set<String> presentEmotions = {};
    for (var i = 0; i < segments.length; i++) {
      final segment = segments[i];
      final timestamp = segment['timestamp'].toDouble();
      final emotion = emotions.contains(segment['emotion'])
          ? segment['emotion']
          : 'Neutral';
      final nextTimestamp = i < segments.length - 1
          ? segments[i + 1]['timestamp'].toDouble()
          : duration; // End at duration if last segment
      segmentPositions.add({
        'start': timestamp,
        'end': nextTimestamp,
        'emotion': emotion,
      });
      presentEmotions.add(emotion);
    }

    // Build legend for present emotions only
    Widget _buildLegend() {
      if (presentEmotions.isEmpty) {
        return SizedBox.shrink(); // No legend if no emotions
      }
      return Padding(
        padding: const EdgeInsets.only(top: 8.0),
        child: Wrap(
          spacing: 12.0,
          runSpacing: 8.0,
          alignment: WrapAlignment.center,
          children: presentEmotions.map((emotion) {
            return Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                Container(
                  width: 12,
                  height: 12,
                  decoration: BoxDecoration(
                    color: emotionColors[emotion],
                    shape: BoxShape.circle,
                  ),
                ),
                SizedBox(width: 4),
                Text(
                  emotion,
                  style: TextStyle(
                    fontSize: 12,
                    color: Colors.black87,
                    fontWeight: FontWeight.w500,
                  ),
                ),
              ],
            );
          }).toList(),
        ),
      );
    }

    return Container(
      padding: EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.grey[200],
        borderRadius: BorderRadius.circular(15),
      ),
      child: Column(
        children: [
          Container(
            height: 80, // Waveform height
            width: double.infinity,
            child: Stack(
              children: [
                Image.asset(
                  'assets/wavb.png',
                  height: 80,
                  width: double.infinity,
                  fit: BoxFit.contain,
                ),
                // Colored overlays for segments
                LayoutBuilder(
                  builder: (context, constraints) {
                    final widgetWidth = constraints.maxWidth;
                    return Stack(
                      children: segmentPositions.map((segment) {
                        final startTime = segment['start'] as double;
                        final endTime = segment['end'] as double;
                        final emotion = segment['emotion'] as String;

                        // Calculate position and width
                        final left = (startTime / duration) * widgetWidth;
                        final width =
                            ((endTime - startTime) / duration) * widgetWidth;

                        return Positioned(
                          left: left,
                          top: 0,
                          width: width,
                          height: 80, // Waveform height
                          child: Container(
                            color: emotionColors[emotion] ??
                                Colors.grey.withOpacity(0.4),
                          ),
                        );
                      }).toList(),
                    );
                  },
                ),
              ],
            ),
          ),
          _buildLegend(), // Legend with only present emotions
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final emotionData = widget.recording["emotion_percentages"] != null
        ? Map<String, double>.from(widget.recording["emotion_percentages"]
            .map((k, v) => MapEntry(k, v.toDouble())))
        : {
            "Unkown": 100.0,
          };

    // Colourblind-friendly colors for PieChart
    final Map<String, Color> emotionColors = {
      'Happy': Color(0xFF4DD0E1),
      'Sad': Color(0xFF42A5F5),
      'Neutral': Color.fromARGB(255, 65, 114, 135),
      'Fear': Color(0xFFEC407A),
      'Anger': Color(0xFFAB47BC),
      'Disgust': Color(0xFFFFAB91),
      'Surprise': Color(0xFFFFCA28),
    };

    final List<Color> colorList = emotionData.keys
        .map((emotion) => emotionColors[emotion] ?? Colors.grey)
        .toList();

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
          IconButton(
            icon: Icon(Icons.delete, color: Colors.red[300]),
            onPressed: _confirmDelete,
          ),
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
            SizedBox(height: 20),
            // Mood over time chart
            _moodAudioGraph(),
            SizedBox(height: 20),
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
              colorList: colorList,
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
            SizedBox(height: 10),
            Text(
              "*Emotional analysis may not reflect a true representation of the user's mood. User's should be aware this is not clinically validated.",
              style: TextStyle(
                fontStyle: FontStyle.italic,
                fontSize: 10,
                color: Colors.grey,
              ),
            ),
            SizedBox(height: 30),
            // Transcript
            Align(
              alignment: Alignment.center,
              child: Text(
                "Transcript",
                style: TextStyle(
                  fontSize: 20,
                  fontWeight: FontWeight.bold,
                  color: Colors.grey,
                ),
              ),
            ),
            SizedBox(height: 10),
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
          ],
        ),
      ),
    );
  }
}

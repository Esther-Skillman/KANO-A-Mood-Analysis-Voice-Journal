import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';
import 'package:audio_session/audio_session.dart';
import 'package:record/record.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:path_provider/path_provider.dart';
import 'dart:io';
import 'dart:async';
import '../services/api.dart';
import 'single_recording.dart';
import 'login.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  _HomeScreenState createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final AudioRecorder _recorder = AudioRecorder();
  StreamSubscription<RecordState>? _recordSub;
  RecordState _recordState = RecordState.stop;
  String? _recordedFilePath;
  DateTime? _recordingStartTime;
  bool isUploading = false;
  bool isLoading = false;
  String? currentDate;
  List<Map<String, dynamic>> recordings = [];

  @override
  void initState() {
    super.initState();
    _initAudioSession();
    _fetchRecordings();
    _setCurrentDate();
    _setupRecorder();
  }

  Future<void> _initAudioSession() async {
    final session = await AudioSession.instance;
    await session.configure(AudioSessionConfiguration(
      // av corresponds to libraries in iOS
      avAudioSessionCategory: AVAudioSessionCategory.playAndRecord,
      avAudioSessionCategoryOptions:
          AVAudioSessionCategoryOptions.allowBluetooth |
              AVAudioSessionCategoryOptions.defaultToSpeaker,
      androidAudioAttributes: const AndroidAudioAttributes(
        contentType: AndroidAudioContentType.speech,
        usage: AndroidAudioUsage.media,
      ),
      androidAudioFocusGainType: AndroidAudioFocusGainType
          .gainTransientMayDuck, // Lowers other app's audio during recording
    ));
  }

  Future<void> _setupRecorder() async {
    _recordSub = _recorder.onStateChanged().listen((state) {
      setState(() => _recordState = state);
    });
  }

  Future<bool> _checkPermissions() async {
    final status = await Permission.microphone.request();
    if (!status.isGranted) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Microphone permission required')),
      );
      return false;
    }
    return true;
  }

  void _setCurrentDate() {
    final now = DateTime.now();
    final months = [
      'JAN',
      'FEB',
      'MAR',
      'APR',
      'MAY',
      'JUN',
      'JUL',
      'AUG',
      'SEP',
      'OCT',
      'NOV',
      'DEC'
    ];
    final day = now.day;
    final month = months[now.month - 1];
    String suffix = _getDaySuffix(day);
    setState(() {
      currentDate = '${_getDayOfWeek(now.weekday)} $day$suffix $month';
    });
  }

  String _getDaySuffix(int day) {
    if (day >= 11 && day <= 13) return 'TH';
    switch (day % 10) {
      case 1:
        return 'ST';
      case 2:
        return 'ND';
      case 3:
        return 'RD';
      default:
        return 'TH';
    }
  }

  String _getDayOfWeek(int day) {
    const days = ['', 'MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN'];
    return days[day];
  }

  Future<void> _fetchRecordings() async {
    setState(() => isLoading = true);
    try {
      final fetchedRecordings = await ApiService.getRecordings();
      setState(() => recordings = fetchedRecordings);
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Failed to load recordings: $e')),
      );
    } finally {
      setState(() => isLoading = false);
    }
  }

  Future<void> _toggleRecording() async {
    if (_recordState == RecordState.record) {
      await _stopRecording();
    } else {
      await _startRecording();
    }
  }

  Future<void> _startRecording() async {
    if (!await _checkPermissions()) return;

    try {
      final directory = await getTemporaryDirectory();
      _recordedFilePath =
          '${directory.path}/recording_${DateTime.now().millisecondsSinceEpoch}.wav';

      await _recorder.start(
        const RecordConfig(
          encoder: AudioEncoder.wav,
          numChannels: 1,
        ),
        path: _recordedFilePath!,
      );

      setState(() {
        _recordingStartTime = DateTime.now();
      });
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Failed to start recording: $e')),
      );
    }
  }

  Future<void> _stopRecording() async {
    try {
      await _recorder.stop();

      final recordingDuration = DateTime.now().difference(_recordingStartTime!);
      if (recordingDuration.inSeconds < 1) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
              content: Text('Recording too short (1 second minimum)')),
        );
        return;
      }

      setState(() => isUploading = true);

      if (_recordedFilePath != null) {
        final file = File(_recordedFilePath!);
        final fileSize = await file.length();

        if (fileSize <= 44) {
          throw Exception('Recorded file is empty or invalid');
        }

        final response = await ApiService.processAudio(file);
        await _fetchRecordings();

        if (mounted) {
          Navigator.push(
            context,
            MaterialPageRoute(
              builder: (context) => RecordingDetailsScreen(recording: response),
            ),
          );
        }
      }
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Failed to process recording: $e')),
      );
    } finally {
      setState(() => isUploading = false);
    }
  }

  // Obsoloete
  Future<void> _pickAndUploadFile() async {
    setState(() => isUploading = true);
    try {
      FilePickerResult? result = await FilePicker.platform.pickFiles(
        type: FileType.custom,
        allowedExtensions: ['wav'],
      );

      if (result != null) {
        final file = File(result.files.single.path!);
        final response = await ApiService.processAudio(file);
        await _fetchRecordings();

        if (mounted) {
          Navigator.push(
            context,
            MaterialPageRoute(
              builder: (context) => RecordingDetailsScreen(recording: response),
            ),
          );
        }
      }
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Upload failed: $e')),
      );
    } finally {
      setState(() => isUploading = false);
    }
  }

  String _formatDuration(double seconds) {
    final minutes = (seconds / 60).floor();
    final remainingSeconds = (seconds % 60).round();
    return '${minutes.toString().padLeft(2, '0')}:${remainingSeconds.toString().padLeft(2, '0')}';
  }

  @override
  void dispose() {
    _recordSub?.cancel();
    _recorder.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      appBar: AppBar(
        backgroundColor: Colors.white,
        elevation: 0,
        title: GestureDetector(
          onTap: () async {
            await _fetchRecordings(); // Refresh recordings when KANO is clicked (add refresh drag down later?)
          },
          child: Text(
            'KANO',
            style: TextStyle(
              color: Colors.purple[200],
              fontWeight: FontWeight.bold,
              fontSize: 24,
            ),
          ),
        ),
        actions: [
          IconButton(
            icon: const Icon(Icons.delete_forever, color: Colors.black),
            onPressed: () {
              showDialog(
                context: context,
                builder: (context) => AlertDialog(
                  title: const Text('Delete All Recordings'),
                  content: const Text(
                      'Are you sure you want to delete all recordings? This action cannot be undone.'),
                  actions: [
                    TextButton(
                      onPressed: () => Navigator.pop(context),
                      child: const Text('Cancel'),
                    ),
                    TextButton(
                      onPressed: () async {
                        Navigator.pop(context);
                        try {
                          await ApiService.deleteAll();
                          await _fetchRecordings();
                          ScaffoldMessenger.of(context).showSnackBar(
                            const SnackBar(
                                content: Text(
                                    'All recordings deleted successfully')),
                          );
                        } catch (e) {
                          ScaffoldMessenger.of(context).showSnackBar(
                            SnackBar(
                                content:
                                    Text('Failed to delete recordings: $e')),
                          );
                        }
                      },
                      child: const Text('Delete'),
                    ),
                  ],
                ),
              );
            },
          ),
          Padding(
            padding: const EdgeInsets.only(right: 16.0),
            child: GestureDetector(
              onTap: () {
                // Placeholder to navigate back to Login screen
                Navigator.pushReplacement(
                    context,
                    MaterialPageRoute(
                        builder: (context) => const LoginScreen()));
              },
              child: CircleAvatar(
                backgroundColor: Colors.purple[100],
                child: Icon(Icons.person, color: Colors.white),
              ),
            ),
          ),
        ],
      ),
      body: Column(
        children: [
          Container(
            padding:
                const EdgeInsets.symmetric(horizontal: 20.0, vertical: 18.0),
            alignment: Alignment.center,
            child: Text(
              currentDate ?? '',
              style: TextStyle(
                color: Colors.purple[200],
                fontWeight: FontWeight.bold,
                fontSize: 40,
              ),
            ),
          ),
          Expanded(
            child: isLoading
                ? const Center(child: CircularProgressIndicator())
                : recordings.isEmpty
                    ? const Center(child: Text('No recordings yet'))
                    : ListView.builder(
                        itemCount: recordings.length,
                        itemBuilder: (context, index) {
                          final recording = recordings[index];
                          final duration =
                              recording["duration"]?.toDouble() ?? 0.0;
                          final durationDisplay = _formatDuration(duration);

                          return Padding(
                            padding: const EdgeInsets.symmetric(
                              horizontal: 16.0,
                              vertical: 8.0,
                            ),
                            child: InkWell(
                              onTap: () {
                                Navigator.push(
                                  context,
                                  MaterialPageRoute(
                                    builder: (context) =>
                                        RecordingDetailsScreen(
                                      recording: recording,
                                    ),
                                  ),
                                );
                              },
                              child: Container(
                                decoration: BoxDecoration(
                                  gradient: LinearGradient(
                                    colors: [
                                      Colors.purple[100]!,
                                      Colors.blue[100]!,
                                    ],
                                    begin: Alignment.centerLeft,
                                    end: Alignment.centerRight,
                                  ),
                                  borderRadius: BorderRadius.circular(15),
                                ),
                                padding: const EdgeInsets.all(12.0),
                                child: Row(
                                  mainAxisAlignment:
                                      MainAxisAlignment.spaceBetween,
                                  children: [
                                    Expanded(
                                      child: Image.asset(
                                        'assets/wav.png',
                                        height: 60,
                                        fit: BoxFit.contain,
                                      ),
                                    ),
                                    Container(
                                      padding: const EdgeInsets.symmetric(
                                        horizontal: 8.0,
                                        vertical: 4.0,
                                      ),
                                      decoration: BoxDecoration(
                                        color: Colors.white.withOpacity(0.3),
                                        borderRadius: BorderRadius.circular(8),
                                      ),
                                      child: Text(
                                        durationDisplay,
                                        style: const TextStyle(
                                          color: Colors.white,
                                          fontWeight: FontWeight.bold,
                                        ),
                                      ),
                                    ),
                                  ],
                                ),
                              ),
                            ),
                          );
                        },
                      ),
          ),
        ],
      ),
      floatingActionButton: Container(
        margin: const EdgeInsets.only(bottom: 16.0),
        child: FloatingActionButton(
          onPressed: isUploading ? null : _toggleRecording,
          backgroundColor: Colors.purple[100],
          child: isUploading
              ? const CircularProgressIndicator(color: Colors.white)
              : Icon(
                  _recordState == RecordState.record ? Icons.stop : Icons.mic,
                  color: Colors.white,
                  size: 32,
                ),
        ),
      ),
      floatingActionButtonLocation: FloatingActionButtonLocation.centerDocked,
    );
  }
}

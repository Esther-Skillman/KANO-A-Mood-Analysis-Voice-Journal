import 'package:flutter_sound/flutter_sound.dart';
import 'package:path_provider/path_provider.dart';

class AudioService {
  final FlutterSoundRecorder _recorder = FlutterSoundRecorder();
  bool _isInitialized = false;

  Future<void> _initialize() async {
    if (!_isInitialized) {
      await _recorder.openRecorder();
      _isInitialized = true;
    }
  }

  Future<void> startRecording() async {
    await _initialize();
    final directory = await getTemporaryDirectory();
    final path = '${directory.path}/recording.wav';
    await _recorder.startRecorder(
      toFile: path,
      codec: Codec.pcm16WAV,
    );
  }

  Future<String?> stopRecording() async {
    final path = await _recorder.stopRecorder();
    await _recorder.closeRecorder();
    _isInitialized = false;
    return path;
  }
}

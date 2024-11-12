[**Datasets - Papers With Code**](https://paperswithcode.com/datasets?task=speech-emotion-recognition&mod=audio)

| Emotion      | CREMA-D | RAVDESS | LSSED |
| ------------ | ------- | ------- | ----- |
| Angry        | ✓       | ✓       | ✓     |
| Disgust      | ✓       | ✓       | ✓     |
| Fear         | ✓       | ✓       | ✓     |
| Happy        | ✓       | ✓       | ✓     |
| Neutral      | ✓       | ✓       | ✓     |
| Sad          | ✓       | ✓       | ✓     |
| Calm         |         | ✓       |       |
| Surprise     |         | ✓       | ✓     |
| Disappointed |         |         | ✓     |
| Bored        |         |         | ✓     |
| Excited      |         |         | ✓     |


| Feature       | CREMA-D        | RAVDESS    | LSSED              |
| ------------- | -------------- | ---------- | ------------------ |
| Male Actors   | 48             | 12         | 335                |
| Female Actors | 42             | 12         | 485                |
| Sentences     | 12             | 2          | 147,025            |
| Size          | ~2–4 GB (est.) | 24.8 GB    | ~100–150 GB (est.) |
| Duration      | 5 hrs 15 mins  | ~9.6 hours | 206 hrs 25 min     |

# Key Details

### CREMA-D
- Actors: 91
	- 48 Male
	- 42 Female
- Ages: 20 - 74
- Ethnicities: 5
	- **Caucasian**
	- African American, Asian, Hispanic, Unspecified
- Sentences: 12
- Size: 2–4 GB (estimated)
- Duration: Unknown
- Emotions: 6
	- Anger, Disgust, Fear, Happy, Neutral, Sad
	- Emotion levels: Low, Medium, High, and Unspecified
### RAVDESS
- Actors: 24
	- 12 Male
	- 12 Female
- Ages: 20 - 74
- Accent: 1
	- Neutral North American
- Sentences: Unknown
- Size: 24.8 GB
- Duration: 9.6 hours (24min/actor)
- Emotions: 6
	- Calm, Happy, Sad, Angry, Fearful, Surprise, and Disgust, Neutral
	- Emotion levels: normal, strong
### LSSED
- Actors: 820
	- 335 Male
	- 485 Female
- Ages: Young - Middle-aged - Old
- Language: 1
	- English
- Sentences: 147,025
- Size: 100–150 GB (estimated)
- Duration: 206 hrs 25 min
- Emotions: 10
	- Angry, Neutral, Fear, Happy, Sad, Disappointed, Bored, Disgusted, Excited, Surprised, Fear and ~~other~~ (ref [[#Other Note]])
	- Emotion levels: Low, Medium, High, and Unspecified

LSSED, a challenging large-scale English dataset for speech emotion recognition. It contains 147,025 sentences (206 hours and 25 minutes in total) spoken by 820 people. Each segment is annotated for the presence of 11 emotions (angry, neutral, fear, happy, sad, disappointed, bored, disgusted, excited, surprised, fear and other)
# CREMA-D

[**CREMA-D**](https://paperswithcode.com/dataset/crema-d) is an emotional multimodal actor data set of 7,442 original clips from **91** actors. These clips were from 48 male and 43 female actors between the ages of **20 and 74** coming from a variety of races and ethnicities (African America, Asian, **Caucasian**, Hispanic, and Unspecified).

Actors spoke from a selection of 12 sentences. The sentences were presented using one of six different emotions (Anger, Disgust, Fear, Happy, Neutral, and Sad) and four different emotion levels (Low, Medium, High, and Unspecified).

Participants rated the emotion and emotion levels based on the combined audiovisual presentation, the video alone, and the audio alone. Due to the large number of ratings needed, this effort was crowd-sourced and a total of 2443 participants each rated 90 unique clips, 30 audio, 30 visual, and 30 audio-visual. 95% of the clips have more than 7 ratings.

# RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)

The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS) contains 7,356 files (total size: 24.8 GB). The database contains **24** professional actors (12 female, 12 male), vocalizing two lexically-matched statements in a **neutral North American accent**. Speech includes calm, happy, sad, angry, fearful, surprise, and disgust expressions, and song contains calm, happy, sad, angry, and fearful emotions. Each expression is produced at two levels of emotional intensity (normal, strong), with an additional neutral expression. All conditions are available in three modality formats: Audio-only (16bit, 48kHz .wav), Audio-Video (720p H.264, AAC 48kHz, .mp4), and Video-only (no sound). Note, there are no song files for Actor_18.

# LSSED

Introduced by Fan et al. in [LSSED: a large-scale dataset and benchmark for speech emotion recognition](https://paperswithcode.com/paper/lssed-a-large-scale-dataset-and-benchmark-for)

LSSED, a challenging large-scale English dataset for speech emotion recognition. It contains 147,025 sentences (206 hours and 25 minutes in total) spoken by 820 people. Each segment is annotated for the presence of 11 emotions (angry, neutral, fear, happy, sad, disappointed, bored, disgusted, excited, surprised, fear and other)

###### Other Note
[13% of other uncommon samples can be used for tasks to distinguish whether they are common emotions.](https://arxiv.org/pdf/2102.01754v1)

![[Pasted image 20241107151827.png]]
# MELD (uncertain)

Multimodal EmotionLines Dataset ([MELD](https://affective-meld.github.io/)) has been created by enhancing and extending EmotionLines dataset. MELD contains the same dialogue instances available in EmotionLines, but it also encompasses audio and visual modality along with text. MELD has more than 1400 dialogues and 13000 utterances from Friends TV series. Multiple speakers participated in the dialogues. Each utterance in a dialogue has been labeled by any of these seven emotions -- Anger, Disgust, Sadness, Joy, Neutral, Surprise and Fear. MELD also has sentiment (positive, negative and neutral) annotation for each utterance.


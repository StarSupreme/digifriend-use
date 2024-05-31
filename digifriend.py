import os
import string
import random
import pandas as pd
import numpy as np
import sounddevice as sd
import librosa
from flask import Flask, render_template, request, jsonify
from gtts import gTTS
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from fer import FER
import cv2
from cv2 import VideoCapture
import speech_recognition as sr
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import pyautogui
import pygetwindow as gw
from datetime import datetime
import time
import io
import tempfile
import subprocess
import ssl
import base64

app = Flask(__name__)
app.secret_key = 'MANBEARPIG_MUDMAN888'

prediction_done = False

model_name = "facebook/blenderbot-400M-distill"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

df = pd.read_csv('./train.csv')
df['Concept'] = df['Concept'].str.lower().str.strip()
df['Description'] = df['Description'].str.lower().str.strip()
df['context'] = df['Concept'] + " " + df['Description']
df = df[['context', 'Description']]
df.columns = ['context', 'response']

fallback_responses = [
    "I'm here to help. Could you give me more details?",
    "Can you elaborate on that?",
    "Let's dive deeper into this topic. What exactly would you like to know?",
    "I'm not sure I understand fully. Could you clarify?",
    "Interesting point! Could you tell me a bit more about what you're thinking?",
    "I'm here to assist! How can I help you further?",
    "That's an interesting topic! Do you have any specific questions about it?",
    "Could you provide more context? I'd love to help out!",
    "I'm eager to assist! What more can I do for you?",
    "Let's explore this topic together. What more can you tell me?"
]

casual_responses = {
    "how are you": "I'm doing well, thanks for asking!",
    "hello": "Hello there! It's great to chat with you.",
    "bye": "Goodbye! Have a wonderful day.",
    "hi": "Hi there! How can I assist you today?",
    "i am happy": "Do you know what I do when I'm in a good mood?",
    "what do you do when you are happy": "I like to read things",
    "so you are a lawyer": "No, I'm talking about the 'Laws of Motion'"
}

def preprocess_text(text):
    return text.lower().translate(str.maketrans('', '', string.punctuation)).strip()

def recognize_speech():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Say something!")
        audio = r.listen(source)

    try:
        text = r.recognize_google(audio)
        print(f"You said: {text}")
        return text
    except sr.UnknownValueError:
        print("Could not understand audio")
        return None
    except sr.RequestError as e:
        print(f"Error: {e}")
        return None

def retrieve_response(query, df, word_threshold=3):
    query_words = set(preprocess_text(query).split())
    best_match = None
    best_count = 0
    exact_match = None

    for _, row in df.iterrows():
        concept_words = set(preprocess_text(row['context']).split())
        context_words = set(preprocess_text(row['context']).split())

        concept_match = concept_words.issubset(query_words)
        if concept_match:
            exact_match = row['response']
            break

        common_words = query_words & context_words
        common_count = len(common_words)

        query_word_count = len(query_words)
        best_count_threshold = query_word_count * 3 // 4

        if common_count > best_count_threshold:
            best_count = common_count
            best_match = row['response']

    if exact_match:
        response = exact_match
    elif best_count >= word_threshold:
        response = best_match
    else:
        response = generate_response_with_dialogpt(query)

    # Add a condition to check if the response contains any of the specified words
    forbidden_words = ["love", "girlfriend", "boyfriend", "cheated", "ex", "not good enough"]
    for word in forbidden_words:
        if word in response.lower():
            response = generate_response_with_dialogpt(query)

    return response

def generate_response_with_dialogpt(query):
    input_ids = tokenizer(query, return_tensors="pt").input_ids
    output_ids = model.generate(input_ids, max_length=50, num_beams=5)
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return response

def get_camera_frame():
    cap = cv2.VideoCapture(0)  # 0 is the default camera index
    ret, frame = cap.read()
    cap.release()
    return frame

def perform_real_time_prediction():
    global final_emotion

    def extract_features(audio_data, sample_rate):
        mfccs = np.mean(librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40).T, axis=0)
        return mfccs

    def generate_audio_data(emotion, duration, sample_rate):
        t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)

        if emotion == 'happy':
            frequency = np.interp(np.random.random(), [0, 1], [220, 630])
            audio_data = np.sin(2 * np.pi * frequency * t + 5 * np.sin(2 * np.pi * 0.25 * t))
        elif emotion == 'sad':
            frequency = np.interp(np.random.random(), [0, 1], [100, 700])
            audio_data = np.sin(2 * np.pi * 220 * t) * np.interp(t, [0, duration], [1, 0])
        elif emotion == 'angry':
            frequency = np.interp(np.random.random(), [0, 1], [300, 700])
            audio_data = librosa.core.tone(frequency, sr=sample_rate, duration=duration) + 0.3 * np.sin(
                2 * np.pi * 0.5 * t)
        else:
            audio_data = np.interp(np.random.rand(int(duration * sample_rate)), [0, 1], [-1, 1])

        return audio_data

    sample_rate = 22050
    duration = 3

    emotions = ['happy', 'sad', 'angry', 'neutral']
    X = []
    y = []

    for emotion in emotions:
        for _ in range(50):
            audio_data = generate_audio_data(emotion, duration, sample_rate)
            features = extract_features(audio_data, sample_rate)
            X.append(features)
            y.append(emotion)

    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)

    def predict_emotion(audio_data, sample_rate):
        features = extract_features(audio_data, sample_rate)
        features = features.reshape(1, -1)
        predicted_emotion = rf_classifier.predict(features)
        return predicted_emotion[0]

    duration = 5
    sample_rate = 22050
    channels = 1

    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels, blocking=True)
    dominant_emotion_voice = predict_emotion(audio_data[:, 0], sample_rate)
    emotions_voice = dominant_emotion_voice

    detector = FER(mtcnn=True)

    emotion_intensities = {
        'happy': 0,
        'sad': 0,
        'fear': 0,
        'disgust': 0,
        'angry': 0,
        'surprise': 0,
        'neutral': 0
    }

    latest_frame = get_camera_frame()
    if latest_frame is None:
        print("Failed to capture camera frame")
    else:
        small_frame = cv2.resize(latest_frame, (0, 0), fx=0.5, fy=0.5)
        result = detector.detect_emotions(small_frame)
        if result:
            for face in result:
                emotions_face = face['emotions']
                dominant_emotion_face = max(emotions_face, key=emotions_face.get)

                def update_emotion_intensities_voice(emotion):
                    emotion_intensities[emotion] += 1

                update_emotion_intensities_voice(emotions_voice)

                weight_face = 0.6
                weight_voice = 0.4

                normalized_emotion_face = emotions_face[dominant_emotion_face] / sum(emotions_face.values())
                normalized_emotion_voice = emotion_intensities.get(emotions_voice, 0) / sum(
                    emotion_intensities.values())

                combined_emotion_score = (normalized_emotion_face * weight_face) + (
                        normalized_emotion_voice * weight_voice)

                final_emotion = dominant_emotion_face if combined_emotion_score >= 0.5 else emotions_voice

                emotion_voice = predict_emotion(audio_data[:, 0], sample_rate)

                if dominant_emotion_face == emotions_voice:
                    final_emotion = dominant_emotion_face
                elif dominant_emotion_face != emotions_voice and dominant_emotion_face in ["happy", "neutral",
                                                                                           "surprise"] and emotions_voice in [
                    "happy", "neutral"]:
                    final_emotion = "happy"
                elif dominant_emotion_face != emotions_voice and dominant_emotion_face in ["disgust", "fear",
                                                                                           "sad"] and emotions_voice in [
                    "sad", "angry"]:
                    final_emotion = "sad"
                elif dominant_emotion_face != emotions_voice and dominant_emotion_face in ["fear",
                                                                                           "surprise"] and emotions_voice in [
                    "sad", "neutral"]:
                    final_emotion = "fear"
                elif dominant_emotion_face != emotions_voice and dominant_emotion_face in ["neutral", "angry",
                                                                                           "sad"] and emotions_voice in [
                    "sad", "angry", "neutral"]:
                    final_emotion = "sad"

                elif dominant_emotion_face == None:
                    final_emotion = dominant_emotion_voice
                break

    print("Emotion: " + final_emotion)
    return final_emotion

def get_corpus_file_path():
    global introline

    if final_emotion == 'neutral':
        introline = "Hi, how are you?"

    elif final_emotion == 'happy':
        introline = "Someone seems happy!"

    elif final_emotion == 'sad':
        introline = "Someone is sad, what is the matter?"

    elif final_emotion == 'angry':
        introline = "You have a bad temper now, what's the matter?"

    elif final_emotion == 'disgust':
        introline = "What is that?!"

    elif final_emotion == 'surprise':
        introline = "OMG! What just happened?"

    elif final_emotion == 'fear':
        introline = "Woah! What happened?!"

def speak(text):
    tts = gTTS(text=text, lang='en')
    tts.save('output.mp3')
    sound = SoundLoader.load('output.mp3')
    if sound:
        sound.play()

@app.route('/', methods=['GET', 'POST'])
def home():
    global introline, final_emotion, prediction_done

    # Execute emotion prediction only if it hasn't been done yet
    if not prediction_done:
        final_emotion = perform_real_time_prediction()
        get_corpus_file_path()
        prediction_done = True

    # Generate the introline audio file
    tts = gTTS(text=introline, lang='en')
    audio_data = io.BytesIO()
    tts.write_to_fp(audio_data)
    audio_data.seek(0)

    # Encode the audio data as base64
    audio_base64 = base64.b64encode(audio_data.getvalue()).decode('utf-8')

    response = ''

    if request.method == 'POST':
        if 'audio' in request.files:
            audio_file = request.files['audio']
            temp_audio = None

            if audio_file.filename.lower().endswith('.webm'):
                temp_audio = tempfile.NamedTemporaryFile(suffix=".webm", delete=False)
                audio_file.save(temp_audio.name)
                temp_wav = temp_audio.name.replace('.webm', '.wav')

                # Convert WebM to WAV using ffmpeg
                try:
                    subprocess.run(['ffmpeg', '-i', temp_audio.name, temp_wav], check=True)
                except subprocess.CalledProcessError as e:
                    os.remove(temp_audio.name)
                    return jsonify({"error": f"Failed to convert WebM to WAV: {e}"}), 500
                except FileNotFoundError:
                    os.remove(temp_audio.name)
                    return jsonify(
                        {"error": "FFmpeg not found. Make sure it's installed and added to your system path."}), 500
            else:
                return jsonify({"error": "No valid WebM audio file provided"}), 400

            try:
                recognizer = sr.Recognizer()
                with sr.AudioFile(temp_wav) as source:
                    audio_data = recognizer.record(source)
                    text = recognizer.recognize_google(audio_data)
            except ValueError as e:
                return jsonify({"error": f"Could not read the audio file: {e}"}), 500
            except sr.UnknownValueError:
                return jsonify({"error": "Google Speech Recognition could not understand audio"}), 400
            except sr.RequestError as e:
                return jsonify({"error": f"Could not request results from Google Speech Recognition service; {e}"}), 500
            finally:
                temp_audio.close()  # Close the file before removing it
                os.remove(temp_audio.name)
                os.remove(temp_wav)

            # Process the recognized text to get the response
            if text in casual_responses:
                response = casual_responses[text]
            else:
                response = retrieve_response(text, df)
                if not response:
                    response = random.choice(fallback_responses)

                tts = gTTS(text=response, lang='en')
                audio_data = io.BytesIO()
                tts.write_to_fp(audio_data)
                audio_data.seek(0)

                audio_base64 = base64.b64encode(audio_data.getvalue()).decode('utf-8')

                return jsonify({"transcript": text, "message": response, "audio_base64": audio_base64})
        else:
            input_text = recognize_speech()
            if not input_text:
                return render_template('index.html', response="Sorry, I didn't understand that.", final_emotion='')

            if input_text in casual_responses:
                response = casual_responses[input_text]
            else:
                response = retrieve_response(input_text, df)
                if not response:
                    response = random.choice(fallback_responses)


    return render_template('index.html', response=response, final_emotion=final_emotion, introline=introline, audio_base64=audio_base64)


@app.route('/analyze_latest_screenshot', methods=['GET'])
def analyze_latest_screenshot():
    final_emotion = perform_real_time_prediction()
    if final_emotion:
        get_corpus_file_path()
        return {
            "emotion": final_emotion,
            "message": text
        }
    else:
        return {"error": "No screenshots available or failed to process the screenshot."}, 404


if __name__ == '__main__':
    print("Starting the application...")
    context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
    context.load_cert_chain('cert.pem', 'key.pem')
    app.run(host='192.168.0.105', port=443, debug=False, ssl_context=context)
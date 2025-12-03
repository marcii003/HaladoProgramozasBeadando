import warnings
from urllib3.exceptions import NotOpenSSLWarning
warnings.filterwarnings("ignore", category=NotOpenSSLWarning)

import os
import pickle
import time
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

#hangfile jellegzetessegeinek betoltese
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Hiba történt a hangfájl feldolgozásakor: {file_path}, {e}")
        return None

#adatbeolvasas az "igen", "nem" es "ertelmetlen" mappakbol
def load_data(data_folder):
    data = []
    labels = []
    for label in ["igen", "nem", "ertelmetlen"]:
        folder = os.path.join(data_folder, label)
        for file in os.listdir(folder):
            if file.endswith(".wav"):
                features = extract_features(os.path.join(folder, file))
                if features is not None:
                    data.append(features)
                    labels.append(label)
    return np.array(data), np.array(labels)

#hasznos adat/adatok ellenorzese
def is_audio_active(audio_file, min_silence_len=1000, silence_thresh=-40):
    try:
        sound = AudioSegment.from_wav(audio_file)
        nonsilent = detect_nonsilent(sound, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
        return len(nonsilent) > 0
    except Exception as e:
        print(f"Hiba történt az aktív hang ellenőrzésekor: {e}")
        return False

#modell betanitas
def train_model():
    print("Adatok betöltése...")
    data_folder = "./data"
    X, y = load_data(data_folder)

    print("Adathalmaz mérete:", X.shape, y.shape)

    #adatok szetosztasa
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Modell tanítása...")
    model = SVC(kernel="linear", probability=True)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"Pontosság: {accuracy * 100:.2f}%")

    with open("speech_model.pkl", "wb") as f:
        pickle.dump(model, f)
        print("A modell elmentve: speech_model.pkl")

#hangfelvetelbol valo adatkinyeres
def predict_command(audio_file, model):
    features = extract_features(audio_file)
    if features is not None:
        prediction = model.predict([features])
        probabilities = model.predict_proba([features])
        max_prob = max(probabilities[0])
        if max_prob < 0.5:  #ha az adott hangfile tartalmi valoszinusege a kulcsszavakra kevesebb mint 50%, ertelmetlent dob
            return "ertelmetlen", max_prob
        return prediction[0], max_prob
    else:
        return None, 0.0

#program futasakor, mikrofonbol valo felismeres
def recognize_speech():
    print("Mond ki a parancsot(igen/nem)!")
    with sr.Microphone() as source:
        try:
            recognizer = sr.Recognizer()
            recognizer.adjust_for_ambient_noise(source)
            print("Figyelek...")

            #mentett modell betoltese
            try:
                with open("speech_model.pkl", "rb") as f:
                    model = pickle.load(f)
            except FileNotFoundError:
                print("A speech_model.pkl nem található. Előbb tanítsd be a modellt!")
                return

            #idozites keret
            start_time = time.time()

            while True:
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)

                #ideiglenes filbeba valo mentes
                with open("temp.wav", "wb") as f:
                    f.write(audio.get_wav_data())

                #hangfile ellenorzes(van e ertelmes beszed)
                if is_audio_active("temp.wav"):
                    command, probability = predict_command("temp.wav", model)
                    if command and probability > 0.5:
                        print(f"Felismert parancs: {command} (Biztonság: {probability:.2f})")
                    else:
                        print("Értelmetlen beszédet észleltem.")
                    break
                else:
                    print("Csendet észleltem, nincs hallható parancs.")

                #5 masodperces csend erzekelese
                if time.time() - start_time > 5:
                    print("Nem volt parancs hallható.")
                    break

        except sr.WaitTimeoutError:
            print("Nem volt parancs hallható (időtúllépés).")
        except Exception as e:
            print(f"Hiba történt: {e}")

#a program menuje / feladatvalasztoja
if __name__ == "__main__":
    print("1: Modell betanítása")
    print("2: Beszédfelismerés indítása")
    choice = input("Választás (1/2): ")

    if choice == "1":
        train_model()
    elif choice == "2":
        recognize_speech()
    else:
        print("Érvénytelen választás!")
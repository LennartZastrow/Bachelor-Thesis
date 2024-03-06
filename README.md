# Multivariante Emotionsanalyse mittels Convolutional Neural Networks


## Übersicht
Dieses Bachelorarbeit erforscht die Kongruenz zwischen Mimik, Körpersprache und Sprache durch den Einsatz vortrainierter CNN-Netzwerke. Speziell für Mimik und Körpersprache wurde ein eigenes Dataset erstellt. Tests erfolgen auf separaten Datensätzen, die die Emotionen einer Person zu einem bestimmten Zeitpunkt erfassen – für Mimik und Körpersprache als Frame und für Sprache als Audioframe (Monolog, 2-10 Sekunden, im Moment der Expression). Anschließend wird mittels des Pearson-Korrelationskoeffizienten eine paarweise Korrelationsanalyse durchgeführt. 

Hinweis: Die Datenintegration wurde gekürzt, damit die Dokumentation übersichtlicher erscheint. Die vollständigen Daten sind der Exeltabelle zu entnehmen. Für die Reproduktion des Projekts können die bereits bearbeiteten Datasets verwendet werden.

## Installation und Setup
#### Hardware
Für optimale Leistung basiert das Setup auf einem 13. Generation Intel Core i9 13900HX Prozessor mit 32 CPUs, ausgestattet mit 32 GB RAM und einer NVIDIA GeForce RTX 4070 Grafikkarte, um die Anforderungen moderner Deep-Learning-Aufgaben zu erfüllen.

#### Enviroment-Aufbau
Für das Enviroment müssen folgende Packete installiert werden
```bash 
pip install numpy matplotlib pandas librosa mtcnn
```

**TensorFlow GPU** wurde installiert, um die Berechnungsgeschwindigkeit und Effizienz zu maximieren, da GPUs parallele Verarbeitung für schnelleres Training und Inferenz von Deep-Learning-Modellen bieten
```bash
# Zur Installation von Cuda und CuDNN
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
# Alles über 2.10 wird auf der GPU unter Windows Native nicht unterstützt
python -m pip install "tensorflow<2.11"
```
##### Überprüfung der Installation:
```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```
## Dataset
Um ein geeignetes Dataset zu erstellen mussten die Videos von den Interviews analysiert werden nach Expressionen und dann in die Ordner Ekel, Freude, Trauer, Angst gespeichert werden. Zu beachten ist, dass danach eine Ausselektierung von ungeeigneten Bilder per Hand erfolgen muss. Vorerst werden die Videos in Bilder geschnitten, welche als Gankörper-Dataset fungieren, danach wird mithilfe eines Gesichtsclassifiers das Mimik-Dataset erstellt
### Gankörper-Dataset 
Die Emotionsexpressionen, welche in Zeitabschnitten(Timestamps) dokumentiert wurden, können mithilfe dieses Codes geschnitten werden und in 24 Frames/Sekunde gespeichert werden. Es wurden fünf Leute interviewt. Die einzelnen Framesschnitte finden sich in dem Ordner wieder. 

```python

# Packages
import cv2
import os
import numpy as np

# Pfad zur Videodatei 
video_path_freude = "Pfad"

timestamp_sequences_freude = [
    ((80, 81), "freude"),  # 1:20
    ((121, 122), "freude"),  # 2:01
    ((180, 181), "freude"),  # 3:00
    ((192, 193), "freude"),  # 3:12
    # Fügen Sie hier die Zeitstempel-Sequenzen für Benny_Freude hinzu
]

# Pfad zu den Emotionsordnern
emotion_folders = {
   "freude": "Pfad" 
}

# Videocapture-Objekt erstellen
cap_freude = cv2.VideoCapture(video_path_freude)

# Eigenschaften der Videos abrufen
fps_freude = cap_freude.get(cv2.CAP_PROP_FPS)
frame_interval_freude = int(fps_freude / images_per_second)

# Funktion zur Verarbeitung der Zeitstempel-Sequenzen mit angepasster Bildspeicherung
def process_timestamp_sequences(sequences, cap, fps, frame_interval):
    for i, (timestamp_range, emotion) in enumerate(sequences):
        start_time, end_time = timestamp_range
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)

        frame_count = start_frame
        file_count = 1  # Zählvariable für die Dateinummer

        while frame_count <= end_frame:
            # Auf den aktuellen Frame setzen
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)

            # Frame lesen
            ret, frame = cap.read()

            if not ret:
                break

            # Frame um 90 Grad nach rechts drehen
            rotated_frame = np.rot90(frame, axes=(1, 0))

            # Zeitstempel für den Dateinamen generieren
            timestamp_str = f"{int(start_time):03d}_{int(end_time):03d}"

            # Dateiname generieren
            file_name = f"Name_{emotion}_{timestamp_str}_{file_count}.jpg"

            # Ausgabepfad für das Frame
            output_folder_path = emotion_folders[emotion]
            output_path = os.path.join(output_folder_path, file_name)

            # Frame speichern
            cv2.imwrite(output_path, rotated_frame)

            frame_count += frame_interval
            file_count += 1

# Frames extrahieren und verarbeiten für 
process_timestamp_sequences(timestamp_sequences_freude, cap_freude, fps_freude, frame_interval_freude)

# Aufräumen
cap_allgemein.release()
cap_trauer.release()
print("Frames erfolgreich extrahiert und gespeichert.")
```


### Mimik-Dataset
Es wurde ein MTCCN(Multi-task Cascaded Convolutional Networks) benutzt um Gesichter zu detektieren und diese aus dem Ganzkörper-Dataset zu extrahieren. Anschließend erfolgte noch eine Ausselektierung der ungeeigneten Bilder per Hand.
```python
import cv2
import os
from mtcnn import MTCNN
import tensorflow as tf

# Basispfade
basis_bildordner = "C:/Thesis/Data/Frames/Fullbody_sorted_Body_Language/"
basis_emotionsordner = "C:/Thesis/Data/Frames/Mimik/"

# Emotionsliste
emotionen = ["Freude", "Ekel", "Trauer", "Wut", "Angst"]

# Initialisiere MTCNN-Detektor
detector = MTCNN()

# Iteriere durch jede Emotion
for emotion in emotionen:
    bildordner = os.path.join(basis_bildordner, emotion)
    emotionsordner = os.path.join(basis_emotionsordner, emotion)

    # Stelle sicher, dass der Ordner existiert
    if not os.path.exists(emotionsordner):
        os.makedirs(emotionsordner)

    # Liste alle Bilddateien im Bildordner auf
    bilder = [datei for datei in os.listdir(bildordner) if datei.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Verarbeite jedes Bild
    for bild_datei in bilder:
        bild_pfad = os.path.join(bildordner, bild_datei)

        # Lade das Bild mit OpenCV
        bild = cv2.imread(bild_pfad)

        # Erkenne Gesichter im Bild
        ergebnisse = detector.detect_faces(bild)

        # Verarbeite jedes erkannte Gesicht
        for ergebnis in ergebnisse:
            x, y, w, h = ergebnis['box']
            erweiterungsfaktor = 0.5
            erweiterung_x = int(w * erweiterungsfaktor)
            erweiterung_y = int(h * erweiterungsfaktor)
            x_neu = max(0, x - erweiterung_x)
            y_neu = max(0, y - erweiterung_y)
            w_neu = min(bild.shape[1], w + 2 * erweiterung_x)
            h_neu = min(bild.shape[0], h + 2 * erweiterung_y)

            # Schneide den Ausschnitt aus dem Bild
            ausschnitt = bild[y_neu:y_neu + h_neu, x_neu:x_neu + w_neu]

            # Speichere den Ausschnitt im entsprechenden Emotionsordner
            ausgabedatei = os.path.join(emotionsordner, f"{bild_datei}_gesicht.jpg")
            cv2.imwrite(ausgabedatei, ausschnitt)

print("Fertig!")

```



## Verarbeitung und Integration bestehender Datasets
Um die Variation in den Daten zu erhöhen und damit die Gerneralierbarkeit des Models zu erhöhen, wurde exterene Datasets genutzt und konkateniert. 
Die Mimikanalyse benötigte keine exterenen Datasets, da die Quantität des erstellten Datasets genügend war. 
Für die *Körpersprache* wurden die beiden Datasets genutzt
- Das BEAST-Set (Brain Emotional Activation of Survival Threats) von Beatrice de Gelder
- Der GEMEP-Corpus

#### BEAST-SET
Zur Verarbeitung des Datasets von De Gelder wurden die entsprechenden Bilder in die Traindata-Ordner sortiert. Die Bilder wurden in den Dateinamen mit folgenden Beitelungen versehen, um die Emotion zuzuordnen.
- AN : Wut
- FE : Angst
- HA : Freude
- SA : Trauer

Der folgende Code sortiert die Bilder in entsprechenden Emotionsordner 
```python
import os
import shutil

# Basispfad für die Quelldateien
source_base_path = "C:\\Users\\zastr\\Desktop\\Thesis\\Data\\Frames\\Externe_Datasets"
# Basispfad für die Zielordner
target_base_path = "C:\\Users\\zastr\\Desktop\\Thesis\\Data\\Frames\\Externe_Datasets\\BodyLanguage_DeGelder"

emotion_folders = {
    "AN": "Wut",
    "FE": "Angst",
    "HA": "Freude",
    "SA": "Trauer",

}

# Durchsuche den Basisordner nach .bmp Bildern
for file in os.listdir(source_base_path):
    if file.endswith(".bmp"):  # Berücksichtigt nur .bmp Dateien
        # Entferne Zahlen am Ende des Dateinamens, falls vorhanden
        file_name_without_numbers = ''.join([i for i in file[:-4] if not i.isdigit()])
        # Extrahiere die letzten zwei Buchstaben des Dateinamens
        emotion_code = file_name_without_numbers[-2:]
        
        # Finde den entsprechenden Emotionsordner basierend auf dem Code
        target_folder = emotion_folders.get(emotion_code)
        if target_folder:
            # Erstelle den vollständigen Pfad zum Zielordner
            target_path = os.path.join(target_base_path, target_folder)
            # Erstelle den Zielordner, falls er noch nicht existiert
            if not os.path.exists(target_path):
                os.makedirs(target_path)
            # Verschiebe die Datei in den Zielordner
            shutil.move(os.path.join(source_base_path, file), os.path.join(target_path, file))
```

#### GEMEP-Corpus

### Sprache
Da kein eigenes Dataset für die Audtio erstellt wurde, wurden drei unterschiedliche externe Datasets verwendet. 
- RAVDESS Dataset
- EMO-Database
- GEMEP-Corpus

##### RAVEDESS Dataset
RAVDESS-Dataset (The Ryerson Audio-Visual Database of Emotional Speech and Song) beinhaltet 7,356 Dateien mit einem Gesamtvolumen von 24.8 GB, aufgenommen von 24 professionellen Schauspielern (12 weiblich, 12 männlich), die zwei lexikalisch angeglichene Aussagen in einem neutralen nordamerikanischen Akzent vokalisieren.
Zur Verarbeitung wurde sich hier wieder nur auf die Emotionen Ekel, Angst, Trauer und Freude konzentiert. Die Dateien sind schon im `.wav` Format, das geeignete Format für Feature Extraktion

```python
import os
import shutil

# Pfad zum RAVDESS-Datensatz
ravdess_path = r"C:\Users\zastr\Desktop\Thesis\Data\Audio\RAVDESS-Dataset"

# Zielordner für die verschiedenen Emotionen
destination_paths = {
    '03': r"C:\Users\zastr\Desktop\Thesis\Data\Audio\Emo_DB\Freude",
    '04': r"C:\Users\zastr\Desktop\Thesis\Data\Audio\Emo_DB\Trauer",
    '06': r"C:\Users\zastr\Desktop\Thesis\Data\Audio\Emo_DB\Angst",
    '07': r"C:\Users\zastr\Desktop\Thesis\Data\Audio\Emo_DB\Ekel"
}

# Funktion zum Verschieben der Dateien
def move_files_based_on_emotion(source_path, dest_paths):
    for root, dirs, files in os.walk(source_path):
        for file in files:
            # Überprüfen, ob es eine Sprachaufnahme ist
            if file.split('-')[1] == '01':  # Vocal channel = speech
                emotion_code = file.split('-')[2]  # Emotion aus dem Dateinamen extrahieren
                
                # Überprüfen, ob die Emotion in unseren Zielordnern ist
                if emotion_code in dest_paths:
                    source_file_path = os.path.join(root, file)
                    destination_file_path = os.path.join(dest_paths[emotion_code], file)
                    
                    # Datei verschieben
                    shutil.copy(source_file_path, destination_file_path)
                    print(f"Moved: {file} to {dest_paths[emotion_code]}")

# Ausführen der Funktion
move_files_based_on_emotion(ravdess_path, destination_paths)
```

#### EMO-DB
Die EMO-DB (Berlin Database of Emotional Speech)ist eine umfangreiche Sammlung für Sprachemotionsanalyse, bestehend aus deutschen Sprachaufnahmen, die verschiedene Emotionen wie Glück, Traurigkeit, Wut, Angst, Ekel, Überraschung und neutrale Zustände darstellen. Sie umfasst etwa 535 Aufnahmen von 10 Schauspielern und benötigt ca. 500 MB Speicherplatz. 
Es wurde sich nur auf die zu vier testen Emotionen beschränkt. Die Dateiennamen waren mit einem Buchstaben für die Emotion gekennzeichnet wie zum Beispiel `03a05Aa.wav`. Der vorletzte Buchstabe kennzeichnet die Emotion, also A entspricht Angst.
Dieser Code hat die EMO-DB sortiert und ausselektiert.

```python
# Dieser Code sortiert die einzelnen Auditodateien aus der EMO-DB in die entsprechenenden Ordner für Ekel, Angst, Trauer, Angst 

import os
import shutil

# Pfad zum Verzeichnis mit den Audiodateien
source_dir = "C:/Users/zastr/Desktop/Thesis/Data/Audio/Emo_DB/wav"

# Zielverzeichnisse für die verschiedenen Emotionen
emotion_dirs = {
    "E": "C:/Users/zastr/Desktop/Thesis/Data/Audio/Emo_DB/Ekel",
    "F": "C:/Users/zastr/Desktop/Thesis/Data/Audio/Emo_DB/Freude",
    "T": "C:/Users/zastr/Desktop/Thesis/Data/Audio/Emo_DB/Trauer",
    "A": "C:/Users/zastr/Desktop/Thesis/Data/Audio/Emo_DB/Angst"
}

# Stelle sicher, dass die Zielverzeichnisse existieren
for dir in emotion_dirs.values():
    os.makedirs(dir, exist_ok=True)

# Durchlaufe alle Dateien im Quellverzeichnis
for filename in os.listdir(source_dir):
    if filename.endswith(".wav"):
        # Bestimme die Emotion basierend auf dem vorletzten Buchstaben des Dateinamens
        emotion = filename[-6].upper()
        
        # Prüfe, ob die Emotion in den Zielverzeichnissen vorhanden ist
        if emotion in emotion_dirs:
            # Vollständiger Pfad zur Quelldatei
            source_file = os.path.join(source_dir, filename)
            
            # Zielverzeichnis basierend auf der Emotion
            target_dir = emotion_dirs[emotion]
            
            # Vollständiger Pfad zur Zieldatei
            target_file = os.path.join(target_dir, filename)
            
            # Verschiebe die Datei in das entsprechende Zielverzeichnis
            shutil.copy(source_file, target_file)

print("Fertig.")
```

## Models 
Für die Analyse des Mimik und Körpersprache Dataset werden Pretrained Networks genutzt. Für die Analyse der Sprache wird eine Feature Extrakion genutzt. Die extrahierten Feautres werden dann als Tensor in ein Convolutional Neural Network übergeben und so analysiert.
### Mimik und Körpersprache
Für die Analyse bezüglich der Auswahl eines geeigneten Pre-Trained Networks wurde auf diese Netwerke geteste `ResNet50` , `Resnet101`,  `VGG16`, `VGG19`, `InceptionV3`, `DenseNet 169` und `EfficientNetB7`. Diese lassen sich in dem Ordner Model wiederfinden. Es wurden die Dataaugmentationtechniken *Rotation*, *RandomFlip* und *Contrast* genutzt. Diese Models und Architekturen wurden für Mimik und Körpersprache genutzt. 

Dies ist ein Beispiel für das `DensetNet169`.
```python
import datetime
import numpy as np
import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.applications import DenseNet169

# TensorFlow Warnungen unterdrücken
tf.get_logger().setLevel('ERROR')

# Betitelung für Tensorboard
experiment_name = "DenseNet169"

# Data Augmentation Layer hinzufügen
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),  # Rotation um bis zu 20%
    tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),  # Horizontaler und vertikaler Flip
    tf.keras.layers.experimental.preprocessing.RandomContrast(0.1),  # Zufällige Kontrastanpassung
])

# Funktion zur Vorbereitung und Augmentation der Bilder
def preprocess_and_augment_dataset(ds):
    # Zuerst Data Augmentation anwenden
    ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y))
    # Dann die spezifische Vorverarbeitung für DenseNet anwenden
    return ds.map(lambda x, y: (preprocess_input(x), y))

# Funktion zur Vorbereitung der Testdaten
def preprocess_dataset(ds):
    return ds.map(lambda x, y: (preprocess_input(x), y))    

# Trainingsdaten laden und vorbereiten
train_ds = image_dataset_from_directory(
    "C:\\Thesis\\Data\\Frames\\Facial_Expressions",
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(224, 224),
    batch_size=32)

# Anwendung der Vorverarbeitung und Augmentation auf Trainingsdaten
train_ds = preprocess_and_augment_dataset(train_ds)

# Validierungsdaten laden und vorbereiten
val_ds = image_dataset_from_directory(
    "C:\\Thesis\\Data\\Frames\\Facial_Expressions",
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(224, 224),
    batch_size=32)

# Anwendung der Vorverarbeitung (ohne Augmentation) auf Validierungsdaten
val_ds = preprocess_and_augment_dataset(val_ds) 

# Vortrainiertes Modell laden
base_model = DenseNet169(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Einfrieren der Basis-Schichten

# Modell anpassen
inputs = tf.keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
outputs = Dense(4, activation='softmax')(x)  # 4 Klassen für Emotionen
model = Model(inputs, outputs)

# Modell kompilieren und trainieren
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Setze den Pfad für die TensorBoard Logs
log_dir = "logs/fit/" + experiment_name + "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, update_freq="epoch")

model.fit(
    train_ds, 
    validation_data=val_ds, 
    epochs=2, 
    callbacks=[tensorboard_callback]
)

```

### Sprache
Zur Feature Extraktion benutzen wir die `libroa`-Bibliothek. Es werden die folgenden Features extrahiert:
- MFCCS
- Mel Spectrogram
- Chromagram
- Spectral contrast
- Tonnetz

```python
import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout, BatchNormalization, MaxPooling1D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt

# Funktion zur Extraktion der Audio-Features
def extract_audio_features(file_path, n_fft=512):
    y, sr = librosa.load(file_path)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    chromagram = librosa.feature.chroma_stft(y=y, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    features = np.hstack((np.mean(mfccs, axis=1), 
                          np.mean(librosa.power_to_db(mel_spectrogram), axis=1),
                          np.mean(chromagram, axis=1),
                          np.mean(spectral_contrast, axis=1),
                          np.mean(tonnetz, axis=1)))
    if len(features) > 193:
        features = features[:193]
    elif len(features) < 193:
        features = np.pad(features, (0, 193 - len(features)), 'constant')
    return features

# Daten und Labels laden
def load_data_and_labels(base_path):
    emotions = {'Angst': 0, 'Ekel': 1, 'Trauer': 2, 'Freude': 3}
    features_list = []
    labels_list = []
    for emotion, label in emotions.items():
        emotion_path = os.path.join(base_path, emotion)
        for filename in os.listdir(emotion_path):
            if filename.endswith(".wav"):
                file_path = os.path.join(emotion_path, filename)
                features = extract_audio_features(file_path)
                features_list.append(features)
                labels_list.append(label)
    return np.array(features_list), np.array(labels_list)

# CNN Modelldefinition mit L2-Regularisierung und erweiterten Schichten
def build_model(input_shape, number_of_classes):
    model = Sequential([
        Conv1D(256, 5, input_shape=input_shape, padding='same', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        MaxPooling1D(pool_size=2),
        Conv1D(128, 5, padding='same', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Dropout(0.2),
        MaxPooling1D(pool_size=2),
        Conv1D(128, 5, padding='same', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Dropout(0.2),
        Flatten(),
        Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.2),
        Dense(number_of_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Hauptskript
base_path = r'C:\Users\zastr\Desktop\Thesis\Data\Audio\Emo_DB'
features, labels = load_data_and_labels(base_path)
X_train, X_temp, y_train, y_temp = train_test_split(features, labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.67, random_state=42)

model = build_model((193, 1), len(np.unique(labels)))

# Callbacks zur Anpassung der Lernrate und zum frühzeitigen Stopp, falls keine Verbesserung mehr stattfindet
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Modelltraining mit Early Stopping und Learning Rate Reduction
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32, callbacks=[reduce_lr, early_stopping])
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc}, Test loss: {test_loss}')
```


## Testdata für Korrelationsanalyse
Für die Korrelationsanaylse würde ein seperates Test-Dataset erstellt. Die Tests werden mit speziellen Datensätzen durchgeführt, die Emotionen einer Person zu einem präzisen Zeitpunkt festhalten. Dabei werden Mimik und Körpersprache in Einzelbildern und sprachliche Äußerungen in Audiosequenzen (Monologe von 2 bis 10 Sekunden während der emotionalen Ausdrucksphase) erfasst. Somit wird getestet, ob Expressionen kongruent zueinander auftreten. 

### Testset für Sprache
Für das Audio-Testset wurden die entscheidenen Stellen geschnitten und sortiert in die passenden Test-Data-Emotionsordner.  Die Installierung des Packets moviepy wir hier benötigt genutzt. In der Dokumenation sind die Daten geküzt, für eine vollständige Auflistung der Daten, schauen Sie gerne hier. 

```bash
pip install moviepy
```
```python

import os
from moviepy.editor import AudioFileClip

# Zähler für jede Emotion initialisieren
datei_zähler = {}

def schneiden_und_verschieben(tabelle, basis_pfad, ziel_ordner, emotion_name):
    global datei_zähler
    
    # Stelle sicher, dass der Zielordner existiert
    if not os.path.exists(ziel_ordner):
        os.makedirs(ziel_ordner)
    
    # Iteriere über jede Zeile in der Tabelle
    for zeile in tabelle:
        name_emotion, zeiten = zeile[0], zeile[1]
        name, _ = name_emotion.split('(')
        startzeit, endzeit = zeiten.split('-')
        
        # Konvertiere Zeiten von Minuten:Sekunden in reine Sekunden
        start_min, start_sec = map(int, startzeit.split(':'))
        end_min, end_sec = map(int, endzeit.split(':'))
        start_seconds = start_min * 60 + start_sec
        end_seconds = end_min * 60 + end_sec
        
        # Aktualisiere den Zähler für die angegebene Emotion
        if emotion_name not in datei_zähler:
            datei_zähler[emotion_name] = 1
        else:
            datei_zähler[emotion_name] += 1
        
        # Pfad zur ursprünglichen Datei
        original_datei_pfad = os.path.join(basis_pfad, name.strip(), f"{name.strip()}_{emotion_name}.m4a")
        
        # Ziel-Dateipfad ohne Zeitsequenz, nur mit Zähler
        ziel_datei_name = f"{emotion_name}{datei_zähler[emotion_name]}.m4a"
        ziel_datei_pfad = os.path.join(ziel_ordner, ziel_datei_name)
        
        # Schneide die Audiodatei und speichere sie
        with AudioFileClip(original_datei_pfad) as audio:
            new_audio = audio.subclip(start_seconds, end_seconds)
            new_audio.write_audiofile(ziel_datei_pfad, codec='aac')  # Verwende 'aac' als Codec

# Basispfad, wo die Originaldateien gespeichert sind
basis_pfad = "C:\\Users\\zastr\\Desktop\\Thesis\\Data\\Audio"

# Zielordner für jede Emotion
ziel_ordner_freude = "C:\\Users\\zastr\\Desktop\\Thesis\\Data\\Testdata\\Sprache\\Freude"
pfad_trauer = "C:\\Users\\zastr\\Desktop\\Thesis\\Data\\Testdata\\Sprache\\Trauer"
pfad_ekel = "C:\\Users\\zastr\\Desktop\\Thesis\\Data\\Testdata\\Sprache\\Ekel"
pfad_angst = "C:\\Users\\zastr\\Desktop\\Thesis\\Data\\Testdata\\Sprache\\Angst"

# Beispiel der Tabelle, die Sie bereitgestellt haben
tabelle_freude = [
    ["Benny(Freude)", "3:08-3:14"],

# Trauer
tabelle_trauer = [
    ["Benny(Trauer)", "1:53-2:04"],
]

# Ekel
tabelle_ekel = [
    ["Benny(Ekel)", "2:26-2:31"],
]

# Angst
tabelle_angst = [
    ["Benny(Angst)", "1:10-1:18"], 
]

# Funktion aufrufen für jede Emotion mit dem entsprechenden Zielordner und Emotionsnamen
schneiden_und_verschieben(tabelle_freude, basis_pfad, ziel_ordner_freude, "Freude")
schneiden_und_verschieben(tabelle_trauer, basis_pfad, pfad_trauer, "Trauer")
schneiden_und_verschieben(tabelle_ekel, basis_pfad, pfad_ekel, "Ekel")
schneiden_und_verschieben(tabelle_angst, basis_pfad, pfad_angst, "Angst")
```


### Testset für Mimik und Körper
Das Testset für Mimik und Körper wird mithilfe dieses Algorithmus geschnitten. Eine gesamte Auflistung aller Schneidealgorithmen ist hier auffindbar. 
Ebenso  ist das Packet shutil nötig, um die gewünschen Funtkionen `shutil.move`zu verwenden 

```bash
pip install shutil
```

```python
import os
import shutil

# Zähler für Dateien initialisieren
datei_zähler = 1

def kopieren_und_benennen(dateipfade, ziel_ordner):
    global datei_zähler
    
    # Stelle sicher, dass der Zielordner existiert
    if not os.path.exists(ziel_ordner):
        os.makedirs(ziel_ordner)
        
    for pfad in dateipfade:
        # Ziel-Dateiname
        ziel_dateiname = f"Freude{datei_zähler}.jpg"
        ziel_pfad = os.path.join(ziel_ordner, ziel_dateiname)
        
        # Kopiere die Datei
        shutil.move(pfad, ziel_pfad)
        print(f"Datei wurde kopiert und umbenannt zu {ziel_dateiname}")
        
        # Zähler erhöhen
        datei_zähler += 1

# Liste der Dateipfade
dateipfade = [
    r"C:\Users\zastr\Desktop\Thesis\Data\Frames\Facial_Expressions\Freude\Benny_freude_192_193_14.jpg_gesicht.jpg",
]

# Zielordner für die kopierten und umbenannten Dateien
ziel_ordner = r"C:\Users\zastr\Desktop\Thesis\Data\Testdata\Mimik\Freude"

# Funktion aufrufen
kopieren_und_benennen(dateipfade, ziel_ordner)
```

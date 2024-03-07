# Multivariante Emotionsanalyse mittels Convolutional Neural Networks


## Übersicht

Diese Bachelorarbeit untersucht die Kongruenz zwischen Mimik, Körpersprache und Sprache mithilfe von vortrainierten CNN-Netzwerken. Ein speziell für die Analyse von Mimik und Körpersprache entwickeltes Dataset kommt zum Einsatz. Die Untersuchung wird auf separaten Datensätzen durchgeführt, welche die Emotionen einer Person zu einem bestimmten Zeitpunkt erfassen – für Mimik und Körpersprache in Form von Frames und für Sprache als Audioframes (Monologe von 2-10 Sekunden Länge, zum Zeitpunkt der Expression). Anschließend erfolgt eine paarweise Korrelationsanalyse mittels des Pearson-Korrelationskoeffizienten.

Hinweis: Aus Gründen der Übersichtlichkeit wurde die Datenintegration (Pfade zu Bildern) in dieser Dokumentation gekürzt. Die vollständigen Informationen sind in der [Exeltabelle](Data/Timestamps) hinterlegt. Für die Reproduktion des Projekts können die bereits aufbereiteten Datasets verwendet werden.

1.  [Übersicht](#übersicht)
2.  [Installation und Setup](#installation-und-setup)
3.  [Dataset](#dataset)
4.  [Verarbeitung und Integration bestehender Datasets](Verarbeitung-und-Integration-bestehender-Datasets)
5.  [Models](#models)
6.  [Testdata für Korrelationsanalyse](#testdata-für-korrelationsanalyse)
7.  [Korrelationsanalyse](#korrelationsanalyse)


## Installation und Setup
#### Hardware
Für eine optimale Leistung basiert das Setup auf einem Intel Core i9 13900HX Prozessor der 13. Generation mit 32 CPUs, ergänzt durch 32 GB RAM und eine NVIDIA GeForce RTX 4070 Grafikkarte. Diese Konfiguration ist speziell darauf ausgerichtet, den hohen Anforderungen moderner Deep-Learning-Prozesse gerecht zu werden.

#### Enviroment-Aufbau
Für die Einrichtung des Arbeitsumfelds müssen folgende Pakete installiert werden:
```bash 
pip install numpy matplotlib pandas librosa mtcnn
```

**TensorFlow GPU** wurde installiert, um die Berechnungsgeschwindigkeit und Effizienz zu maximieren. GPUs ermöglichen parallele Verarbeitung, was das Training und die Inferenz von Deep-Learning-Modellen erheblich beschleunigt.
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
Um ein passendes Dataset für die Analyse zu erstellen, war es notwendig, die Interview-Videos auf emotionale Ausdrücke hin zu untersuchen und diese dann entsprechend den Emotionen Ekel, Freude, Trauer und Angst zu kategorisieren. Wichtig dabei ist, dass nach der initialen Auswahl eine manuelle Überprüfung und ggf. das Entfernen ungeeigneter Bilder erfolgen muss. Im ersten Schritt werden die Videos in Einzelbilder zerlegt, die als Ganzkörperaufnahmen dienen. Anschließend wird mithilfe eines Gesichtserkennungs-Algorithmus das spezifische Dataset für Mimik zusammengestellt.

### Gankörperbilder

Die emotionellen Ausdrücke, die innerhalb bestimmter Zeitmarken (Timestamps) dokumentiert wurden, lassen sich durch den Einsatz des bereitgestellten Codes in Einzelbilder zerlegen und mit einer Rate von 24 Frames pro Sekunde abspeichern. Interviews mit insgesamt fünf Personen wurden für diese Zwecke geführt. Die resultierenden Einzelbilder der Framesschnitte sind im [Ordner](Support_Code/Körpersprache/Fullbody_Framsplit) zu finden. Nach diesem Prozess erfolgte eine manuelle Selektion, um Bilder, die für die Analyse ungeeignet waren, auszuschließen.

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
   "freude": "Pfad",
   "angst": "Pfad"
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
Zur Detektion und Extraktion von Gesichtern aus dem Ganzkörper-Dataset wurde MTCNN (Multi-task Cascaded Convolutional Networks) eingesetzt. Nach der automatischen Extraktion erfolgte eine manuelle Ausselektierung ungeeigneter Bilder. Den vollständigen Code für diesen Prozess finden Sie [hier](Support_Code/Körpersprache/Mimik).


Für die Verarbeitung werden die Bibliotheken MTCNN für die Gesichtserkennung, os für Betriebssystem-interaktionen und cv2 für die Bildverarbeitung importiert. Anschließend erfolgt die Festlegung der notwendigen Pfade zur Organisation und Zugriff auf die Daten.

```python
import cv2
import os
from mtcnn import MTCNN

# Basispfade
basis_bildordner = "C:/Thesis/Data/Frames/Fullbody_sorted_Body_Language/"
basis_emotionsordner = "C:/Thesis/Data/Frames/Mimik/"
```

Die Gesichtserkennung auf den Ganzkörperbildern wird mithilfe des MTCNN-Algorithmus durchgeführt. Dieser erkennt und extrahiert Gesichter aus den Bildern für die weitere Verarbeitung.

```python
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
```


Um den Bildrahmen für eine vollständige Ansicht des Gesichts zu erweitern und sicherzustellen, dass Stirn und Haare nicht abgeschnitten werden, wurde eine Anpassung vorgenommen. Diese Erweiterung gewährleistet, dass die Gesichtserkennung alle relevanten Merkmale inkludiert. 

```python
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
Um die Datenvariation zu erhöhen und die Generalisierbarkeit des Modells zu verbessern, wurden externe Datensätze hinzugefügt und konkateniert. Für die Mimikanalyse war dies nicht notwendig, da das selbst erstellte Datenset bereits ausreichend umfangreich war. Für die Analyse der *Körpersprache* wurden jedoch folgende externe Datensätze verwendet:

- Das **BEAST-Set** (Brain Emotional Activation of Survival Threats) von Beatrice de Gelder. Der Download ist [hier](http://www.beatricedegelder.com/beast.html) verfügbar.
- Der **GEMEP-Corpus** (The GEneva Multimodal Emotion Portrayals) von Bänziger und Scherer, zugänglich nach Einreichung einer Einverständniserklärung [hier](https://www.unige.ch/cisa/gemep).

#### BEAST-SET
Für die Verarbeitung des Datensets von De Gelder wurden die entsprechenden Bilder gemäß ihrer Emotionen sortiert und in die jeweiligen Trainingsdatenordner einsortiert. Die Dateinamen wurden mit spezifischen Kürzeln versehen, um die zugehörige Emotion zu kennzeichnen:

- **AN**: Wut
- **FE**: Angst
- **HA**: Freude
- **SA**: Trauer

Mit dem folgenden Code werden die Bilder in die entsprechenden Emotionsordner sortiert:
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
Für die Audioanalyse im Rahmen dieses Projekts wurde kein eigenes Datenset erstellt, stattdessen wurden drei verschiedene externe Datensets genutzt:

- **RAVDESS Dataset**: Zugänglich [hier](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio). 

- **EMO-Database**: Verfügbar [hier](https://www.kaggle.com/datasets/piyushagni5/berlin-database-of-emotional-speech-emodb). 

- **GEMEP-Corpus**: Der Download ist [hier](https://www.unige.ch/cisa/gemep) möglich.

##### RAVDESS Dataset
Das RAVDESS-Dataset (The Ryerson Audio-Visual Database of Emotional Speech and Song) enthält Aufnahmen, die spezifisch auf die Emotionen Ekel, Angst, Trauer und Freude fokussieren.Es umfasst 7.356 Dateien mit einem Gesamtvolumen von 24.8 GB, eingesprochen von 24 professionellen Schauspielern (12 weiblich, 12 männlich), die zwei lexikalisch abgestimmte Aussagen in einem neutralen nordamerikanischen Akzent darbieten. Alle Dateien liegen im `.wav` Format vor, welches sich optimal für die Feature-Extraktion eignet.

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
Die EMO-DB (Berlin Database of Emotional Speech) ist ein umfangreiches Datenset für die Analyse von Sprachemotionen, bestehend aus deutschen Sprachaufnahmen, die Emotionen wie Glück, Traurigkeit, Wut, Angst, Ekel, Überraschung und neutrale Zustände wiedergeben. Es beinhaltet ca. 535 Aufnahmen von 10 Sprechern und benötigt rund 500 MB Speicherplatz. Für dieses Projekt wurden lediglich die vier relevanten Emotionen berücksichtigt. Die Dateinamen sind mit einem Buchstaben zur Kennzeichnung der Emotion versehen, z.B. `03a05Aa.wav`, wobei der vorletzte Buchstabe die Emotion angibt (A steht für Angst). Der folgende Code wurde verwendet, um die EMO-DB entsprechend zu sortieren und auszuwählen.

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
Für die Analyse der Mimik und Körpersprache werden vortrainierte Netzwerke eingesetzt, während für die Sprachanalyse eine Feature-Extraktion zur Anwendung kommt. Die extrahierten Features werden anschließend als Tensoren in ein Convolutional Neural Network (CNN) eingespeist, um eine detaillierte Analyse durchzuführen.
### Mimik und Körpersprache
Für die Auswahl eines geeigneten vortrainierten Netzwerks zur Analyse von Mimik und Körpersprache wurden diverse Netze getestet: `ResNet50`, `ResNet101`, `VGG16`, `VGG19`, `InceptionV3`, `DenseNet169` und `EfficientNetB7`. Diese können im Ordner `Model` eingesehen werden. Zudem kamen Data-Augmentation-Techniken wie *Rotation*, *RandomFlip* und *Kontrastanpassung* zum Einsatz. Diese Modelle und Architekturen wurden speziell für die Analyse von Mimik und Körpersprache verwendet.

Beispiel für das `DensetNet169`.

Import von Framworkes und Packages
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
```


*Data Augmentation* wird durchgeführt, um die Variabilität der Trainingsdaten zu erhöhen, indem Bilder gedreht, gespiegelt und im Kontrast angepasst werden. Dabei werden moderate Einstellungen verwendet, um sicherzustellen, dass das Modell weiterhin effektiv lernt, ohne zu underfitten.
```python
# Data Augmentation Layer hinzufügen
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),  # Rotation um bis zu 20%
    tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),  # Horizontaler und vertikaler Flip
    tf.keras.layers.experimental.preprocessing.RandomContrast(0.1),  # Zufällige Kontrastanpassung
])

```

Das Preprocessing der Daten erfolgt mit der Keras-Bibliothek. Hierbei werden die Bilder für das Training des neuronalen Netzwerks vorbereitet, indem sie skaliert und normalisiert werden. Dieser Schritt ist entscheidend, um sicherzustellen, dass das Modell die Daten effizient verarbeiten kann und die Input-Daten in einem einheitlichen Format vorliegen.

```python
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

# Betitelung für Tensorboard
experiment_name = "DenseNet169"
```
Das *DenseNet169*-Modell wird geladen und die Basisschichten für das Fine-Tuning eingefroren. Durch das Einfrieren dieser Schichten werden die vortrainierten Gewichte beibehalten, während nur die oberen Schichten des Netzwerks für die spezifische Aufgabe trainiert werden. Dieser Ansatz ermöglicht es, von den bereits gelernten Merkmalen des Netzwerks auf umfangreichen Datensätzen zu profitieren und gleichzeitig die Modellanpassung auf die spezifischen Anforderungen der aktuellen Aufgabe zu fokussieren.

```python
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
```

Das Modell nutzt den Optimierer 'adam' für effizientes Lernen durch adaptive Lernraten, kompiliert mit 'sparse_categorical_crossentropy' für die Verlustfunktion und 'accuracy' als Leistungsmetrik. TensorBoard wird für die Echtzeit-Visualisierung von Trainingsmetriken hinzugefügt, was die Optimierung und Anpassung des Trainingsprozesses erleichtert.

```python
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
Für die Feature-Extraktion aus Audiodateien wird die `librosa`-Bibliothek eingesetzt. Die extrahierten Features umfassen:

- **MFCCs (Mel-Frequenz-Cepstralkoeffizienten):** Eine Darstellung der Kurzzeit-Leistungsspektren von Sound, basierend auf der Mel-Skala. Sie sind wichtig für die Sprach- und Musikerkennung, da sie die wahrgenommenen Frequenzen des menschlichen Gehörs nachahmen.

- **Mel-Spektrogramm:** Eine visuelle Darstellung des Spektrums der Frequenzen eines Signals, wie sie über die Zeit variieren, ebenfalls basierend auf der Mel-Skala. Es wird oft in der Musik- und Spracherkennung verwendet, um die Energieverteilung bei verschiedenen Frequenzen zu zeigen.

- **Chromagramm:** Ein Diagramm, das die Intensität der zwölf verschiedenen Tonhöhenklassen oder Chroma zeigt, unabhängig von der Oktavlage. Es ist nützlich für die Analyse von Musik, da es Informationen über die Harmonie liefert.

- **Spektraler Kontrast:** Misst den Kontrast in den spektralen Spitzen und Tälern des Signals. Es wird verwendet, um unterschiedliche Klangtexturen zu differenzieren und kann zur Genreerkennung oder Instrumentenerkennung beitragen.

- **Tonnetz:** Eine Darstellung von harmonischen Beziehungen zwischen Tönen. Es basiert auf Tonhöhenklassen und wird oft für die Analyse von musikalischer Harmonie und Tonart verwendet.

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
```
Laden der Data und Labeln der vier Emotionen
```python
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
```

Die extrahierten Features werden als 1-dimensionaler Vektor in ein Convolutional Neural Network (CNN) eingespeist. Um Overfitting aufgrund der begrenzten Datenmenge zu verhindern, kommen Dropout und L2-Regularisierung zum Einsatz. Dropout reduziert die Komplexität des Modells, indem zufällig Ausgaben von Neuronen während des Trainings null gesetzt werden, was eine zu starke Anpassung an die Trainingsdaten verhindert. Die L2-Regularisierung fügt der Verlustfunktion einen Term hinzu, der große Gewichte bestraft, und fördert so ein einfacheres Modell mit glatteren Gewichtsverteilungen.

```python
# CNN Modelldefinition mit L2-Regularisierung und Droppout
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
```
Initialiserung des Models

```python
# Hauptskript
base_path = r'C:\Users\zastr\Desktop\Thesis\Data\Audio\Emo_DB'
features, labels = load_data_and_labels(base_path)
X_train, X_temp, y_train, y_temp = train_test_split(features, labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.67, random_state=42)

model = build_model((193, 1), len(np.unique(labels)))
```
Um optimale Testergebnisse zu erzielen und Overfitting zu reduzieren, wird eine reduzierte Lernrate verwendet. Zusätzlich wird Early Stopping eingesetzt, um den optimalen Zeitpunkt für den Abbruch des Trainings zu bestimmen, basierend auf der Leistung der Validierungsdaten. Diese Strategie hilft dabei, den Punkt zu identifizieren, an dem das Modell am besten generalisiert, bevor es beginnt, sich zu sehr an die Trainingsdaten anzupassen.

```python
# Callbacks zur Anpassung der Lernrate und zum frühzeitigen Stopp, falls keine Verbesserung mehr stattfindet
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Modelltraining mit Early Stopping und Learning Rate Reduction
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32, callbacks=[reduce_lr, early_stopping])
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc}, Test loss: {test_loss}')
```


## Testdata für Korrelationsanalyse
Für die Korrelationsanalyse wurde ein separates Test-Dataset erstellt, um die Kongruenz zwischen Mimik, Körpersprache und sprachlichen Äußerungen zu untersuchen. Dieses Dataset besteht aus spezifisch ausgewählten Datensätzen, die die Emotionen einer Person zu einem exakten Zeitpunkt erfassen, mit Einzelbildern für Mimik und Körpersprache sowie Audiosequenzen für sprachliche Äußerungen (2 bis 10 Sekunden lange Monologe während der Phase des emotionalen Ausdrucks). Ziel ist es zu analysieren, ob und wie stark diese Expressionen übereinstimmen.

### Testset für Sprache
Für die Videoverarbeitung wird das Paket `moviepy` benötigt.
```bash
pip install moviepy
```

Das Audio-Testset wurde präzise zugeschnitten und entsprechend den Emotionskategorien in die zugehörigen Testdatenordner einsortiert.
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
```

Initialisierung der Pfade und Daten erfolgt hier exemplarisch anhand einer ausgewählten Stelle aus den Videos. Den vollständigen Code für die Verarbeitung von Sprache, Mimik und Körpersprache finden Sie [hier](Support_Code/Evaluation/Testdata/testdaten_schneiden.ipynb).
```python
# Basispfad, wo die Originaldateien gespeichert sind
basis_pfad = "C:\\Users\\zastr\\Desktop\\Thesis\\Data\\Audio"

# Zielordner für jede Emotion
ziel_ordner_freude = "C:\\Users\\zastr\\Desktop\\Thesis\\Data\\Testdata\\Sprache\\Freude"
pfad_trauer = "C:\\Users\\zastr\\Desktop\\Thesis\\Data\\Testdata\\Sprache\\Trauer"
pfad_ekel = "C:\\Users\\zastr\\Desktop\\Thesis\\Data\\Testdata\\Sprache\\Ekel"
pfad_angst = "C:\\Users\\zastr\\Desktop\\Thesis\\Data\\Testdata\\Sprache\\Angst"

# Beispiel der Tabelle, die Sie bereitgestellt haben
tabelle_freude = [["Benny(Freude)", "3:08-3:14"]]

# Trauer
tabelle_trauer = [ ["Benny(Trauer)", "1:53-2:04"]]

# Ekel
tabelle_ekel = [["Benny(Ekel)", "2:26-2:31"]]

# Angst
tabelle_angst = [ ["Benny(Angst)", "1:10-1:18"]]
```
Funktion aufrufen 
```python

# Funktion aufrufen für jede Emotion mit dem entsprechenden Zielordner und Emotionsnamen
schneiden_und_verschieben(tabelle_freude, basis_pfad, ziel_ordner_freude, "Freude")
schneiden_und_verschieben(tabelle_trauer, basis_pfad, pfad_trauer, "Trauer")
schneiden_und_verschieben(tabelle_ekel, basis_pfad, pfad_ekel, "Ekel")
schneiden_und_verschieben(tabelle_angst, basis_pfad, pfad_angst, "Angst")
```


### Testset für Mimik und Körper
Das Testset für Mimik und Körpersprache wird mit dem angegebenen Algorithmus zugeschnitten. Eine vollständige Übersicht aller verwendeten Schneidealgorithmen ist über den angegebenen Link zugänglich. Zusätzlich wird das Paket `shutil` benötigt, um Funktionen wie `shutil.move` für die Dateiverwaltung einzusetzen.

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

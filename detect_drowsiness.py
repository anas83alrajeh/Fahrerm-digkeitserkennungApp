import cv2
import numpy as np
import tensorflow as tf
from playsound import playsound
import threading

# Modell laden (trainiertes Keras-Modell)
modell = tf.keras.models.load_model('trained_model.keras')
klassen = ['Drowsy', 'NonDrowsy']  # Klassenreihenfolge wie beim Training

# Schwellenwert und Einstellungen
schwelle = 0.6  # Wahrscheinlichkeitsschwelle für "Drowsy"
alarm_frame_limit = 5  # Anzahl aufeinanderfolgender Frames zur Alarmierung
frame_ueberspringen = 2  # Jeden 2. Frame überspringen zur Leistungsoptimierung

muedigkeits_zaehler = 0
frame_id = 0

# Funktion zur asynchronen Wiedergabe des Alarmsounds
def alarm_ton_abspielen():
    threading.Thread(target=playsound, args=('alert.mp3',), daemon=True).start()

# Webcam starten
kamera = cv2.VideoCapture(0)

while True:
    ret, bild = kamera.read()
    if not ret:
        break

    frame_id += 1
    if frame_id % frame_ueberspringen != 0:
        continue

    # Bild vorverarbeiten
    bild_rgb = cv2.cvtColor(bild, cv2.COLOR_BGR2RGB)
    bild_geglättet = cv2.GaussianBlur(bild_rgb, (5, 5), 0)
    bild_skaliert = cv2.resize(bild_geglättet, (224, 224))
    bild_normalisiert = bild_skaliert / 255.0
    eingabe = np.expand_dims(bild_normalisiert, axis=0)

    # Vorhersage des Modells
    vorhersage = modell.predict(eingabe, verbose=0)[0]
    wahrscheinlichkeit_muedigkeit = vorhersage[klassen.index('Drowsy')]
    vorhergesagte_klasse = 'Drowsy' if wahrscheinlichkeit_muedigkeit > schwelle else 'NonDrowsy'

    # Alarmlogik basierend auf mehreren Frames
    if vorhergesagte_klasse == 'Drowsy':
        muedigkeits_zaehler += 1
        if muedigkeits_zaehler >= alarm_frame_limit:
            alarm_ton_abspielen()
    else:
        muedigkeits_zaehler = 0

    # Ausgabe im Videofenster
    vertrauen = wahrscheinlichkeit_muedigkeit if vorhergesagte_klasse == 'Drowsy' else vorhersage[klassen.index('NonDrowsy')]
    farbe = (0, 0, 255) if vorhergesagte_klasse == 'Drowsy' else (0, 255, 0)
    cv2.putText(bild, f"{vorhergesagte_klasse} ({vertrauen:.2f})", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, farbe, 3)

    # Fenster anzeigen
    cv2.imshow('Fahrerüberwachung', bild)

    # Beenden mit der Taste 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

kamera.release()
cv2.destroyAllWindows()

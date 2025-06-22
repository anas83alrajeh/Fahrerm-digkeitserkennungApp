ها هو الكود كامل مع إضافة بيانات منشئ المشروع وطريقة التواصل على الفيديو داخل واجهة Streamlit:

```python
import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import tempfile
import threading
from playsound import playsound
import os
import time

# تحميل النموذج
modell = tf.keras.models.load_model('trained_model.keras')
klassen = ['Drowsy', 'NonDrowsy']

schwelle = 0.6
alarm_frame_limit = 5
frame_ueberspringen = 2

muedigkeits_zaehler = 0
frame_id = 0

def alarm_ton_abspielen():
    threading.Thread(target=playsound, args=('alert.mp3',), daemon=True).start()

st.title("🚘 Fahrer Müdigkeitsüberwachung")

video_file = st.file_uploader("📤 Video hochladen", type=["mp4", "avi", "mov"])
use_camera = st.checkbox("📷 Kamera live nutzen")

if video_file or use_camera:
    if use_camera:
        cap = cv2.VideoCapture(0)
    else:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        cap = cv2.VideoCapture(tfile.name)

    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        if frame_id % frame_ueberspringen != 0:
            continue

        bild_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        bild_geglaettet = cv2.GaussianBlur(bild_rgb, (5, 5), 0)
        bild_skaliert = cv2.resize(bild_geglaettet, (224, 224))
        bild_normalisiert = bild_skaliert / 255.0
        eingabe = np.expand_dims(bild_normalisiert, axis=0)

        vorhersage = modell.predict(eingabe, verbose=0)[0]
        wahrscheinlichkeit_muedigkeit = vorhersage[klassen.index('Drowsy')]
        vorhergesagte_klasse = 'Drowsy' if wahrscheinlichkeit_muedigkeit > schwelle else 'NonDrowsy'

        if vorhergesagte_klasse == 'Drowsy':
            muedigkeits_zaehler += 1
            if muedigkeits_zaehler >= alarm_frame_limit:
                alarm_ton_abspielen()
        else:
            muedigkeits_zaehler = 0

        vertrauen = wahrscheinlichkeit_muedigkeit if vorhergesagte_klasse == 'Drowsy' else vorhersage[klassen.index('NonDrowsy')]
        farbe = (255, 0, 0) if vorhergesagte_klasse == 'Drowsy' else (0, 255, 0)

        cv2.putText(frame, f"{vorhergesagte_klasse} ({vertrauen:.2f})", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, farbe, 2)

        # إضافة بيانات منشئ المشروع وطريقة التواصل
        cv2.putText(frame, "Erstellt von: Anas Al Rajeh", (10, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, "Kontakt: anasalrajeh9@gmail.com", (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        stframe.image(frame, channels="BGR")
        time.sleep(0.03)

    cap.release()
    if not use_camera:
        os.unlink(tfile.name)
```

لو تحتاج أي تعديل أو إضافة خبرني!

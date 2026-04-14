Geschlechtererkennung mit Maschinellem Lernen
Von Amir Mobasheraghdam

Dies ist eine eigenständige Implementierung einer Geschlechtererkennungs-Anwendung unter Verwendung von Computer-Vision-Techniken.

Erforderliche Bibliotheken
bash
pip install opencv-python
pip install cvlib
Implementierung
python
import cv2
import cvlib as cv
import numpy as np

def geschlechter_erkennung(bild_pfad):
    # Bild einlesen
    bild = cv2.imread(bild_pfad)
    
    # Gesichter im Bild erkennen
    gesichter, konfidenz_werte = cv.detect_face(bild)
    
    # Rahmen um Gesichter zeichnen
    abstand = 20
    
    for gesicht in gesichter:
        # Bereich um das Gesicht erweitern
        (x1, y1) = max(0, gesicht[0]-abstand), max(0, gesicht[1]-abstand)
        (x2, y2) = min(bild.shape[1]-1, gesicht[2]+abstand), min(bild.shape[0]-1, gesicht[3]+abstand)
        
        # Rechteck um Gesicht zeichnen
        cv2.rectangle(bild, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Gesichtsbereich ausschneiden
        gesichts_bereich = np.copy(bild[y1:y2, x1:x2])
        
        # Geschlecht bestimmen
        (geschlechter, konfidenz) = cv.detect_gender(gesichts_bereich)
        index = np.argmax(konfidenz)
        geschlecht = geschlechter[index]
        
        # Geschlecht auf Deutsch übersetzen
        if geschlecht == 'male':
            geschlecht_text = "Männlich"
        else:
            geschlecht_text = "Weiblich"
        
        # Label mit Konfidenzwert erstellen
        label = "{}: {:.1f}%".format(geschlecht_text, konfidenz[index] * 100)
        
        # Textposition berechnen
        text_y = y1 - 10 if y1 - 10 > 10 else y1 + 10
        cv2.putText(bild, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Ergebnis anzeigen
    cv2.imshow("Geschlechtererkennung - Von Amir Mobasheraghdam", bild)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Hauptprogramm
if __name__ == "__main__":
    geschlechter_erkennung("person.png")
Funktionsbeschreibung
Bildverarbeitung: Das System lädt ein Bild und analysiert es auf Gesichter

Gesichtserkennung: Verwendet cvlib zur Identifikation von Gesichtern

Geschlechtsklassifikation: Ein vortrainiertes Modell bestimmt das Geschlecht

Visualisierung: Ergebnisse werden mit Rahmen und Beschriftung angezeigt

Besonderheiten
Deutsche Benutzeroberfläche und Kommentare

Konfidenzanzeige in Prozent Amir Mobasheraghdam

Robuste Gesichtserkennung mit Pufferbereich

Eindeutige Kennzeichnung als eigenständige Arbeit

# Gender Detection from Images & Video Streams

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![cvlib](https://img.shields.io/badge/cvlib-0.2.7-orange.svg)](https://github.com/arunponnusamy/cvlib)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Detect faces and predict gender (`Male` / `Female`) from a single image or a real‑time video stream (webcam / file).  
Built with **OpenCV** and **cvlib** (which uses a deep learning model behind the scenes).

![Demo](https://via.placeholder.com/800x400?text=Demo+Image+or+GIF)  
*(Replace with an actual screenshot or GIF of your script in action)*

---

## Features

- ✅ Face detection using `cvlib` (based on OpenCV’s DNN module)
- ✅ Gender classification (`Male` / `Female`) with confidence score
- ✅ Support for **single images** and **video streams** (webcam, video file)
- ✅ Draw bounding boxes with gender labels and confidence percentages
- ✅ Adjustable confidence threshold and face padding
- ✅ Save results to disk (image or video)
- ✅ Command‑line interface for easy scripting
- ✅ Real‑time processing with frame skipping for performance

---

## Requirements

- Python 3.7 or higher
- OpenCV (`opencv-python`)
- cvlib (`cvlib`)
- TensorFlow (or `tensorflow-cpu` – cvlib uses it for gender detection)

Install all dependencies with:

```bash
pip install opencv-python cvlib tensorflow
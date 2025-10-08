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

Konfidenzanzeige in Prozent

Robuste Gesichtserkennung mit Pufferbereich

Eindeutige Kennzeichnung als eigenständige Arbeit

Verwendung
Ersetzen Sie "person.png" mit Ihrem gewünschten Bilddateinamen und führen Sie das Skript aus.

# Bachelorarbeit: Beurteilung des Wohlergehens von Mäusen mit Hilfe von Videoanalyse durch tiefe neuronale Netze
Die automatisierte Beurteilung des Wohlergehens von Mäusen ist ein wichtiger Faktor zur weiteren Verbesserung des Tierschutzes in der tierexperimentellen Forschung.
Ein Indikator für ein beeinträchtigtes Wohlergehen der Mäuse kann dabei unter anderem das Vorhandensein schmerztypischer Bewegungsmuster sein.
Der vorgeschlagene Ansatz zur zweistufigen Bewegungsanalyse von Mäusen zielt darauf ab, diese Bewegungsmuster in einem postoperativen Umfeld zu erkennen und zeitlich einzuordnen.
Dafür werden im ersten Schritt 35 Skelettpunkte aus Videos von Mäusen, die einer *plantar incision* Operation unterzogen wurden, extrahiert.
Im zweiten Schritt werden die Bewegungsmuster in den Zeitreihen der Koordinaten der Skelettpunkte durch ein *CNN* erkannt.
Dabei werden *1-Stream-CNNs* und *2-Stream-CNNs*, sowie die Verwendung einer Skelett-Transformator-Schicht untersucht.
Auf den gegebenen Daten kann so eine Genauigkeit von bis zu 91,7% bei aus dem Training exkludierten Videos erreicht werden.

## Setup
Für den ersten Schritt werden *DeepLabCut* und die mitgelieferten *TensorFlow* und *napari* Versionen benötigt.
*DeepLabCut* Installation siehe [hier](https://deeplabcut.github.io/DeepLabCut/docs/installation.html).

Für den zweiten Schritt wird die *TensorFlow* Version `2.11.0` verwendet.

Für beide Schritte wird eine kompatible *CUDA* Installation empfohlen (siehe [hier](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html))

## Extraktion der Skelettpunkte
Im ersten Schritt dieses zweistufigen Ansatzes zur Bewegungsanalyse werden die Koordinaten von relevanten Skelettpunkten aus den vorliegenden Videos extrahiert.

Es wird zunächst überprüft, ob die vortrainierten Modelle des *DeepLabCut Model Zoo* ausreichend genaue Ergebnisse liefern.
```python
# taken from supermodel_inference_test.py
videopaths = ["<paths>"]
superanimal_name = 'superanimal_quadruped'
scale_list = []

deeplabcut.video_inference_superanimal(
    video_paths, superanimal_name, scale_list=scale_list, video_adapt=True, videotype="mp4")
```
Da die Ergebnisse nicht zufriedenstellend sind, wird ein neues Modell für den verwendeten Datensatz trainiert.

Die dafür nötigen Schritte sind in `DLC_setup_and_training.ipynb` zu finden.
Der Datensatz für das Training wird durch das Labeln von 320 Bildern in `napari` erstellt.

## Klassifizierung der Bewegungen
Im zweiten Schritt der vorgeschlagenen Methodik klassifiziert ein zweites KI-Modell die Zeitreihen der Skelettkoordinaten in die 4 vorbestimmten Zeitpunkte.
Dazu werden zunächst die Daten aus dem ersten Schritt eingelesen und vorverarbeitet.
Anschließend werden diese Daten zum Training eines *CNNs* genutzt.
Die Evaluation der Ergebnisse erfolgt auf 12 aus dem Training exkludierten Videos.

Die Schritte sind in `action_recognition.ipynb` nachzuverfolgen.

## Ergebnisse
Die Genauigkeit der empfohlenen trainierten Modelle liegt bei der Klassifizierung aus dem Training exkludierter Videos bei 75% bis 91,7%.
Die empfohlene Daten-, Modell- und Trainingskonfiguration ist:
- 1-Stream Modell mit den absoluten Positionen als Input
- 250 Frames lange Abschnitte mit 150 Frames Überlappung
- Architektur:
    - Skelett-Transformator-Schicht
    - *Convolutional* Schicht mit 16 Filtern a (10 x 50)
    - *Maxpooling* Schicht mit (2, 4) *pooling*
    - *Convolutional* Schicht mit 8 Filtern a (5 x 20)
    - *Flatten* Schicht
    - *Dense* Schicht mit *Softmax* Aktivierungsfunktion und 4 output Neuronen

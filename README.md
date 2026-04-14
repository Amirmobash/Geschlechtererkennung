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
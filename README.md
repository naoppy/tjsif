Surveillance Camera to Track People
====

# Demo
past movie which shows our work

# Overview
Detects only people with AI and records while tracking people with a camera.

# Description
### System specification
The specification of this system is written [here](tjsif_flowchart.svg).

### How to extract image changes
The method of extracting image changes uses weighted accumulation on the image. Then binarize.
Judgment is made based on whether the area extracted by contour extraction exceeds a certain level.

### How to speed up human detection
Use Edge TPU and connect to the USB3.0 terminal on the RaspberyPi.
Then use [TensorFlow Lite Interpreter](https://www.tensorflow.org/lite/guide/python).
The model is [this one](all_models/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite)

### How to control Powerful Motor
Unlike a DC motor, a large current flows through the bipolar motor and its control circuit.

So I used a transistor that can withstand a large current and soldered a circuit like an H-bridge circuit.

The circuit is [this]().

# Requirements
- RaspberryPi4
- Coral USB Accelerator(Edge TPU)
- Bipolar Motor
- WebCamera(Anything is OK)

# Requirement Packages
- OpenCV4.0
- numpy
- PIL(Pillow)
- Tensorflow Lite Library(runtime only)
- RPi.GPIO

# Usage
```bash
$ git clone https://github.com/naoppy/tjsif.git
$ python3 products/main.py
```

# Author
- [naoppy](https://github.com/naoppy)
- [toshimitsu sakai](https://github.com/sakai36)
# Kalman-Tracker - Object Tracking using Kalman Filter Approach


https://github.com/nathanaday/Kalman-Tracker/assets/79942554/97e959e2-08cf-41ea-804b-208272bf72bf


## Project Description

This project demonstrates the benefits of applying Kalman filtering algorithms for real-time object tracking applications, where spatial awareness of the object is dependent on slower detection models prone to missing frames and producing noise.
- Enhance Accuracy: Reduce the noise in object positions detected by TensorFlow models
- Robust Tracking: Track objects in real-time motion without relying on speed of the object detection model
- Increase Reliability: Maintain prediction of object location even when the detection misses a frame or the object is occluded

A custom Detection Manager class is implemented to maintain awareness that objects detected between two inferences are the same object if they have a matching category and a size similiarity within a certain tolerance. This allows the same object to be tracked over time, instead of treating each observation as a one-time event.

Using the Detection Manager class, TensorFlow detection results are registered as detections, and the Kalman filter instance for each detection is updated in state. The kalman filter predictions can be queried much faster than the observation updates from TensforFlow, so real-time object estimations can be displayed at the same framerate as the video stream

An OpenCV video capture can provide frames to the detection class for inference, while the detection predictions are displayed in real-time.



### Dependencies

This project requires the following libraries and frameworks:

    Python 3.8+
    TensorFlow 2.x
    NumPy
    opencv-Python
    filterpy

Install depencies directly:

`pip install tensorflow numpy opencv-python filterpy`

or use the requirements file:

`pip install -r requirements.txt`

### Usage

1. Clone the repository

```
git clone https://github.com/nathanaday/Kalman-Tracker.git
cd object-tracking-kalman
```


2. Locally save a tensorflow detection model. This project has been tested with `CenterNet Resnet50 V1 FPN 512x512`

[Download model .tar.gz from TF Hub](http://download.tensorflow.org/models/object_detection/tf2/20200711/centernet_resnet50_v1_fpn_512x512_coco17_tpu-8.tar.gz)

Tensorflow model zoo:
```
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md
```


3. Specify the path to the saved model in the main.py entry-point:

```
if __name__ == "__main__":
    
    # model_path = ".../tensorflow_datasets/centernet_resnet101_v1_fpn_512x512_coco17_tpu-8/saved_model"
    model_path = " Replace with path to saved_model directory "
```


4. Run the application:

```
python main.py
```



### Kalman Filter Overview

The Kalman filter is a recursive algorithm used in linear dynamic systems to estimate the state of a linear process. It is particularly effective in situations where the measurements of the system are noisy or incomplete.

[https://en.wikipedia.org/wiki/Kalman_filter](https://en.wikipedia.org/wiki/Kalman_filter)

In vision applications, a Kalman filter can significantly improve the tracking of moving objects, especially when measurements are noisy, or there are not enough processing resources to make real-time detection inferences. By applying the Kalman Filter, it is possible to obtain predicted knowledge of the object's location between inference times.




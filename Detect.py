import copy
import math
import random
import string
import threading

import cv2
import tensorflow as tf
print(tf.__version__)

import time

from Kalman import Kalman
from mscoco_label_map import category_map


def generate_random_id(length=6):
    characters = string.ascii_lowercase + string.digits  # Lowercase letters and digits
    random_id = ''.join(random.choices(characters, k=length))
    return random_id


class DetectionObject:

    def __init__(self, category, box_dims, center_dims, score):
        """
        Instance of a detected object, useful for capturing category identity, position, and time since last observed
        Also contains a Kalman instance for generating spatial predictions between observation points for real-time tracking

        Parameters:
            - category (str): Detection classification and category (i.e. "person", "car")
            - box_dims (tuple(int, int, int, int)): Bounding box coordinates in this order: y_min, x_min, y_max, x_max
            - center_dims (tuple(int, int)): Coordinate box geometric center in (center_x, center_y) order
            - score (float): the confidence score of the initial detection (for future improvement, some minimum score
                criteria can be established)

        Attributes:
            - unique_id (str): randomly generated 6-character string to uniquely identify object (for future use)
            - last_updated (float): timestamp of last observation update
            - last_prediction (float): timestamp of last time a prediction was generated from the kalman instance
            - kalman (Kalman): instance of the kalman filter class (see Kalman.py)
        """

        self.unique_id = generate_random_id()
        self.last_updated = None
        self.last_prediction = None

        self.kalman = Kalman()

        self.category = category
        self.score = score
        self.x_min = None
        self.x_max = None
        self.y_min = None
        self.y_max = None
        self.center_x = None
        self.center_y = None
        self.delta_time = None
        self.update_position(box_dims, center_dims)

    def update_position(self, box_dims, center_dims):
        """
        Updates the object with new spatial observation and updates (or initialized) Kalman filter instance.
        Only called when new observations are available, e.g. upon TensorFlow detection result
        """
        try:
            self.y_min, self.x_min, self.y_max, self.x_max = box_dims
            self.center_x, self.center_y = center_dims

            self.last_updated = time.time()

            if self.kalman.is_initialized:
                self.kalman.update(self.center_x, self.center_y)
            else:
                self.kalman.initialize_filter(self.center_x, self.center_y, 0.2)

        except ValueError as e:
            print(f"Failed to update position. Details: {e}")
            return

    def get_prediction(self):
        """ Fetches prediction from kalman filter instance (can be called in real-time process) """
        now = time.perf_counter()
        if self.last_prediction is None:
            delta_t = 0.03  # For first time pass only, uses typical camera framerate as first guess
        else:
            delta_t = now - self.last_prediction

        if self.kalman.is_initialized:
            self.last_prediction = now
            return self.kalman.predict(delta_t)
        else:
            print("Cannot make prediction: filter is not initialized")

    def get_bounding_box(self):
        return self.y_min, self.x_min, self.y_max, self.x_max

    def get_box_dims(self):
        height = self.y_max - self.y_min
        width = self.x_max - self.x_min
        return height, width

    def get_center_coord(self):
        return self.center_x, self.center_y


class DetectionObjectManager:

    def __init__(self, memory_seconds=3, max_distance_tolerance=100):
        """
        Parameters:
            - memory_seconds (float): how long to retain detected object ref based on elapsed time since last update
            - max_distance_tolerance (float): how far (in pixels) to allow this object to stray from last update before
            it is considered a different object

        Attributes:
            - detections (list(DetectionObject)): list of DetectionObject instances
        """

        self.detections = []  # Collection of DetectionObject instances
        self.memory_seconds = memory_seconds
        self.max_distance_tolerance = max_distance_tolerance

    def summary_dump(self):
        """ Can add in the main loop to print detection details in the console (useful for development) """
        print("-" * 40)
        print(f"Number of detection: {len(self.detections)}:")
        print([obj.category for obj in self.detections])
        print("-" * 40, "\n")

    def get_collection(self):
        """ Returns a copy of the detection list """
        return copy.deepcopy(self.detections)

    def add_detection_object(self, category, box_dims, center_dims, score):
        """ Instantiates new DetectionObject and adds it to the detections list """
        created_object = DetectionObject(category, box_dims, center_dims, score)
        self.detections.append(created_object)

    def is_within_distance(self, x1, y1, x2, y2):
        """ Checks if two coordinates (x1, y1), (x2, y2) are within the distance tolerance """

        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return distance <= self.max_distance_tolerance

    def lookup(self, category, center_dims):
        """ Given a category and location, finds an object in the collection that matches based on criteria """

        center_x, center_y = center_dims

        # 1. Create a list of category matches (if any)
        category_matches = [obj for obj in self.detections if obj.category == category]
        if len(category_matches) == 0:
            return

        # 2. Create a list of spatial matches (if any)
        spatial_matches = [obj for obj in category_matches if self.is_within_distance(obj.center_x, obj.center_y, center_x, center_y)]
        if len(spatial_matches) == 0:
            return

        # Returns unique_id of matching object; note, if multiple matches, returns only one index
        return spatial_matches[0]

    def register_detection(self, category: str, box_dims: tuple, center_dims: tuple, score: float):
        """ Commit a new detection to an existing object or instantiate a new detection object """

        # First look it up in the collection:
        matching_object = self.lookup(category, center_dims)
        if matching_object:
            # Update the object in place
            matching_object.update_position(box_dims, center_dims)
        else:
            # Register a new object
            self.add_detection_object(category, box_dims, center_dims, score)

    def import_detections(self, detections: dict):
        """ Takes TensorFlow detection payload and registers it into the class """
        for i, result in detections.items():
            category = result.get("category")
            box_dims = result.get("bounding_box")
            center_dims = result.get("center")
            score = result.get("score")
            self.register_detection(category, box_dims, center_dims, score)

        self.cleanup_collection()  # Prune old objects in the list

    def get_predictions(self):
        """ Formats and returns the prediction set for all objects in the detection list """

        predictions = {}
        for i, obj in enumerate(self.detections):
            height, width = obj.get_box_dims()
            center_x, center_y = obj.get_prediction()

            y_min = int(center_y - height / 2)
            y_max = int(center_y + height / 2)
            x_min = int(center_x - width / 2)
            x_max = int(center_x + width / 2)

            predictions[i] = {
                "bounding_box": (y_min, x_min, y_max, x_max),
                "center": (int(center_x), int(center_y)),
                "score": obj.score,
                "category": obj.category
            }
        return predictions

    def cleanup_collection(self):
        """ Purges collection of old detections based on memory_seconds property """
        current_time = time.time()
        self.detections = [
            detection for detection in self.detections
            if detection.last_updated is not None and (current_time - detection.last_updated) <= self.memory_seconds
        ]


class Detector:

    def __init__(self, model_path):
        """
        Tested using: centernet_resnet101_v1_fpn_512x512_coco17_tpu-8
        https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md

        A class for performing object detection on an image. Supports running automatic detections on any set frame in a
        separate thread, useful for making inferences on real-time data input.

        The detection model must be saved on the local machine and specified using the "model_path"

        Parameters:
            - model_path (string): Path to 'saved_model' directory with saved_model.pb file

        Attributes:
            - detect_fn (tf model): instance of the loaded TensorFlow model
            - label_map (dict): Mapping of classification id (int) to string names (i.e. "person"), imported from
            mscoco_label_map.py
            - frame (numpy(H, W, Channels)): OpenCV RGB image frame to perform detection on
            - results (dict): detection results dictionary, where key is the index, and value is in the format:
                parsed_results[i] = {
                    "bounding_box": tuple(y_min, x_min, y_max, x_max),
                    "center": tuple(center_x, center_y),
                    "score": float,
                    "category": string
                }
        """

        self.detect_fn = None
        try:
            self.detect_fn = tf.saved_model.load(model_path)
        except OSError as oser:
            print(f"Failed it load model in {self.__class__.__name__} class: {oser}")
            return

        self.label_map = category_map

        self.frame = None
        self.image_w = None
        self.image_h = None
        self.results = {}

        self.thread = None
        self.stopped = False

    def stop(self):
        self.stopped = True
        if self.thread:
            self.thread.join()

    def detect_threaded(self):
        if not self.thread:
            print("Starting detection thread...")
            self.stopped = False
            self.thread = threading.Thread(target=self.unattended_detect)
            self.thread.start()
        else:
            print("Thread is already running!")

    def unattended_detect(self):
        """
        Usage: This method can run threaded (see detect_threaded). use `set_frame(frame)` from outside the class, and
        the method will populate `self.results` as fast as it can make inferences

        """
        while not self.stopped:
            if self.frame is None:
                print(f"Missing `frame` attribute in unattended detect")
                continue
            self.results = self.detect(self.frame)

    def set_frame(self, frame):
        self.frame = frame

    def get_results(self):
        return self.results

    def detect(self, frame, max_results=5, min_score=0.5):
        """
        frame: height, width, channel color image (cv2)
        """

        if not self.detect_fn:
            print(f"Cannot run detection: model has not been loaded")
            return

        image_height, image_width, _ = frame.shape

        # Converts BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to tensor
        img_tensor = tf.convert_to_tensor(frame_rgb, dtype=tf.uint8)
        img_tensor = tf.expand_dims(img_tensor, 0)

        # Run the detector
        t1 = time.time()
        result = self.detect_fn.signatures['serving_default'](input_tensor=img_tensor)
        # print(f"Detection inference time: {time.time() - t1}")

        return self.parse_result(result, image_width, image_height, max_results, min_score)

    def parse_result(self, result, image_width, image_height, max_results, min_score):

        # Converts tensors to numpy array
        scores = result['detection_scores'][0].numpy()
        labels = result['detection_classes'][0].numpy().astype(int)
        boxes = result['detection_boxes'][0].numpy()

        # Map class IDs to class names
        class_names = [self.label_map.get(label, 'Unknown') for label in labels]

        parsed_results = {}

        for i in range(min(boxes.shape[0], max_results)):
            if scores[i] >= min_score:
                # Box bounding dimensions (based on model results and image dims) (tuple: left, right, top, bottom)
                ymin, xmin, ymax, xmax = tuple(boxes[i])

                # Image width, height

                xmin_scaled = xmin * image_width
                xmax_scaled = xmax * image_width
                ymin_scaled = ymin * image_height
                ymax_scaled = ymax * image_height

                bounding_box = (ymin_scaled, xmin_scaled, ymax_scaled, xmax_scaled)

                center_x = int((xmin + xmax) / 2 * image_width)
                center_y = int((ymin + ymax) / 2 * image_height)
                center = (center_x, center_y)

                score = scores[i]
                category = class_names[i]

                parsed_results[i] = {
                    "bounding_box": bounding_box,
                    "center": center,
                    "score": score,
                    "category": category
                }

        return parsed_results

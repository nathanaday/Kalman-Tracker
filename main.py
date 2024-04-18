import copy
import glob
import time

import cv2

from Detect import Detector, DetectionObjectManager
from Kalman import Kalman

from Display import Display


def main(detector: Detector, display: Display):

    # Initialize video capture
    cap = cv2.VideoCapture(0)  # 0 for the default camera

    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return

    try:
        ret, frame = cap.read()
        detector.set_frame(frame)
        detector.detect_threaded()

        if frame is not None:
            height, width, _ = frame.shape
        else:
            height, width = (1080, 1080)  # TODO default assumption?

        detection_manager = DetectionObjectManager(memory_seconds=0.5, max_distance_tolerance=int(width*0.3))

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Failed to capture frame. Exiting...")
                break

            print(type(frame))

            detector.set_frame(frame)

            # Detect objects in the frame

            # TODO the import detections would be moved to the detection class
            res = detector.get_results()
            detection_manager.import_detections(res)

            predictions = detection_manager.get_predictions()

            # detection_manager.summary_dump()

            # Display the frame and detections
            display.cvshow(frame, predictions)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                detector.stop()
                break
    finally:
        # Release the capture and close any open windows
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # model_path = "/Users/nathanaday/tensorflow_datasets/centernet_resnet101_v1_fpn_512x512_coco17_tpu-8/saved_model"
    model_path = r"C:\Users\naday\Documents\Tensorflow\centernet_resnet101_v1_fpn_512x512_coco17_tpu-8.tar\centernet_resnet101_v1_fpn_512x512_coco17_tpu-8\centernet_resnet101_v1_fpn_512x512_coco17_tpu-8\saved_model"
    det = Detector(model_path=model_path)

    disp = Display(wait=False)
    main(det, disp)

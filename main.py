import cv2

from Detect import Detector, DetectionObjectManager
from Display import Display
from config import model_path


def main(detector: Detector, display: Display):
    """
    Opens an OpenCV video capture and demonstrates object detection and kalman filter usage
    """

    if not isinstance(detector, Detector):
        raise ValueError(f"Input detector class in main is not an expected `Detector` class instance. "
                         f"It may not have initialized properly. "
                         f"This is usually due to issues with loading the tensorflow model")

    if not isinstance(display, Display):
        raise ValueError(f"Input display class in main is not an expected `Display` class instance. "
                         f"It may not have initialized properly.")

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
            print(f"Working with image dimensions: {frame.shape}")
        else:
            print(f"Failed to capture frame from source (problem encountered)")
            return

        detection_manager = DetectionObjectManager(memory_seconds=0.5, max_distance_tolerance=int(width*0.3))

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Failed to capture frame. Exiting...")
                break

            detector.set_frame(frame)

            # Detect objects in the frame
            res = detector.get_results()
            detection_manager.import_detections(res)

            predictions = detection_manager.get_predictions()

            # # # Optionally, if you want to see the summary of current detections in the console...
            # detection_manager.summary_dump()

            # Display the frame and detections
            # display.cvshow(frame, res)     # <-- To display results WITHOUT kalman filter
            display.cvshow(frame, predictions)  # <-- To display results WITH kalman filter

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                detector.stop()
                break
    finally:
        # Release the capture and close any open windows
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":

    # Instructions:
    #  1. Update 'model_path' in config.py with the path to your saved tensorflow model (detail in config.py)
    #  2. Run this file as main.py

    det = Detector(model_path=model_path)
    disp = Display(wait=False)
    main(det, disp)

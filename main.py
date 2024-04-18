import cv2

from Detect import Detector, DetectionObjectManager
from Display import Display


def main(detector: Detector, display: Display):
    """
    Opens an OpenCV video capture and demonstrates object detection and kalman filter usage
    """

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
            # display.cvshow(frame, res)
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
    model_path = "/Users/nathanaday/tensorflow_datasets/centernet_resnet101_v1_fpn_512x512_coco17_tpu-8/saved_model"
    # model_path = r"C:\Users\naday\Documents\Tensorflow\centernet_resnet101_v1_fpn_512x512_coco17_tpu-8.tar\centernet_resnet101_v1_fpn_512x512_coco17_tpu-8\centernet_resnet101_v1_fpn_512x512_coco17_tpu-8\saved_model"

    det = Detector(model_path=model_path)
    disp = Display(wait=False)
    main(det, disp)

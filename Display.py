import cv2


class Display:

    def __init__(self, wait=False, **kwargs):

        self.wait = wait  # While use cv2.waitKey() when true to wait for user interaction before next frame

        self.bounding_box_color = kwargs.get("bounding_box_color", (0, 255, 0))
        self.bounding_box_thickness = kwargs.get("bounding_box_thickness", 2)
        self.show_text = kwargs.get("show_text", True)
        self.text_color = kwargs.get("text_color", (36, 255, 12))
        self.text_size = kwargs.get("text_size", 0.6)
        self.text_weight = kwargs.get("text_weight", 2)

    def cvshow(self, image, results):
        for ids, details in results.items():
            bbox = details['bounding_box']
            score = details['score']
            category = details['category']

            center_x, center_y = details['center']
            # cv2.circle(image, (center_x, center_y), 5, (0, 20, 220), -1)

            # Draw bounding box
            # Bounding box coordinates are expected to be (ymin, xmin, ymax, xmax)
            start_point = (int(bbox[1]), int(bbox[0]))  # (xmin, ymin)
            end_point = (int(bbox[3]), int(bbox[2]))  # (xmax, ymax)
            cv2.rectangle(image, start_point, end_point, self.bounding_box_color, self.bounding_box_thickness)

            # Put the label and score on the image
            if self.show_text:
                label = f"{category}: {score:.2f}"
                cv2.putText(image,
                            label,
                            (int(bbox[1]), int(bbox[0] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            self.text_size,
                            self.text_color,
                            self.text_weight)

        cv2.imshow('Detected Image', image)

        if self.wait:
            cv2.waitKey(0)

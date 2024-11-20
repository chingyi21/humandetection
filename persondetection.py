import os
import cv2
import numpy as np
import tensorflow as tf
from datetime import datetime
import matplotlib.pyplot as plt


class DetectorAPI:
    def __init__(self, path_to_ckpt='frozen_inference_graph.pb'):
        self.path_to_ckpt = path_to_ckpt
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.sess = tf.compat.v1.Session(graph=self.detection_graph)

        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def process_frame(self, image):
        image_np_expanded = np.expand_dims(image, axis=0)
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded}
        )
        return np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes).astype(np.int32), int(num)

    def close(self):
        self.sess.close()


class HumanDetectionProcessor:
    def __init__(self, detector: DetectorAPI, threshold=0.7):
        self.detector = detector
        self.threshold = threshold

    def detect_humans_in_frame(self, frame):
        boxes, scores, classes, num = self.detector.process_frame(frame)
        human_boxes = [
            box for i, box in enumerate(boxes)
            if classes[i] == 1 and scores[i] > self.threshold
        ]
        return human_boxes

    def draw_boxes_on_frame(self, frame, boxes):
        for box in boxes:
            y1, x1, y2, x2 = box
            frame_height, frame_width = frame.shape[:2]
            start_point = (int(x1 * frame_width), int(y1 * frame_height))
            end_point = (int(x2 * frame_width), int(y2 * frame_height))
            color = (255, 0, 0)  # Blue color in BGR
            thickness = 2
            cv2.rectangle(frame, start_point, end_point, color, thickness)
        return frame


class ReportGenerator:
    @staticmethod
    def generate_enumeration_plot(detection_data):
        times = [data['time'] for data in detection_data]
        people_counts = [data['count'] for data in detection_data]

        plt.figure(figsize=(8, 6))
        plt.plot(times, people_counts, marker='o', color="blue", label="Human Count")
        plt.xlabel("Time (sec)")
        plt.ylabel("Human Count")
        plt.title("Enumeration Plot")
        plt.legend()
        plt.grid()
        plt.show()

    @staticmethod
    def generate_avg_accuracy_plot(detection_data):
        times = [data['time'] for data in detection_data]
        accuracies = [data['accuracy'] for data in detection_data]

        plt.figure(figsize=(8, 6))
        plt.plot(times, accuracies, marker='o', color="green", label="Avg. Accuracy")
        plt.xlabel("Time (sec)")
        plt.ylabel("Avg. Accuracy")
        plt.title("Average Accuracy Plot")
        plt.legend()
        plt.grid()
        plt.show()

    @staticmethod
    def generate_crowd_report(detection_data):
        max_count = max(detection_data, key=lambda x: x['count'])['count']
        max_time = max(detection_data, key=lambda x: x['count'])['time']
        avg_accuracy = sum(data['accuracy'] for data in detection_data) / len(detection_data)

        # Plot the report
        plt.figure(figsize=(6, 8))
        plt.axis('off')
        plt.title("CROWD REPORT", fontsize=20, color="#3e4660")

        text = (
            f"Max Human Limit: 25\n"
            f"Max Human Count: {max_count}\n"
            f"Max Accuracy: {max(detection_data, key=lambda x: x['accuracy'])['accuracy']:.4f}\n"
            f"Max Avg. Accuracy: {avg_accuracy:.4f}\n"
            f"Time of Peak Count: {max_time}\n\n"
            f"Status:\n{'Crowded' if max_count > 25 else 'Not Crowded'}"
        )
        plt.text(0.5, 0.5, text, ha='center', va='center', wrap=True, fontsize=12, color="#3e4660")

        # Save the report as an image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"Crowd_Report_{timestamp}.png")
        plt.show()


def main():
    """
    This script is designed to handle detection processes but does not include GUI logic.
    GUI elements (Tkinter or other frameworks) should invoke these classes as modules.
    """
    detector = DetectorAPI()
    processor = HumanDetectionProcessor(detector)

    # Example usage for a single frame (replace 'frame' with actual image data)
    frame = cv2.imread('example_image.jpg')  # Replace with a valid image path
    boxes = processor.detect_humans_in_frame(frame)
    output_frame = processor.draw_boxes_on_frame(frame, boxes)

    # Display the image with detected humans
    cv2.imshow('Human Detection', output_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    detector.close()


if __name__ == "__main__":
    main()

import cv2
from ultralytics import YOLO
import time
from torch.profiler import profile, record_function, ProfilerActivity

def main():
    # Load the model
    model = YOLO(r"D:\01-KULIAH\0-SEMESTER 8\01-Undergraduate Thesis\0-Laboratory\YOLOv8n Inference\detect\train13\weights\best.pt")

    # Start the webcam
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(r"D:\01-KULIAH\0-SEMESTER 8\01-Undergraduate Thesis\Datasets\Used For Research\Raw\YawDD Supplement\Dash\10-MaleGlasses.avi")  

    # Initialize the dictionary to keep track of detection times and last durations
    detections = {
        'closed-eyes': {'duration': 0, 'last_duration': 0, 'last_seen': None},
        'yawn': {'duration': 0, 'last_duration': 0, 'last_seen': None}
    }

    drowsy_state = False

    with profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=True,
        with_stack=True,  # This will capture the stack trace
        profile_memory=True,  # This will capture memory usage
    ) as prof:
        while True:
            with record_function("frame_capture"):
                ret, frame = cap.read()
                if not ret:
                    break

            with record_function("model_inference"):
                # Perform inference
                results = model.predict(frame, conf=0.6)

            # Current time in seconds
            current_time = time.time()

            # Track which classes are currently detected
            current_detections = set()

            # Draw the bounding boxes by iterating over the results
            for result in results:
                for r in result.boxes.data.tolist():
                    x1, y1, x2, y2, conf, class_id = r
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    class_name = model.names[int(class_id)]
                    current_detections.add(class_name)

                    with record_function("drawing"):
                        # Draw rectangle
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        # Display class label and confidence
                        cv2.putText(frame, f'{class_name} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 2)

                    with record_function("updating_durations"):
                        # Update detection times
                        if detections[class_name]['last_seen'] is None:
                            detections[class_name]['last_seen'] = current_time
                        else:
                            detections[class_name]['duration'] += current_time - detections[class_name]['last_seen']
                            detections[class_name]['last_seen'] = current_time

                        # Check if the driver is drowsy
                        if detections['closed-eyes']['duration'] > 0.5 or detections['yawn']['duration'] > 5:
                            drowsy_state = True
                            cv2.putText(frame, 'Drowsy', (500, 60), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
                        else:
                            drowsy_state = False

            with record_function("check_undetected"):
                # Check for classes not currently detected and update last_duration
                for class_name in detections.keys():
                    if class_name not in current_detections:
                        if detections[class_name]['last_seen'] is not None:
                            detections[class_name]['last_duration'] = detections[class_name]['duration']
                            detections[class_name]['duration'] = 0
                            detections[class_name]['last_seen'] = None

            with record_function("drawing_durations"):
                # Draw a white rectangle as a background for the class durations
                cv2.rectangle(frame, (0, 10), (200, 60), (255, 255, 255), -1)
                # Display the duration for each class
                y_offset = 30
                for class_name, times in detections.items():
                    duration = times['last_duration'] if times['last_seen'] is None else times['duration']
                    cv2.putText(frame, f'{class_name}: {duration:.2f} s', (10, y_offset), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 2)
                    y_offset += 20

            with record_function("show_frame"):
                # Display the frame
                cv2.imshow('Inference-YOLOv8n', frame)

            # Break the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release the VideoCapture object
    cap.release()
    cv2.destroyAllWindows()

    # Print profiling results
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

if __name__=='__main__':
    main()

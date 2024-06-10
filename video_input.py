import cv2
from ultralytics import YOLO
import time
import os


def main():
    current_directory = os.getcwd()
    # Load the model
    model = YOLO(os.path.join(current_directory, r"detect\16-512-25Epochs\weights\best.pt"))

    #creating videos_metadata dictionary
    videos_metadata = {
        'webcam': {
            'path': 0,
            'light_sufficient': True,
            'looking_lr': False,
            'detected_drowsiness': [],
            'ground_truth_drowsiness': [],
            'inference_time':[]
        },
        'debug_video_sample': {
            'path': os.path.join(current_directory, 'test_video\debugging_sample.avi'),
            'light_sufficient': True,
            'looking_lr': False,
            'detected_drowsiness': [],
            'ground_truth_drowsiness': [],
            'inference_time':[]
        }
    }
    # #example of adding metadata (used later for the four video data)
    # videos_metadata['video_name'] = {
    #     'path': os.path.join(current_directory, 'path/to/another/vid'),
    #     'light_sufficient': False,
    #     'looking_lr': True,
    #     'detected_drowsiness': [0.1, 0.3, 0.5],  # Example list of floats
    #     'ground_truth_drowsiness': [0.2, 0.4, 0.6],  # Example list of floats
    #     'inference_time':0
    # }

    #iterating every metadata element in every 'video_name' (videos_metadata members) element
    for video_name, metadata in videos_metadata.items():
        temp_inference_time = []
        video_path = metadata['path']
        # Start the webcam
        cap = cv2.VideoCapture(video_path)
        # cap = cv2.VideoCapture(os.path.join(current_directory, r"test_video\10-MaleGlasses-Trim.avi"))  
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_duration = 1 / fps
        frame_number = 0
        print("FPS: ", fps)

        # Initialize the dictionary to keep track of detection times and last durations
        detections = {
            'closed-eyes': {'duration': 0, 'frame_count': 0, 'last_seen_frame': None},
            'yawn': {'duration': 0, 'frame_count': 0, 'last_seen_frame': None}
        }

        # videos_metadata

        drowsy_state = False

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_number += 1 

            # Perform inference
            inference_start=time.time()
            results = model.predict(frame, conf=0.6)
            inference_end=time.time()
            temp_inference_time.append(inference_end-inference_start) 

            # Track which classes are currently detected
            current_detections = set()

            # Draw the bounding boxes by iterating over the results
            for result in results:
                for r in result.boxes.data.tolist():
                    x1, y1, x2, y2, conf, class_id = r
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    class_name = model.names[int(class_id)]
                    current_detections.add(class_name)

                    # Draw rectangle
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # Display class label and confidence
                    cv2.putText(frame, f'{class_name} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 2)

                    # Update detection frame counts
                    if detections[class_name]['last_seen_frame'] is None:
                        detections[class_name]['last_seen_frame'] = frame_number
                    else:
                        detections[class_name]['frame_count'] += frame_number - detections[class_name]['last_seen_frame']
                        detections[class_name]['last_seen_frame'] = frame_number
                    detections[class_name]['was_detected'] = True  # params to state the detected state 

            # Reset the durations when the class are not detected anymore
            for class_name in detections:
                if class_name not in current_detections:  # check whether this current frame still detecting the same class
                    detections[class_name]['was_detected'] = False  # reset the previously detected class state
                    detections[class_name]['frame_count'] = 0  # reset counted frame value
                    detections[class_name]['last_seen_frame'] = None  # reset previously seen frame

            # Convert frame counts to time using FPS
            for class_name in detections:
                # this default frames per second are 30FPS -> and the duration for each frame are 1/30 ~ 0.33
                # for instance, if the closed-eyes class are detected for 65 frames consecutively
                # that means the durations of detected closed-eyes are 65*0.03 which are 1.95s
                detections[class_name]['duration'] = detections[class_name]['frame_count'] * frame_duration

            # Detect drowsiness based on the duration of closed-eyes and yawn
            closed_eyes_duration = detections['closed-eyes']['duration']
            yawn_duration = detections['yawn']['duration']
            
            # Logic for detecting drowsiness
            if closed_eyes_duration > 0.5 or yawn_duration > 5.0:  # thresholds in seconds
                drowsy_state = True
            else:
                drowsy_state = False

            # debugging print state
            print(f"Closed-eyes duration: {closed_eyes_duration:.2f} seconds")
            print(f"Yawn duration: {yawn_duration:.2f} seconds")
            print(f"Drowsy state: {drowsy_state}")
            
            # Drowsy State Branch Logic
            if drowsy_state is True:
                cv2.rectangle(frame, (500, 20), (640, 60), (255, 255, 255), -1)
                cv2.putText(frame, 'Drowsy', (500, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)

            """
            Static Information Display Function
            """
            # drawing and writing the annotation
            cv2.rectangle(frame, (0, 10), (200, 60), (255, 255, 255), -1)
            y_offset = 30
            cv2.putText(frame, f'closed-eyes: {closed_eyes_duration:.2f} s', (10, y_offset), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 2)
            y_offset += 20
            cv2.putText(frame, f'yawn: {yawn_duration:.2f} s', (10, y_offset), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 2)
            y_offset += 20

            # Display the frame
            cv2.imshow('Inference-YOLOv8n', frame)
            
            # Break the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        #insert printing logic in here
        #For Inference Time (counting inference time average):
        metadata['inference_time']=sum(temp_inference_time)/len(temp_inference_time)
        print(metadata['inference_time']) #debugging prompt
        # Release the VideoCapture object
        cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

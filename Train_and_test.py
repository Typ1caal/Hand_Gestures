import cv2
import joblib
import mediapipe as mp
import numpy as np
from sklearn import svm

# Gesture labels and corresponding landmarks
gesture_labels = {
    0: 'Like',
    1: 'Dislike',
    2: 'Stop',
    3: 'Peace',
    4: 'Fist'
}

# Define gesture detection thresholds
gesture_thresholds = {
    'Like': 0.12,   # Threshold value for Like gesture
    'Dislike': 0.2,   # Threshold value for Dislike gesture
    'Stop': 0.4,   # Threshold value for Stop gesture
    'Peace': 0.6,   # Threshold value for Peace gesture
    'Fist': 0.8   # Threshold value for Fist gesture
}

# Initialize MediaPipe hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Capture video from webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Initialize variables
training_data = []
training_labels = []
data_count = 0
target_data_count = 1

# Collect training data
gesture_count = 0
current_gesture = gesture_labels[gesture_count]
collect_data = False

# Create OpenCV window with named trackbars
cv2.namedWindow("Hand Gestures")
cv2.createTrackbar("Collect Data", "Hand Gestures", 0, 1, lambda x: None)

gesture_label = 'Unknown'  # Declare gesture_label outside the if statement

while True:
    # Read frame from video capture
    ret, frame = cap.read()
    if not ret:
        print("Cannot receive frame (stream end?). Exiting...")
        break

    # Convert the frame to RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe
    results = hands.process(frame_rgb)

    # Recognize hand gestures
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        thumb_landmark = hand_landmarks.landmark[4]  # Thumb tip landmark
        index_finger_landmark = hand_landmarks.landmark[8]  # Index finger tip landmark
        middle_finger_landmark = hand_landmarks.landmark[12]  # Middle finger tip landmark
        ring_finger_landmark = hand_landmarks.landmark[16]  # Ring finger tip landmark
        pinky_finger_landmark = hand_landmarks.landmark[20]  # Pinky finger tip landmark

        # Calculate distances between fingertips
        thumb_index_distance = np.linalg.norm(
            np.array([thumb_landmark.x, thumb_landmark.y]) - np.array([index_finger_landmark.x, index_finger_landmark.y]))
        thumb_middle_distance = np.linalg.norm(
            np.array([thumb_landmark.x, thumb_landmark.y]) - np.array([middle_finger_landmark.x, middle_finger_landmark.y]))
        thumb_ring_distance = np.linalg.norm(
            np.array([thumb_landmark.x, thumb_landmark.y]) - np.array([ring_finger_landmark.x, ring_finger_landmark.y]))
        thumb_pinky_distance = np.linalg.norm(
            np.array([thumb_landmark.x, thumb_landmark.y]) - np.array([pinky_finger_landmark.x, pinky_finger_landmark.y]))

        # Determine gesture based on distances and thresholds
        if thumb_index_distance > gesture_thresholds['Fist']:
            if thumb_middle_distance < gesture_thresholds['Like']:
                gesture_label = 'Like'
            elif thumb_middle_distance > gesture_thresholds['Dislike']:
                gesture_label = 'Dislike'
            elif thumb_ring_distance > gesture_thresholds['Stop'] and thumb_pinky_distance > gesture_thresholds['Stop']:
                gesture_label = 'Stop'
            elif thumb_ring_distance < gesture_thresholds['Peace'] and thumb_pinky_distance < gesture_thresholds['Peace']:
                gesture_label = 'Peace'
            elif thumb_index_distance <= gesture_thresholds['Fist']:
                gesture_label = 'Fist'
        else:
            gesture_label = 'Unknown'

        # Add data to training set
        if collect_data:
            if gesture_label != 'Unknown':
                training_data.append([thumb_index_distance, thumb_middle_distance, thumb_ring_distance, thumb_pinky_distance])
                training_labels.append(current_gesture)
                data_count += 1
                print("Data collected for gesture:", current_gesture)
                print("Data count:", data_count)

                if data_count >= target_data_count:
                    gesture_count += 1
                    if gesture_count >= len(gesture_labels):
                        break

                    current_gesture = gesture_labels[gesture_count]
                    data_count = 0
                    print("Switched to gesture:", current_gesture)
                    print("Collecting data for gesture:", current_gesture)
                    print("Make the gesture and keep it steady...")

    # Display the frame with gesture label and landmarks
    cv2.putText(frame, "Gesture: " + gesture_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display data collection status indicator
    if collect_data:
        cv2.putText(frame, "Collecting Data: ON", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "Collecting Data: OFF", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Hand Gestures', frame)

    # User input to start/stop collecting data
    key = cv2.waitKey(1)
    if key == ord('c'):  # 'c' key to start/stop collecting data
        collect_data = not collect_data
        if collect_data:
            print("Collecting data for gesture:", current_gesture)
            print("Make the gesture and keep it steady...")
        else:
            print("Stopped collecting data")

    elif key == ord('q'):  # 'q' key to exit the program
        break

# Convert training data to NumPy arrays
training_data = np.array(training_data)
training_labels = np.array(training_labels)

# Train the classifier
svm_model = svm.SVC(kernel='linear', C=1.0)
svm_model.fit(training_data, training_labels)

# Save the trained model
joblib.dump(svm_model, 'gesture_classifier.pkl')

# Release the video capture and close the windows
cap.release()
cv2.destroyAllWindows()

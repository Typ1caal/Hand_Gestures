import cv2
import joblib
import mediapipe as mp
import numpy as np
from sklearn import svm

# Gesture labels and corresponding landmarks
gesture_labels = {
    0: 'Like',
    1: 'Ok',
    2: 'Stop',
    3: 'Peace',
    4: 'Fist'
}

# Define gesture detection thresholds
gesture_thresholds = {
    'Like': 0.12,   # Threshold value for Like gesture
    'Ok': 0.2,      # Threshold value for Ok gesture
    'Stop': 0.4,    # Threshold value for Stop gesture
    'Peace': 0.6,   # Threshold value for Peace gesture
    'Fist': 0.8     # Threshold value for Fist gesture
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
target_data_count = 200

# Collect training data
gesture_count = 0
current_gesture = gesture_labels[gesture_count]
collect_data = False

# Create OpenCV window with named trackbars
cv2.namedWindow("Hand Gestures")

# Initialize gesture_label outside the loop
gesture_label = 'Unknown'

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

            # Create feature vector
            feature_vector = [thumb_index_distance, thumb_middle_distance, thumb_ring_distance, thumb_pinky_distance]

            # Collect training data
            if collect_data:
                training_data.append(feature_vector)
                training_labels.append(gesture_count)
                data_count += 1
                print("Collected data for", current_gesture + ":", data_count)

                if data_count == target_data_count:
                    gesture_count += 1
                    if gesture_count < len(gesture_labels):
                        current_gesture = gesture_labels[gesture_count]
                        data_count = 0  # Reset data count for the next gesture
                        print("Collecting data for", current_gesture + "...")
                    else:
                        collect_data = False
                        print("Data collection stopped.")

    # Display frame
    cv2.imshow('Hand Gestures', frame)

    # Check if the 'c' key is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        collect_data = not collect_data
        if collect_data:
            print("Collecting data for", current_gesture + "...")
        else:
            print("Data collection stopped.")

    # Exit when the 'q' key is pressed
    if key == ord('q'):
        break

# Release video capture and destroy OpenCV windows
cap.release()
cv2.destroyAllWindows()

# Convert training data and labels to numpy arrays
training_data = np.array(training_data)
training_labels = np.array(training_labels)

# Train SVM classifier
svm_classifier = svm.SVC()
svm_classifier.fit(training_data, training_labels)

# Save SVM classifier to file
joblib.dump(svm_classifier, 'svm_classifier.joblib')

# Save training data and labels to file
np.save('training_data.npy', training_data)
np.save('training_labels.npy', training_labels)

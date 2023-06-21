import cv2
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

# Load training data
X_train = np.load('training_data.npy', allow_pickle=True)
y_train = np.load('training_labels.npy', allow_pickle=True)

# Create SVM classifier and train it
svm_model = svm.SVC()
svm_model.fit(X_train, y_train)

# Capture video from webcam
cap = cv2.VideoCapture(0)

# Main loop
while True:
    # Read frame from video capture
    ret, frame = cap.read()

    # Convert the frame to RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe
    results = hands.process(frame_rgb)

    # Recognize hand gestures
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
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

        # Predict gesture using SVM classifier
        X_test = np.array([[thumb_index_distance, thumb_middle_distance, thumb_ring_distance, thumb_pinky_distance]])
        gesture_id = svm_model.predict(X_test)
        gesture_label = gesture_labels[gesture_id[0]]

        # Draw gesture label on the frame
        cv2.putText(frame, gesture_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Hand Gestures', frame)

    # Check for key press and exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture
cap.release()
cv2.destroyAllWindows()

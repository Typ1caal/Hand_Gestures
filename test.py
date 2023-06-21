import cv2
import mediapipe as mp
import numpy as np
from sklearn.svm import SVC

# Gesture labels and corresponding landmarks
gesture_labels = {
    0: 'Like',
    1: 'Dislike',
    2: 'Stop',
    3: 'Peace',
    4: 'Fist'
}

# Define training data
training_data = {
    'Like': [],
    'Dislike': [],
    'Stop': [],
    'Peace': [],
    'Fist': []
}

# Initialize MediaPipe hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Capture video from webcam
cap = cv2.VideoCapture(0)

# Flag to indicate training mode
training_mode = True

# Main loop
while True:
    # Read frame from video capture
    ret, frame = cap.read()

    # Convert the frame to RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe
    results = hands.process(frame_rgb)

    # Draw hand landmarks on the frame
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for idx, landmark in enumerate(hand_landmarks.landmark):
                # Extract landmark coordinates
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])

                # Draw a circle at each landmark position
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

                # Add landmark coordinates to training data if in training mode
                if training_mode:
                    gesture_label = gesture_labels[len(training_data)]
                    training_data[gesture_label].append((x, y))

    # Enable/disable training mode
    if len(training_data) == len(gesture_labels):
        training_mode = False

    # Gesture recognition
    if not training_mode and results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = []
        for landmark in hand_landmarks.landmark:
            # Extract landmark coordinates
            x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
            landmarks.append((x, y))

        # Train SVM model
        svm_model = SVC()
        X_train = []
        y_train = []
        for gesture_label, data in training_data.items():
            X_train.extend(data)
            y_train.extend([gesture_label] * len(data))
        svm_model.fit(X_train, y_train)

        # Predict gesture
        predicted_label = svm_model.predict([landmarks])[0]

        # Display gesture label
        cv2.putText(frame, predicted_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow('Hand Tracking & Gesture Control', frame)

    # Check for quit event
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture
cap.release()
cv2.destroyAllWindows()

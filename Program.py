import cv2
import mediapipe as mp
import numpy as np
from sklearn import svm

# Oznake gesta i odgovarajući landmarkovi
oznake_gesta = {
    0: 'Like',  # Sviđa mi se
    1: 'Ok',  # U redu
    2: 'Stop',  # Stani
    3: 'Peace',  # Mir
    4: 'Fist'  # Šaka
}

# Definiranje pragova prepoznavanja gesta
pragovi_gesta = {
    'Like': 0.12,  # Prag za gestu "Sviđa mi se"
    'Ok': 0.2,  # Prag za gestu "U redu"
    'Stop': 0.4,  # Prag za gestu "Stani"
    'Peace': 0.6,  # Prag za gestu "Mir"
    'Fist': 0.8  # Prag za gestu "Šaka"
}

# Inicijalizacija MediaPipe biblioteke za praćenje ruku
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Učitavanje podataka za treniranje
X_train = np.load('training_data.npy', allow_pickle=True)
y_train = np.load('training_labels.npy', allow_pickle=True)

# Stvaranje SVM klasifikatora i treniranje
svm_model = svm.SVC()
svm_model.fit(X_train, y_train)


# Izračunavanje točnosti učenja za svaku gestu
geste = list(oznake_gesta.values())
točnosti_gesta = {}
for gesta in geste:
    X_gesta = X_train[y_train == geste.index(gesta)]
    y_gesta = y_train[y_train == geste.index(gesta)]
    accuracy = svm_model.score(X_gesta, y_gesta)
    točnosti_gesta[gesta] = accuracy

# Ispisivanje točnosti učenja za svaku gestu
for gesta, accuracy in točnosti_gesta.items():
    print(f"Točnost učenja za gestu '{gesta}': {accuracy}")

# Snimanje videa pomoću web kamere
cap = cv2.VideoCapture(0)

# Glavna petlja
while True:
    # Čitanje frejma iz snimke
    ret, frame = cap.read()

    # Konverzija frejma u RGB za MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Procesuiranje frejma pomoću MediaPipe
    results = hands.process(frame_rgb)

    # Prepoznavanje gesta ruke
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        thumb_landmark = hand_landmarks.landmark[4]  # Landmark vrha palca
        index_finger_landmark = hand_landmarks.landmark[8]  # Landmark vrha kažiprsta
        middle_finger_landmark = hand_landmarks.landmark[12]  # Landmark vrha srednjeg prsta
        ring_finger_landmark = hand_landmarks.landmark[16]  # Landmark vrha prstenjaka
        pinky_finger_landmark = hand_landmarks.landmark[20]  # Landmark vrha malog prsta

        # Računanje udaljenosti između vrhova prstiju
        thumb_index_distance = np.linalg.norm(
            np.array([thumb_landmark.x, thumb_landmark.y]) - np.array([index_finger_landmark.x, index_finger_landmark.y]))
        thumb_middle_distance = np.linalg.norm(
            np.array([thumb_landmark.x, thumb_landmark.y]) - np.array([middle_finger_landmark.x, middle_finger_landmark.y]))
        thumb_ring_distance = np.linalg.norm(
            np.array([thumb_landmark.x, thumb_landmark.y]) - np.array([ring_finger_landmark.x, ring_finger_landmark.y]))
        thumb_pinky_distance = np.linalg.norm(
            np.array([thumb_landmark.x, thumb_landmark.y]) - np.array([pinky_finger_landmark.x, pinky_finger_landmark.y]))

        # Predviđanje gesta pomoću SVM klasifikatora
        X_test = np.array([[thumb_index_distance, thumb_middle_distance, thumb_ring_distance, thumb_pinky_distance]])
        gesture_id = svm_model.predict(X_test)
        gesture_label = oznake_gesta[gesture_id[0]]

        # Ispisivanje oznake gesta na frejmu
        cv2.putText(frame, gesture_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Crtanje točaka na vrhovima prstiju
        radius = 5
        color = (0, 255, 0)  # Zelena boja
        thickness = -1  # Ispuni točku
        cv2.circle(frame, (int(thumb_landmark.x * frame.shape[1]), int(thumb_landmark.y * frame.shape[0])), radius, color, thickness)
        cv2.circle(frame, (int(index_finger_landmark.x * frame.shape[1]), int(index_finger_landmark.y * frame.shape[0])), radius, color, thickness)
        cv2.circle(frame, (int(middle_finger_landmark.x * frame.shape[1]), int(middle_finger_landmark.y * frame.shape[0])), radius, color, thickness)
        cv2.circle(frame, (int(ring_finger_landmark.x * frame.shape[1]), int(ring_finger_landmark.y * frame.shape[0])), radius, color, thickness)
        cv2.circle(frame, (int(pinky_finger_landmark.x * frame.shape[1]), int(pinky_finger_landmark.y * frame.shape[0])), radius, color, thickness)

    # Prikazivanje frejma
    cv2.imshow('Prepoznavanje gesta ruke', frame)

    # Provjera pritisnutih tipki i izlazak ako je pritisnut 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Oslobađanje snimanja videa
cap.release()
cv2.destroyAllWindows()

import cv2
import joblib
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

# Snimanje videa pomoću web kamere
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Nije moguće otvoriti kameru")
    exit()

# Inicijalizacija varijabli
training_data = []
training_labels = []
data_count = 0
target_data_count = 200

# Prikupljanje podataka za treniranje
gesture_count = 0
current_gesture = oznake_gesta[gesture_count]
collect_data = False

# Kreiranje OpenCV prozora s imenovanim trakbarima
cv2.namedWindow("Prepoznavanje gesta ruke")

# Inicijalizacija "gesture_label" izvan petlje
gesture_label = 'Nepoznato'

while True:
    # Čitanje frejma iz snimke
    ret, frame = cap.read()
    if not ret:
        print("Nije moguće primiti frejm (kraj snimanja?). Izlazim...")
        break

    # Konverzija frejma u RGB za MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Procesuiranje frejma pomoću MediaPipe
    results = hands.process(frame_rgb)

    # Prepoznavanje gesta ruke
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Crtanje landmarkova ruke
            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

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

            # Kreiranje vektora značajki
            feature_vector = [thumb_index_distance, thumb_middle_distance, thumb_ring_distance, thumb_pinky_distance]

            # Prikupljanje podataka za treniranje
            if collect_data:
                training_data.append(feature_vector)
                training_labels.append(gesture_count)
                data_count += 1
                print("Prikupljeni podaci za", current_gesture + ":", data_count)

                if data_count == target_data_count:
                    gesture_count += 1
                    if gesture_count < len(oznake_gesta):
                        current_gesture = oznake_gesta[gesture_count]
                        data_count = 0  # Resetiranje brojača podataka za sljedeći gestu
                        print("Prikupljanje podataka za", current_gesture + "...")
                    else:
                        collect_data = False
                        print("Prikupljanje podataka zaustavljeno.")

    # Prikazivanje frejma
    cv2.imshow('Prepoznavanje gesta ruke', frame)

    # Provjera pritisnutih tipki 'c'
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        collect_data = not collect_data
        if collect_data:
            print("Prikupljanje podataka za", current_gesture + "...")
        else:
            print("Prikupljanje podataka zaustavljeno.")

    # Izlaz iz programa ako je pritisnuta tipka 'q'
    if key == ord('q'):
        break

# Oslobađanje snimanja videa i zatvaranje OpenCV prozora
cap.release()
cv2.destroyAllWindows()

# Pretvaranje podataka i oznaka za treniranje u numpy nizove
training_data = np.array(training_data)
training_labels = np.array(training_labels)

# Treniranje SVM klasifikatora
svm_classifier = svm.SVC()
svm_classifier.fit(training_data, training_labels)

# Spremanje SVM klasifikatora u datoteku
joblib.dump(svm_classifier, 'svm_classifier.joblib')

# Spremanje podataka i oznaka za treniranje u datoteke
np.save('training_data.npy', training_data)
np.save('training_labels.npy', training_labels)

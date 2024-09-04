import cv2
import numpy as np
import mediapipe as mp

# MediaPipe untuk deteksi tangan
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Muat model MobileNet SSD yang telah dilatih untuk deteksi objek
# Ganti jalur relatif dengan jalur absolut ke file .pb dan .pbtxt
net = cv2.dnn_DetectionModel(
    'D:/Projects/face_rec_env/src/ssd_mobilenet_v3_large_coco_2020_01_14/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt',
    'D:/Projects/face_rec_env/src/ssd_mobilenet_v3_large_coco_2020_01_14/frozen_inference_graph.pb'
)

net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Daftar objek yang dapat dikenali oleh MobileNet SSD (COCO dataset)
classNames = []
# Gunakan jalur absolut untuk file coco.names
with open('D:/Projects/face_rec_env/src/coco.names', 'r') as f:
    classNames = f.read().splitlines()

# Akses webcam
cap = cv2.VideoCapture(0)

# Setup MediaPipe untuk deteksi tangan
with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Gagal menangkap gambar dari webcam.")
            break

        try:
            # Deteksi tangan
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Deteksi objek dengan MobileNet SSD
            # Tingkatkan ambang batas keyakinan menjadi 0.6 atau lebih
            class_ids, confs, bbox = net.detect(frame, confThreshold=0.6)

            if len(class_ids) != 0:
                for class_id, confidence, box in zip(class_ids.flatten(), confs.flatten(), bbox):
                    if box is not None and 0 <= class_id - 1 < len(classNames):
                        cv2.rectangle(frame, box, color=(0, 255, 0), thickness=2)
                        cv2.putText(frame, f'{classNames[class_id - 1].upper()} {int(confidence * 100)}%',
                                    (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        except Exception as e:
            print(f"Kesalahan saat mendeteksi: {str(e)}")

        # Tampilkan hasil deteksi
        cv2.imshow('Deteksi Objek dan Tangan', frame)

        # Cek jika 'q' ditekan untuk keluar
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Lepaskan kamera dan tutup semua jendela
cap.release()
cv2.destroyAllWindows()

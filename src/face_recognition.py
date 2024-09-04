import cv2
from deepface import DeepFace

# Akses webcam
cap = cv2.VideoCapture(0)

# Kamus untuk terjemahan ekspresi
emotion_translation = {
    'angry': 'Marah',
    'disgust': 'Jijik',
    'fear': 'Takut',
    'happy': 'Bahagia',
    'sad': 'Sedih',
    'surprise': 'Terkejut',
    'neutral': 'Netral'
}

while True:
    # Tangkap frame dari webcam
    ret, frame = cap.read()
    
    if not ret:
        print("Gagal menangkap gambar dari webcam.")
        break

    try:
        # Analisis frame untuk wajah dan emosi dengan model default DeepFace
        analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        
        # Dapatkan bounding box wajah
        face_rectangles = analysis['region'] if isinstance(analysis, dict) else analysis[0]['region']
        
        # Gambar kotak di sekitar wajah
        x, y, w, h = face_rectangles['x'], face_rectangles['y'], face_rectangles['w'], face_rectangles['h']
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Ambil emosi dominan dari hasil
        dominant_emotion = analysis['dominant_emotion'] if isinstance(analysis, dict) else analysis[0]['dominant_emotion']

        # Terjemahkan ekspresi ke bahasa Indonesia
        dominant_emotion = emotion_translation.get(dominant_emotion, 'Tidak Diketahui')

        # Tampilkan hasil analisis pada frame
        cv2.putText(frame, f'Emosi: {dominant_emotion}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    except Exception as e:
        print(f"Error saat menganalisis: {str(e)}")
        cv2.putText(frame, "Error", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    
    cv2.imshow('Video', frame)

    # Keluar dari loop dengan menekan 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Ketika selesai, lepaskan kamera dan tutup jendela
cap.release()
cv2.destroyAllWindows()
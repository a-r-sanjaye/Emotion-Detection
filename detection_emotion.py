from fer import FER
import cv2

# Create emotion detector
emotion_detector = FER(mtcnn=True)

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB to match model input
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect emotions
    emotions = emotion_detector.detect_emotions(rgb_frame)

    # Display results
    for face in emotions:
        (x, y, w, h) = face["box"]
        dominant_emotion, score = emotion_detector.top_emotion(rgb_frame)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"{dominant_emotion} ({score:.2f})", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()





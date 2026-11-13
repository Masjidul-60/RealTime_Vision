
import cv2
import time

# Webcam open kora (0 = default camera)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Camera open kora jaachhe na")

# OpenCV er built-in face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

mode = "normal"
prev_time = time.time()

print("Keys: n = normal | g = grayscale | e = edges | q = quit")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)  # mirror effect
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Face detect
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    # Display mode onujayi frame toiri
    if mode == "grayscale":
        display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    elif mode == "edges":
        display = cv2.cvtColor(cv2.Canny(gray, 100, 200), cv2.COLOR_GRAY2BGR)
    else:
        display = frame.copy()

    # Face er chardike box
    for (x, y, w, h) in faces:
        cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(display, "Face", (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # FPS
    now = time.time()
    fps = 1 / (now - prev_time)
    prev_time = now
    cv2.putText(display, f"FPS: {fps:.0f} | Faces: {len(faces)} | Mode: {mode}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Realtime Vision", display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("g"):
        mode = "grayscale"
    elif key == ord("e"):
        mode = "edges"
    elif key == ord("n"):
        mode = "normal"

cap.release()
cv2.destroyAllWindows()
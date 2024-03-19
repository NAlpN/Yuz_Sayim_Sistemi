import cv2
import time

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Kamera açılamadı!")
        return

    detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    count = 0
    start_time = time.time()
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Kare alınamadı!")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        count += len(faces)

        elapsed_time = time.time() - start_time
        if elapsed_time >= 10:
            with open("nesne_sayisi.txt", "a") as file:
                file.write(f"{int(elapsed_time)} saniyede {count} nesne tespit edildi.\n")

            count = 0
            start_time = time.time()

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

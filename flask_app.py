import cv2
from flask import Flask, render_template, Response

app = Flask(__name__)

# Load the face classifiers
# face_classifier_haar = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_classifier_lbp = cv2.CascadeClassifier('lbpcascade_frontalface.xml')

# Check if the classifiers are loaded correctly
# if face_classifier_haar.empty():
#     print("Error: Could not load Haar cascade classifier")
#     exit()
if face_classifier_lbp.empty():
    print("Error: Could not load LBP cascade classifier")
    exit()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # faces_haar = face_classifier_haar.detectMultiScale(gray, scaleFactor=1.1,  minNeighbors=5, minSize=(50, 50))
        faces_lbp = face_classifier_lbp.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

        # Debugging: Print number of faces detected
        # print(f"Haar detected faces: {len(faces_haar)}")
        print(f"LBP detected faces: {len(faces_lbp)}")

        # Convert faces to tuples to allow set intersection
        # faces_haar = [tuple(face) for face in faces_haar]
        faces_lbp = [tuple(face) for face in faces_lbp]

        # Find intersection of detected faces
        # faces = list(set(faces_haar) & set(faces_lbp))

        # Debugging: Print number of faces after intersection
        # print(f"Faces after intersection: {len(faces)}")

        for (x, y, w, h) in faces_lbp:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)

        print("buffer is: ", buffer)

        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('page1.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)

import atexit
atexit.register(lambda: cap.release())
cv2.destroyAllWindows()

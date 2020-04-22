import face_recognition
import cv2
import os
import numpy as np
import time
import datetime
from config import ROOT_DIR


img = cv2.imread(os.path.join(ROOT_DIR, "known-faces/alisher.jpg"))
known_faces = face_recognition.load_image_file(os.path.join(ROOT_DIR, "known-faces/alisher.jpg"))
known_face_encoding = face_recognition.face_encodings(known_faces)[0]

# merve_image = face_recognition.load_image_file("/Users/alisher/Desktop/known_face/Merve.jpg")
# merve_face_encoding = face_recognition.face_encodings(merve_image)[0]

known_face_encodings = [known_face_encoding]
known_face_names = ["Alisher"]

unknown_face_dir = os.path.join(ROOT_DIR, "unknown-faces/")
if not os.path.exists(unknown_face_dir):
    os.makedirs(unknown_face_dir)

cv2.namedWindow("preview")
video_capture = cv2.VideoCapture(0)
unknown_count = 0
while True:
    ret, frame = video_capture.read()
    if ret:
        rgb_frame = frame[:, :, ::-1]

        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        if len(face_encodings) == 0:
            pass
        else:
            names = ["Unknown"] * len(face_encodings)
            for i, ((top, right, bottom, left), face_encoding) in enumerate(zip(face_locations, face_encodings)):
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)

                if matches[best_match_index]:
                    names[i] = known_face_names[best_match_index]

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            if all(name == 'Unknown' for name in names) and unknown_count > 10:
                unknown_count += 1
                for (top, right, bottom, left) in face_locations:
                    ts = time.time()
                    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
                    unknown_face = frame[top:bottom, left:right]
                    cv2.imwrite(unknown_face_dir + st + '.jpg', unknown_face)
                os.system("pmset sleepnow")
                # exit()

            for (top, right, bottom, left), name in zip(face_locations, names):
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()

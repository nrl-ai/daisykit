import cv2
import numpy as np
import daisykit


config_file = "./assets/configs/face_detector_config.json"
with open(config_file, "r") as f:
    config = f.read()
face_detector = daisykit.FaceDetectorFlow(config)

# Open video stream from webcam
vid = cv2.VideoCapture(0)

while(True):

    # Capture the video frame
    ret, frame = vid.read()

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_detector.Process(frame)
    for face in faces:
        print([face.x, face.y, face.w, face.h,
              face.confidence, face.wearing_mask_prob])
    face_detector.DrawResult(frame, faces)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # The 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()

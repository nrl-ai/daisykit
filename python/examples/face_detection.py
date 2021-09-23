import cv2
import json
from daisykit.utils import get_asset_file
import daisykit

config = {
    "face_detection_model": {
        "model": get_asset_file("models/face_detection/yolo_fastest_with_mask/yolo-fastest-opt.param"),
        "weights": get_asset_file("models/face_detection/yolo_fastest_with_mask/yolo-fastest-opt.bin"),
        "input_width": 320,
        "input_height": 320,
        "score_threshold": 0.7,
        "iou_threshold": 0.5
    },
    "with_landmark": True,
    "facial_landmark_model": {
        "model": get_asset_file("models/facial_landmark/pfld-sim.param"),
        "weights": get_asset_file("models/facial_landmark/pfld-sim.bin")
    }
}

face_detector_flow = daisykit.FaceDetectorFlow(json.dumps(config))

# Open video stream from webcam
vid = cv2.VideoCapture(0)

while(True):

    # Capture the video frame
    ret, frame = vid.read()

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    faces = face_detector_flow.Process(frame)
    # for face in faces:
    #     print([face.x, face.y, face.w, face.h,
    #           face.confidence, face.wearing_mask_prob])
    face_detector_flow.DrawResult(frame, faces)

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

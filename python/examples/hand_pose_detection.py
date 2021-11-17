import cv2
import json
from daisykit.utils import get_asset_file, to_py_type
from daisykit import HandPoseDetectorFlow

config = {
    "hand_detection_model": {
        "model": get_asset_file("models/hand_pose/yolox_hand_swish.param"),
        "weights": get_asset_file("models/hand_pose/yolox_hand_swish.bin"),
        "input_width": 256,
        "input_height": 256,
        "score_threshold": 0.45,
        "iou_threshold": 0.65,
        "use_gpu": False
    },
    "hand_pose_model": {
        "model": get_asset_file("models/hand_pose/hand_lite-op.param"),
        "weights": get_asset_file("models/hand_pose/hand_lite-op.bin"),
        "input_size": 224,
        "use_gpu": False
    }
}

flow = HandPoseDetectorFlow(json.dumps(config))

# Open video stream from webcam
vid = cv2.VideoCapture(0)

while(True):

    # Capture the video frame
    ret, frame = vid.read()

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    poses = flow.Process(frame)
    flow.DrawResult(frame, poses)

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Convert poses to Python list of dict
    poses = to_py_type(poses)

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

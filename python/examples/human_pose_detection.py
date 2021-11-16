import cv2
import json
from daisykit.utils import get_asset_file
from daisykit import HumanPoseMoveNetFlow

config = {
  "person_detection_model": {
    "model": get_asset_file("models/human_detection/ssd_mobilenetv2.param"),
    "weights": get_asset_file("models/human_detection/ssd_mobilenetv2.bin")
  },
  "human_pose_model": {
    "model": get_asset_file("models/human_pose_detection/movenet/lightning.param"),
    "weights": get_asset_file("models/human_pose_detection/movenet/lightning.bin"),
    "input_width": 192,
    "input_height": 192
  }
}

human_pose_flow = HumanPoseMoveNetFlow(json.dumps(config))

# Open video stream from webcam
vid = cv2.VideoCapture(0)

while(True):

    # Capture the video frame
    ret, frame = vid.read()

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = human_pose_flow.Process(frame)
    human_pose_flow.DrawResult(frame)

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

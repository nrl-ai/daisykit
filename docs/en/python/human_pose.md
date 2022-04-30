# Python: Human Pose

The human pose detector module contains an SSD-MobileNetV2 body detector
and a ported Google MoveNet model for human keypoints. This module can
be applied in fitness applications and AR games.

![](/images/python/image3.png)

![](/images/python/image12.gif)

**Source code:**

```py
import cv2
import json
from daisykit.utils import get_asset_file, to_py_type
from daisykit import HumanPoseMoveNetFlow
 
config = {
    "person_detection_model": {
        "model": get_asset_file("models/human_detection/ssd_mobilenetv2.param"),
        "weights": get_asset_file("models/human_detection/ssd_mobilenetv2.bin"),
        "input_width": 320,
        "input_height": 320,
        "use_gpu": False
    },
    "human_pose_model": {
        "model": get_asset_file("models/human_pose_detection/movenet/lightning.param"),
        "weights": get_asset_file("models/human_pose_detection/movenet/lightning.bin"),
        "input_width": 192,
        "input_height": 192,
        "use_gpu": False
    }
}
 
human_pose_flow = HumanPoseMoveNetFlow(json.dumps(config))
 
# Open video stream from webcam
vid = cv2.VideoCapture(0)
 
while(True):
 
    # Capture the video frame
    ret, frame = vid.read()
 
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
 
    poses = human_pose_flow.Process(frame)
    human_pose_flow.DrawResult(frame, poses)
 
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
 
    # Convert poses to Python list of dict
    poses = to_py_type(poses)
 
    # Display the result frame
    cv2.imshow('frame', frame)
 
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

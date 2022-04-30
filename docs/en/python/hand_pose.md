# Python: Hand Pose Detection

The hand pose detection flow comprises two models: a hand detection
model based on YOLOX and a 3D hand pose detection model released by
Google this November. Thanks to
[*FeiGeChuanShu*](https://github.com/FeiGeChuanShu) for the effort in
early model conversion.

![](/images/python/image1.png)

This hand pose flow can be used in AR games, hand gesture control, and
many cool DIY projects.

![](/images/python/image11.gif)

**Source code:**

```py
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
 
    # Display the result frame
    cv2.imshow('frame', frame)
 
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

In the above source code, `input_width` and `input_height` of the
`hand_detection_model` can be adjusted for speed/accuracy trade-off.

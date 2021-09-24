# DaisyKit Python

<https://pypi.org/project/daisykit/>

Python bindings for DaisyKit. This package only supports Ubuntu Linux - Python 3 now. We will add support for other platforms and models in the future.

## Install and run example

Install dependencies. Below commands are for Ubuntu. You can try other installation methods based on your OS.

```
sudo apt install pybind11-dev # Pybind11 - For Python/C++ Wrapper
sudo apt install libopencv-dev # For OpenCV
sudo apt install libvulkan-dev # Optional - For GPU support
```

Install DaisyKit

```
pip3 install daisykit
```

**Face Detection with mask recognition:**

```py
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
```

**Background Matting:**

```py
import cv2
import json
from daisykit.utils import get_asset_file
from daisykit import BackgroundMattingFlow

config = {
    "background_matting_model": {
        "model": get_asset_file("models/human_matting/erd/erdnet.param"),
        "weights": get_asset_file("models/human_matting/erd/erdnet.bin")
    }
}

# Load background
default_bg_file = get_asset_file("images/background.jpg")
background = cv2.imread(default_bg_file)
background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)

background_matting_flow = BackgroundMattingFlow(json.dumps(config), background)

# Open video stream from webcam
vid = cv2.VideoCapture(0)

while(True):

    # Capture the video frame
    ret, frame = vid.read()

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mask = background_matting_flow.Process(frame)
    background_matting_flow.DrawResult(frame, mask)

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
```

## Build Python package

Build environment: Ubuntu.

```
sudo apt install ninja-build
python3 -m pip install --user --upgrade twine
```

Build package:

```
python3 setup.py sdist
```

Upload to Pypi (for DaisyKit authors only)

```
twine upload dist/*
```

## TODO

- Multiplatform build.

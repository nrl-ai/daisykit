# DaisyKit Python

<https://pypi.org/project/daisykit/>

Deploy AI Systems Yourself (DAISY) Kit. DaisyKit Python is the wrapper of DaisyKit SDK, an AI framework focusing on the ease of deployment. At present, this package only has prebuilt distribution for Windows - Python 3. For other platform, you need to compile from source.

## How to install ?

**For Windows:**

```
pip3 install daisykit
```

**For Ubuntu:**

- Install dependencies

```
sudo apt install pybind11-dev # Pybind11 - For Python/C++ Wrapper
sudo apt install libopencv-dev # For OpenCV
sudo apt install libvulkan-dev # Optional - For GPU support
```

- Install DaisyKit (compile from source)

```
pip3 install --upgrade pip # Ensure pip is updated
pip3 install daisykit
```

**For other platforms:**

- Install OpenCV, Pybind11 and Vulkan development package (if you want GPU support)

- Install DaisyKit (compile from source)

```
pip3 install --upgrade pip # Ensure pip is updated
pip3 install daisykit
```

## Examples

**Object detection:**

```py
import cv2
import json
from daisykit.utils import get_asset_file, to_py_type
from daisykit import ObjectDetectorFlow

config = {
    "object_detection_model": {
        "model": get_asset_file("models/object_detection/yolox-tiny.param"),
        "weights": get_asset_file("models/object_detection/yolox-tiny.bin"),
        "input_width": 416,
        "input_height": 416,
        "score_threshold": 0.5,
        "iou_threshold": 0.65,
        "use_gpu": False,
        "class_names": [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
            "hair drier", "toothbrush"
        ]
    }
}


flow = ObjectDetectorFlow(json.dumps(config))

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
```

**Face Detection with mask recognition:**

```py
import cv2
import json
from daisykit.utils import get_asset_file, to_py_type
import daisykit

config = {
    "face_detection_model": {
        "model": get_asset_file("models/face_detection/yolo_fastest_with_mask/yolo-fastest-opt.param"),
        "weights": get_asset_file("models/face_detection/yolo_fastest_with_mask/yolo-fastest-opt.bin"),
        "input_width": 320,
        "input_height": 320,
        "score_threshold": 0.7,
        "iou_threshold": 0.5,
        "use_gpu": False
    },
    "with_landmark": True,
    "facial_landmark_model": {
        "model": get_asset_file("models/facial_landmark/pfld-sim.param"),
        "weights": get_asset_file("models/facial_landmark/pfld-sim.bin"),
        "input_width": 112,
        "input_height": 112,
        "use_gpu": False
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

    # Convert faces to Python list of dict
    faces = to_py_type(faces)

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
        "model": get_asset_file("models/background_matting/erd/erdnet.param"),
        "weights": get_asset_file("models/background_matting/erd/erdnet.bin"),
        "input_width": 256,
        "input_height": 256,
        "use_gpu": False
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

**Human Pose Detection:**

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

**Barcode Detection:**

```py
import cv2
import json
from daisykit.utils import get_asset_file
from daisykit import BarcodeScannerFlow

config = {
    "try_harder": True,
    "try_rotate": True
}

barcode_scanner_flow = BarcodeScannerFlow(json.dumps(config))

# Open video stream from webcam
vid = cv2.VideoCapture(0)

while(True):

    # Capture the video frame
    ret, frame = vid.read()

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = barcode_scanner_flow.Process(frame, draw=True)

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

**Hand Pose Detection:**

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

## Note for Python build

Current CD (continuous delivery) flow is partial, which means we only have prebuilt linux wheels for x86_64 and for Windows.

- Prebuilt wheels for linux x86_64 are built with Github actions.
- Windows wheels (64bit) are built manually on a local machine.
- macOS prebuilt wheels are not available for now. However, you can install dependencies (OpenCV, Vulkan) manually, then install Daisykit with pip command.

We will be happy if you can make a pull request to make the CD build fully automated. A good choice is using Github flow for all building tasks.

**Current steps for Windows build:**

```sh
bash ./build_tools/py_windows/build_dk_all_pythons_windows.sh
bash ./build_tools/py_windows/build_dk_python_source_dist.sh
bash ./build_tools/upload_pypi.sh
```

## Bug report

Please open an issue on our official repository if you find any error.

<https://github.com/Daisykit-AI/daisykit>

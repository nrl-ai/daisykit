# Python: Face Detection


Face detector flow in Daisykit contains a face detection model based on
[*YOLO Fastest*](https://github.com/dog-qiuqiu/Yolo-Fastest) and a
facial landmark detection model based on
[*PFLD*](https://github.com/polarisZhao/PFLD-pytorch). In addition, to
encourage makers to join hands in the fighting with COVID-19, we
selected a face detection model that can recognize people wearing face
masks or not.

![](/images/python/image5.png)

![](/images/python/image14.gif)

Let’s see into the source code below to understand how to run this model
with your webcam. First, we initialize the flow with a `config`
dictionary. It contains information about the models used in the flow.
`get_asset_file()` function will automatically download the models
and weights files from
[*https://github.com/DaisyLabSolutions/daisykit-assets*](https://github.com/DaisyLabSolutions/daisykit-assets),
so you don’t have to care about downloading them manually. The
downloading only happens the first time we use the models. After that,
you can run this code offline. You also can download all files yourself
and put the paths to file in instead of `get_asset_file(<assets address>)`. If you don’t need facial landmark output, set
`with_landmark` to `False`.

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

All flows in Daisykit are initialized by a configuration string.
Therefore we use **json.dumps()** to convert the config dictionary to a
string. For example, in face detector flow:

```py
face_detector_flow = daisykit.FaceDetectorFlow(json.dumps(config))
```

Run the flow to get detected faces by:

```py
faces = face_detector_flow.Process(frame)
```

You can use the **DrawResult()** method to visualize the result or write
a drawing function yourself. This AI flow can be used in DIY projects
such as [*smart COVID-19
camera*](https://github.com/vietanhdev/smart-face-mask-cam) or
Snap-chat-like camera decorators.

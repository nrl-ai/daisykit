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
[*https://github.com/Daisykit-AI/daisykit-assets*](https://github.com/Daisykit-AI/daisykit-assets),
so you don’t have to care about downloading them manually. The
downloading only happens the first time we use the models. After that,
you can run this code offline. You also can download all files yourself
and put the paths to file in instead of `get_asset_file(<assets address>)`. If you don’t need facial landmark output, set
`with_landmark` to `False`.

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
 
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
 
    mask = background_matting_flow.Process(image)
    background_matting_flow.DrawResult(image, mask)
 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
 
    # Display the result frame
    cv2.imshow('frame', frame)
    cv2.imshow('result', image)
 
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
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

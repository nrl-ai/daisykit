# Python: Background Matting

Background matting use only one segmentation model to generate a human
body mask. This mask can figure out which pixels belong to humans and
which belong to the background. This output can be used for background
replacement (like in the Google Meet app). The segmentation model was
taken from [*this
implementation*](https://github.com/nihui/ncnn-webassembly-portrait-segmentation)
by [*nihui*](https://github.com/nihui), the author of the NCNN
framework. The author also has a webpage for a live demo on web
browsers.

[*https://github.com/nihui/ncnn-webassembly-portrait-segmentation*](https://github.com/nihui/ncnn-webassembly-portrait-segmentation).

![](/images/python/image8.gif)

**Source code:**

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

In the source code, we use `get_asset_file("images/background.jpg")`
to download the default background. You can use another image for the
background by replacing it with a path to an image file.

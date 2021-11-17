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

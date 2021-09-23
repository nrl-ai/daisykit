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

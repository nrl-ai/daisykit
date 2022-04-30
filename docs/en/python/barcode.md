# Python: Barcode Detection

Barcodes can be used in a wide range of robotics and software
applications. That’s why we integrated a barcode reader into Daisykit.
The core algorithms of the barcode reader are from [*the Zxing-CPP
project*](https://github.com/nu-book/zxing-cpp). This barcode processor
can read QR codes and bar codes in different formats.

![](/images/python/image10.gif)

**Source code:**

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
 
    # Display the result frame
    cv2.imshow('frame', frame)
 
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

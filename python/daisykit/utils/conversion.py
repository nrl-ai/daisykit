import daisykit


def to_py_dict(obj):
    """Convert prediction object to Python dictionary

    Args:
        obj: Prediction object - Often C++ class or struct.
    Returns:
        result: Python dict
    """

    if isinstance(obj, daisykit.daisykit.Face):
        return {"x": obj.x, "y": obj.y, "w": obj.w, "h": obj.h, "confidence": obj.confidence, "wearing_mask_prob": obj.wearing_mask_prob, "landmark": daisykit.utils.to_py_type(obj.landmark)}
    elif isinstance(obj, daisykit.daisykit.Keypoint):
        return {"x": obj.x, "y": obj.y, "confidence": obj.confidence}
    elif isinstance(obj, daisykit.daisykit.KeypointXYZ):
        return {"x": obj.x, "y": obj.y, "z": obj.z, "confidence": obj.confidence}
    elif isinstance(obj, daisykit.daisykit.Box):
        return {"x": obj.x, "y": obj.y, "w": obj.w, "h": obj.h}
    elif isinstance(obj, daisykit.daisykit.Object):
        return {"x": obj.x, "y": obj.y, "w": obj.w, "h": obj.h, "class_id": obj.class_id, "confidence": obj.confidence}
    elif isinstance(obj, daisykit.daisykit.ObjectWithKeypoints):
        return {"x": obj.x, "y": obj.y, "w": obj.w, "h": obj.h, "class_id": obj.class_id, "confidence": obj.confidence, "keypoints": daisykit.utils.to_py_type(obj.keypoints)}
    elif isinstance(obj, daisykit.daisykit.ObjectWithKeypointsXYZ):
        return {"x": obj.x, "y": obj.y, "w": obj.w, "h": obj.h, "class_id": obj.class_id, "confidence": obj.confidence, "keypoints": daisykit.utils.to_py_type(obj.keypoints)}
    else:
        raise "Type not supported for conversion: {}".format(type(obj))


def to_py_type(obj):
    """Convert prediction result to Python list or dictionary

    Args:
        obj: Results from models. Object or vector.
    """

    if isinstance(obj, list):
        result = []
        for item in obj:
            result.append(to_py_dict(item))
        return result

    return to_py_dict(obj)

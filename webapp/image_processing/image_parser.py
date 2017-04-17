import cv2
import numpy as np

def get_bounding_boxes(filename, path="../tmp"):
    """
    Args:
        filename: Filename of image to generate bounding boxes for.

    Returns:
        A list of tuples in format (x,y,w,h)

    Raises:

    """
    MAX_HEIGHT_WIDTH = 300
    MIN_HEIGHT_WIDTH = 40

    # get image
    im = cv2.imread(path + '/' + filename)


    if type(im) != type(np.ndarray([])):
        raise FileNotFoundError("No file with that path or name found!")



    # preprocess image  and generate contours for bounding box detection
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray,150,255,cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    dilated = cv2.dilate(thresh[1],kernel,iterations = 13)
    im_temp,contours, hier = cv2.findContours(dilated,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


    # generate bounding boxes
    bounding_boxes = []
    for contour in contours:
        [x,y,w,h] = cv2.boundingRect(contour)

        # Discard contours greater than max and smaller than min
        if h > MAX_HEIGHT_WIDTH and w > 300:
            continue
        if h < MIN_HEIGHT_WIDTH or w < MIN_HEIGHT_WIDTH:
            continue

        bounding_boxes.append((x,y,w,h))

    return bounding_boxes

import cv2
import numpy as np
import pytesseract
from PIL import Image
from helper import preprocess_and_predict, get_first_horizontal_line
from mathproblem import MathProblem

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
    gray_arr = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray_arr,(5,5),0)

    ret3,thresh_arr = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    dilated = cv2.dilate(thresh_arr,kernel,iterations = 12)
    contours, hier = cv2.findContours(dilated,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #_,contours, hier = cv2.findContours(dilated,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # generate bounding boxes
    bounding_boxes = []
    for contour in contours:
        [x,y,w,h] = cv2.boundingRect(contour)

        # Discard contours greater than max and smaller than min
        if h > MAX_HEIGHT_WIDTH and w > MAX_HEIGHT_WIDTH:
            continue
        if h < MIN_HEIGHT_WIDTH or w < MIN_HEIGHT_WIDTH:
            continue

        bounding_boxes.append((x,y,w,h))

    average_height = sum([x[3] for x in bounding_boxes ])//len(bounding_boxes)
    bounding_boxes = map(lambda x: x if x[3] >= average_height else (x[0],x[1],x[2],average_height),bounding_boxes)

    bounding_boxes.sort(key=lambda x: (x[1],-x[0]))


    return bounding_boxes


def generate_problems(bounding_boxes,filename, path="../tmp"):
    # BIAS added to ensure line is not included in image
    BIAS = 2

    # get image
    im = cv2.imread(path + '/' + filename)

    if type(im) != type(np.ndarray([])):
        raise FileNotFoundError("No file with that path or name found!")

    count = 0
    problems = []
    for box in bounding_boxes:
        # Crop image to include problem and answer
        (x,y,w,h) = box
        problem_img = im[y:y+h, x:x+w]

        # Apply canny edge detection to image and use the processed image
        # in conjuction with HoughLine detection to split problem from answer
        edges = cv2.Canny(problem_img, 100, 200)
        lines = cv2.HoughLinesP(edges,1,np.pi/180,40,50,5,5)

        #(x_s, y_s, x_e, y_e) = lines[0][0]

        (x_s, y_s, x_e, y_e) = get_first_horizontal_line(lines)

        question_img = problem_img[:y_e, :w]
        question_img = Image.fromarray(question_img)

        #DEBUG
        #if count == 5:
        #    print(lines)

        problem_string = pytesseract.image_to_string(question_img, config="-c tessedit_char_whitelist=0123456789x -psm 6")
        for element in [" ","\n","\t"]:
            problem_string = problem_string.replace(element,"")
        

        # Split off given answer image from problem image
        answer_given_img = problem_img[y_e + BIAS: , :w]

        # Split and feed image to keras for prediction of handwritten answer
        prediction = preprocess_and_predict(answer_given_img)


        #DEBUG
        #if count < 6:
            #prediction = preprocess_and_predict(answer_given_img)
            #p_img = Image.fromarray(answer_given_img)
            #p_img.save("test_{}.jpg".format(count),"jpeg")
        #count += 1

        math_problem = MathProblem(problem_string, prediction)
        problems.append(math_problem)

    return problems

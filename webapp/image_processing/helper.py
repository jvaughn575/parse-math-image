import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from PIL import Image


def get_keras_model():
    # Load model
    json_file = open("model.json", 'r')
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)

    # Load weights
    model.load_weights("model.h5")

    return model

def generate_handwritten_bounding_boxes(contours):

    MIN_HEIGHT = 10
    bounding_boxes = []
    for contour in contours:
        [x,y,w,h] = cv2.boundingRect(contour)

        if h > MIN_HEIGHT:
            bounding_boxes.append((x,y,w,h))
    bounding_boxes.sort(key=lambda x: x[0])
    return bounding_boxes

def get_first_horizontal_line(lines):
    MAX_SLOPE = 30
    for x in range(0, len(lines)):
        for x1,y1,x2,y2 in lines[x]:
            m = float(y2 - y1)/(x2-x1 + .0001)
            if abs(m) < MAX_SLOPE:
                return (x1, y1, x2, y2)



def preprocess_and_predict(img_arr):
    gray_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray_arr,(5,5),0)
    ret3,thresh_arr = cv2.threshold(blur,100,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    ctrs, hier = cv2.findContours(thresh_arr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


    bounding_boxes = generate_handwritten_bounding_boxes(ctrs)

    #DEBUG
    debug_img_num = 0

    given_answer = ""
    for box in bounding_boxes:
        (x,y,w,h) = box
        crop_array = img_arr[y:y+h, x:x+w]
        gray_arr = cv2.cvtColor(crop_array, cv2.COLOR_BGR2GRAY)

        blur = cv2.GaussianBlur(gray_arr,(5,5),0)
        ret3,thresh_arr = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        thresh_im = Image.fromarray(thresh_arr)

        # Resize image to 28 x 28 for keras convo model
        BORDER_WIDTH = 15
        size = (thresh_im.height
                if thresh_im.height > thresh_im.width
                else thresh_im.width) + BORDER_WIDTH

        bg_image = Image.new('L', (size,size), "black")
        offset = ((bg_image.width - thresh_im.width) // 2, (bg_image.height - thresh_im.height) // 2)
        bg_image.paste(thresh_im, offset)
        bg_image_resized = bg_image.resize((28,28), Image.BICUBIC)

        # Get keras model and make prediction
        model = get_keras_model()
        prediction = model.predict(np.array(bg_image_resized).reshape(1,1,28,28))
        try:
            #DEBUG
            #print(prediction[0])

            given_answer += str(list(prediction[0]).index(1))
        except ValueError:
            print("No prediction")

    return given_answer

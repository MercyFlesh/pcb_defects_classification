from tensorflow.keras.models import load_model
import numpy as np
import pickle
import os
import glob
import cv2

from tools import config
from tools.ensemble_fuzzy_integrals import ensemble


def get_measures():
    with open(os.path.sep.join([config.OUTPUT_PATH, "accuracy_models.pkl"]), "rb") as f:
        accuracy_models = pickle.load(f)

    return [accuracy_models["vgg16"], accuracy_models["resnet101"], accuracy_models["inceptionv3"]]


def get_defects_list(test_name, temp_name):
    img_temp = cv2.imread(temp_name)
    img_test = cv2.imread(test_name)
    test_copy = img_test.copy()
    difference = cv2.bitwise_xor(img_test, img_temp, mask=None)
    substractGray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(substractGray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    test_copy[mask != 255] = [0, 255, 0]
    hsv = cv2.cvtColor(test_copy, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (36, 0, 0), (70, 255,255))
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    close = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    
    vgg = load_model(os.path.sep.join([config.OUTPUT_PATH, "vgg.model"]))
    resnet = load_model(os.path.sep.join([config.OUTPUT_PATH, "resnet101.model"]))
    inception = load_model(os.path.sep.join([config.OUTPUT_PATH, "inceptionV3.model"]))
    measures = get_measures()
    
    offset = 20
    predictions = list()
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        x1 = x - offset
        x2 = x + w + offset
        y1 = y - offset
        y2 = y + h + offset
        ROI = img_test[y1:y2, x1:x2]
        try:
            ROI = cv2.resize(ROI, (224, 224))
            ROI = ROI.reshape(-1, 224, 224, 3)
          
            vgg_pred = vgg.predict([ROI])[0]
            resnet_pred = resnet.predict([ROI])[0]
            inception_pred = inception.predict([ROI])[0]
            
            """if vgg_pred.argmax(axis=0) < 0.7 or resnet_pred.argmax(axis=0) < 0.6:
                continue"""
            
            pred = ensemble([vgg_pred, resnet_pred, inception_pred], measures)
            predictions.append((x1, y1, x2, y2, pred))
        except cv2.error as e:
            pass
    
    return predictions


def get_image_with_ROI(image_name, defects):
    img = cv2.imread(image_name)
    for defect in defects:
        x1, y1, x2, y2, c = defect
        cv2.rectangle(img, (x1, y1), (x2, y2), (36, 255, 10), 2)
        cv2.putText(img, config.CLASSES[c], (x1, y1), 0, 1, (180, 40, 100), 2, cv2.LINE_AA)
    
    return img


if __name__ == "__main__":
    defects = get_defects_list(f"{config.DATASET_PATH}/group00041/00041/00041086_test.jpg", f"{config.DATASET_PATH}/group00041/00041/00041086_temp.jpg")
    img = get_image_with_ROI(f"{config.DATASET_PATH}/group00041/00041/00041086_test.jpg", defects)
    cv2.imwrite('result.png', img)

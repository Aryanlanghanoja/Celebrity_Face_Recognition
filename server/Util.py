import base64
import json
from typing import Any, Dict, List, Union

import cv2
import joblib
import numpy as np
import pywt
import stubs

# from pywt import w2d

__Class_Name_to_Number: Dict[str, int] = {}
__Class_Number_to_Name: Dict[int, str] = {}
__model: Union[None, Any] = None


def w2d(img, mode="haar", level=1):
    imArray = img
    # Datatype conversions
    # convert to grayscale
    imArray = cv2.cvtColor(imArray, cv2.COLOR_RGB2GRAY)
    # convert to float
    imArray = np.float32(imArray)
    imArray /= 255
    # compute coefficients
    coeffs = pywt.wavedec2(imArray, mode, level=level)

    # Process Coefficients
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0

    # reconstruction
    imArray_H = pywt.waverec2(coeffs_H, mode)
    imArray_H *= 255
    imArray_H = np.uint8(imArray_H)

    return imArray_H

def Load_Saved_Artifacts() -> None:
    print("Loading saved artifacts.... Start")

    global __Class_Name_to_Number
    global __Class_Number_to_Name
    global __model

    with open("./Artifacts/Celebrity_Name.json", 'r') as f:
        __Class_Name_to_Number = json.load(f)
        __Class_Number_to_Name = {v: k for k,
                                  v in __Class_Name_to_Number.items()}
        
        f.close()

    if __model is None:
        with open("./Artifacts/Celebrity_Face_Recognition.pkl", 'rb') as f:
            __model = joblib.load(f)
            f.close()

    print("Loading saved artifacts.... End")


def class_number_to_name(class_num: int) -> str:
    return __Class_Number_to_Name[class_num]


def Get_Base64_test_Image_for_Virat() -> str:
    with open("./Base64.txt") as f:
        data = f.read()
    return data


def Get_cv2_Image_from_Base64_String(b64str: str) -> np.ndarray:
    encoded_data = b64str.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def Get_Cropped_Image(image_path: str, image_base64_data: str) -> List[np.ndarray]:
    face_cascade = cv2.CascadeClassifier(
        './opencv/data/haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(
        './opencv/data/haarcascades/haarcascade_eye.xml')

    if image_path:
        img = cv2.imread(image_path)
    else:
        img = Get_cv2_Image_from_Base64_String(image_base64_data)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    cropped_faces = []
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            cropped_faces.append(roi_color)
    return cropped_faces


def Classify_Image(image_base64_data: str, file_path: str = None) -> List[Dict[str, Any]]:
    imgs = Get_Cropped_Image(file_path, image_base64_data)
    result = []

    for img in imgs:
        scaled_raw_image = cv2.resize(img, (32, 32))
        img_har = w2d(img, 'db1', 5)
        scaled_img_har = cv2.resize(img_har, (32, 32))
        combined_img = np.vstack((scaled_raw_image.reshape(
            32*32*3, 1), scaled_img_har.reshape(32*32, 1)))

        len_image_array = 32*32*3 + 32*32
        final = combined_img.reshape(1, len_image_array).astype(float)
        result.append({
            # 'class': class_number_to_name(__model.predict(final)[0]),
            # 'class_probability': np.around(__model.predict_proba(final) * 100, 2).tolist()[0],
            # 'class_dictionary': __Class_Name_to_Number
            'class': class_number_to_name(__model.predict(final)[0]),
            'class_probability': np.around(__model.predict_proba(final) * 100, 2).tolist()[0],
            'class_dictionary': __Class_Name_to_Number
        })

    return result


if __name__ == "__main__":
    Load_Saved_Artifacts()
    print(Classify_Image(image_base64_data=Get_Base64_test_Image_for_Virat()))

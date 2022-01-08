"""
캐글 놀이터 1주차 : 내 얼굴로 Anoying Orange 만들기
GitHub : https://github.com/kairess/annoying-orange-face
"""

import cv2
import dlib
import os
from imutils import face_utils, resize
import numpy as np

# orange 이미지
orange_img = cv2.imread('orange.jpg')
orange_img = cv2.resize(orange_img, dsize=(512,512))

# 얼굴 인식
face_detector = dlib.get_frontal_face_detector()

# 68개 얼굴 점 파일 가져오기
cwd = os.path.abspath(os.path.dirname(__file__))
model_path = os.path.abspath(os.path.join(cwd, "shape_predictor_68_face_landmarks.dat"))
shape_predictor = dlib.shape_predictor(model_path)

# 웹캠 카메라 정보 인식
webcam = cv2.VideoCapture(0)

# left, right eye landmarks
LEFT_EYE_POINTS = [36, 37, 38,  39, 40, 41]
RIGHT_EYE_POINTS = [42, 43, 44, 45, 46, 47]

# mouth landmarks
MOUTH_POINTS = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]

while True:
    ret, frame = webcam.read()

    if not ret : # 비디오 프레임을 제대로 읽어오지 않으면 break
        break

    # frame에서 얼굴들 인식
    faces = face_detector(frame)

    # 오렌지 이미지 띄우기
    result = orange_img.copy()

    height, width = frame.shape[:2]

    le_mask = np.zeros((height, width), np.uint8)
    re_mask = np.zeros((height, width), np.uint8)
    mou_mask = np.zeros((height, width), np.uint8)

    """
    webcam frame에서 사람의 눈과 입을 
    오렌지 이미지 위에 올리기!
    """
    try:
        # 얼굴 하나
        face = faces[0]
        # landmarks
        landmarks = shape_predictor(frame, face)

        region_left = np.array([(landmarks.part(point).x, landmarks.part(point).y) for point in LEFT_EYE_POINTS])
        region_left = region_left.astype(np.int32)
        le_mask = cv2.fillPoly(le_mask, [region_left], (255, 255, 255))  # 검은 배경 흰 다각형
        le_mask_inv = cv2.bitwise_not(le_mask)
        fg_le = cv2.bitwise_and(frame, frame, mask=le_mask)
        cv2.imshow('fg_le', fg_le)

        # 눈, 입 영역만 자르고
        # resize left_eye_img = resize(left_eye_img, width=100)
        # 위치 좌표 고정해서 오렌지 이미지 위에 올리기기 (100, 200)

        region_right = np.array([(landmarks.part(point).x, landmarks.part(point).y) for point in RIGHT_EYE_POINTS])
        region_right = region_right.astype(np.int32)
        cv2.fillPoly(re_mask, [region_right], (255, 255, 255))
        re_frame = cv2.bitwise_and(frame, frame, mask=re_mask)
        cv2.imshow('re_frame', re_frame)

        # 눈, 입 영역만 자르고
        # resize right_eye_img = resize(right_eye_img, width=100)
        # 위치 좌표 고정해서 오렌지 이미지 위에 올리기기 (250, 200)

        region_mouth = np.array([(landmarks.part(point).x, landmarks.part(point).y) for point in MOUTH_POINTS])
        region_mouth = region_mouth.astype(np.int32)
        cv2.fillPoly(mou_mask, [region_mouth], (255, 255, 255))
        mou_frame = cv2.bitwise_and(frame, frame, mask=mou_mask)
        cv2.imshow('mou_frame', mou_frame)

        # 눈, 입 영역만 자르고
        # resize  mouth_img = resize(mouth_img, width=250)
        # 위치 좌표 고정해서 오렌지 이미지 위에 올리기기 (180, 320)

        # eye_mouth_frame = resize(eye_mouth_frame, width=512)

        cv2.imshow("Annoying_you", result)

    except:
        pass

    # 키보드 esc 누르면 종료
    if cv2.waitKey(1) == 27:
        break

# 메모리 해제 : 카메라 장치에서 받아온 메모리를 해제
webcam.release()

# 모든 윈도우 창을 닫는다.
cv2.destroyAllWindows()




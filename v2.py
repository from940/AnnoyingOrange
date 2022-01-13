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
orange_img = cv2.resize(orange_img, dsize=(512, 512))

# 얼굴 인식
face_detector = dlib.get_frontal_face_detector()

# 68개 얼굴 점 파일 가져오기
cwd = os.path.abspath(os.path.dirname(__file__))
model_path = os.path.abspath(os.path.join(cwd, "trained_models", "shape_predictor_68_face_landmarks.dat"))
shape_predictor = dlib.shape_predictor(model_path)

# 웹캠 카메라 정보 인식
webcam = cv2.VideoCapture(0)

while True:
    ret, frame = webcam.read()
    cv2.imshow('frame', frame)

    if not ret : # 비디오 프레임을 제대로 읽어오지 않으면 break
        break

    # frame에서 얼굴들 인식
    faces = face_detector(frame)

    # 오렌지 이미지 띄우기
    result = orange_img.copy()

    try:
        # 얼굴 하나
        face = faces[0]

        # landmarks
        landmarks = shape_predictor(frame, face)


        """
        윤곽선 
        """
        # left, right eye landmarks
        LEFT_EYE_POINTS = [36, 37, 38, 39, 40, 41]
        RIGHT_EYE_POINTS = [42, 43, 44, 45, 46, 47]
        # mouth landmarks
        MOUTH_POINTS = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]

        # left eye region
        region_left = np.array([(landmarks.part(point).x, landmarks.part(point).y) for point in LEFT_EYE_POINTS])
        region_left = region_left.astype(np.int32)

        # right eye region
        region_right = np.array([(landmarks.part(point).x, landmarks.part(point).y) for point in RIGHT_EYE_POINTS])
        region_right = region_right.astype(np.int32)

        # mouth region
        region_mouth = np.array([(landmarks.part(point).x, landmarks.part(point).y) for point in MOUTH_POINTS])
        region_mouth = region_mouth.astype(np.int32)

        # 마스크
        height, width = frame.shape[:2]
        le_mask = np.zeros((height, width), np.uint8) # 0: 검정, 255: 하양
        re_mask = np.zeros((height, width), np.uint8)
        mou_mask = np.zeros((height, width), np.uint8)

        cv2.fillPoly(le_mask, [region_left], (255, 255, 255))  # 0검은 배경 흰 다각형. 255원하는 부분이 흰색
        cv2.imshow('le_mask', le_mask)
        le_frame = cv2.bitwise_or(frame, frame, mask=le_mask)
        cv2.imshow('le_frame', le_frame)

        le_x1, le_x2 = shape[36, 0], shape[39, 0]
        le_y1, le_y2 = shape[37, 1], shape[41, 1]
        le_margin = int((le_x2 - le_x1) * 0.18)

        le_mask1 = le_mask[le_y1 - le_margin: le_y2 + le_margin,
                         le_x1 - le_margin: le_x2 + le_margin].copy()

        # le_mask1 = le_mask[y1 - margin: y2 + margin, x1 - margin: x2 + margin].copy()  # 다각형 주위 사각형 자르기
        le_mask1 = resize(le_mask1, width=100)
        cv2.imshow('le_mask', le_mask1)

        # left_eye_frame = resize(le_frame, width=1000 )
        # cv2.imshow('left_eye_frame', left_eye_frame)

        cv2.fillPoly(re_mask, [region_right], (255, 255, 255))
        re_frame = cv2.bitwise_and(frame, frame, mask=re_mask)
        # cv2.imshow('re_frame', re_frame)

        cv2.fillPoly(mou_mask, [region_mouth], (255, 255, 255))
        mou_frame = cv2.bitwise_and(frame, frame, mask=mou_mask)
        # cv2.imshow('mou_frame', mou_frame)

        cv2.imshow("Annoying_you", result)

        # resize left_eye_img = resize(left_eye_img, width=100)
        # 위치 좌표 고정해서 오렌지 이미지 위에 올리기 (100, 200)

        # resize right_eye_img = resize(right_eye_img, width=100)
        # 위치 좌표 고정해서 오렌지 이미지 위에 올리기 (250, 200)

        # resize  mouth_img = resize(mouth_img, width=250)
        # 위치 좌표 고정해서 오렌지 이미지 위에 올리기 (180, 320)

    except:
        pass

    # 키보드 esc 누르면 종료
    if cv2.waitKey(1) == 27:
        break

# 메모리 해제 : 카메라 장치에서 받아온 메모리를 해제
webcam.release()

# 모든 윈도우 창을 닫는다.
cv2.destroyAllWindows()




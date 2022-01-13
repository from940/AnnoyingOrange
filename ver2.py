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

# 웹캠 카메라 정보를 받아옴
webcam = cv2.VideoCapture(0)

while True:

    ret, frame = webcam.read()

    if not ret:  # 비디오 프레임을 제대로 읽어오지 않으면 break
        break

    # frame에서 얼굴들 인식
    faces = face_detector(frame)

    # 오렌지 이미지 띄우기
    result = orange_img.copy()

    try:
        face = faces[0]

        # 얼굴 영역 인식
        x1, x2 = face.left(), face.right()
        y1, y2 = face.top(), face.bottom()
        face_frame = frame[y1:y2, x1:x2].copy()

        # landmarks
        shape = shape_predictor(frame, face)
        shape = face_utils.shape_to_np(shape)

        """
        윤곽선
        """
        # left, right eye landmarks
        LEFT_EYE_POINTS = [36, 37, 38, 39, 40, 41]
        RIGHT_EYE_POINTS = [42, 43, 44, 45, 46, 47]
        # mouth landmarks
        MOUTH_POINTS = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]

        """
        사각 프레임
        """
        le_condition = {'x': [36, 39], 'y': [37, 41], 'width': 100, 'location': (100, 200)}
        re_condition = {'x': [42, 45], 'y': [43, 47], 'width': 100, 'location': (250, 200)}
        mou_condition = {'x': [48, 54], 'y': [50, 57], 'width': 250, 'location': (180, 320)}

        def add_part(condition, result, POINTS):
            x1, x2 = shape[condition['x'][0], 0], shape[condition['x'][1], 0]
            y1, y2 = shape[condition['y'][0], 1], shape[condition['y'][1], 1]
            margin = int((x2 - x1) * 0.18)

            new_frame = frame[y1 - margin: y2 + margin, x1 - margin: x2 + margin].copy()
            new_frame = resize(new_frame, width=condition['width'])

            # region
            region = np.array([(shape[point][0], shape[point][1]) for point in POINTS])
            region = region.astype(np.int32)

            # 마스크
            height, width = frame.shape[:2]
            mask = np.zeros((height, width), np.uint8)  # 0 검정색
            cv2.fillPoly(mask, [region], (255, 255, 255))  # 0 검정 검은 배경 흰 다각형. 255 하양 원하는 부분이 흰색

            mask = mask[y1 - margin: y2 + margin, x1 - margin: x2 + margin].copy() # 다각형 주위 사각형 자르기
            mask = resize(mask, width=condition['width'])

            part_frame = cv2.bitwise_or(new_frame, new_frame, mask=mask)
            cv2.imshow('part_frame1', part_frame)

            result =cv2.seamlessClone(
                new_frame,
                result,
                # np.full(new_frame.shape[:2], 255, new_frame.dtype),
                mask,
                condition['location'],
                # cv2.MIXED_CLONE
                cv2.NORMAL_CLONE
            )

            return result

        result= add_part(le_condition, result, LEFT_EYE_POINTS)
        result= add_part(re_condition, result, RIGHT_EYE_POINTS)
        result= add_part(mou_condition, result, MOUTH_POINTS)

        cv2.imshow("Annoying_you", result)

    except:
        pass
    # 키보드 esc 누르면 종료
    if cv2.waitKey(1) == 27:
        break

# 메모리 해제 : 카메라 장치에서 받아온 메모리를 해제한다.
webcam.release()
# 모든 윈도우 창을 닫는다.
cv2.destroyAllWindows()

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

    """
    ret, frame 반환
    비디오 프레임을 제대로 읽어오면 ret값 True, 실패하면 False
    ret값 체크하여 비디오프레임 제대로 읽었는지 확인
    """
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

        # facial landmark (x, y) 를 Numpy로 변환
        shape = face_utils.shape_to_np(shape)

        le_condition = {'x': [36, 39], 'y': [37, 41], 'width': 100, 'location': (100, 200)}
        re_condition = {'x': [42, 45], 'y': [43, 47], 'width': 100, 'location': (250, 200)}
        mou_condition = {'x': [48, 54], 'y': [50, 57], 'width': 250, 'location': (180, 320)}

        def add_part(condition, result):
            x1, x2 = shape[condition['x'][0], 0], shape[condition['x'][1], 0]
            y1, y2 = shape[condition['y'][0], 1], shape[condition['y'][1], 1]
            margin = int((x2 - x1) * 0.18)

            new_frame = frame[y1 - margin: y2 + margin, x1 - margin: x2 + margin].copy()
            new_frame = resize(new_frame, width=condition['width'])

            result =cv2.seamlessClone(
                new_frame,
                result,
                np.full(new_frame.shape[:2], 255, new_frame.dtype),
                condition['location'],
                cv2.MIXED_CLONE
            )
            return result

        result= add_part(le_condition, result)
        result= add_part(re_condition, result)
        result= add_part(mou_condition, result)

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

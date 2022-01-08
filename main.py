"""
캐글 놀이터 1주차 : 내 얼굴로 Anoying Orange 만들기
GitHub : https://github.com/kairess/annoying-orange-face
"""

import cv2
import dlib
import os
# imutils: OpenCV 기능 보완 패키지
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

# 웹캠 카메라에서 정보를 받아옴
webcam = cv2.VideoCapture(0)

while True:

    """
    ret, frame 반환
    비디오 프레임을 제대로 읽어오면 ret값 True, 실패하면 False
    ret값 체크하여 비디오프레임 제대로 읽었는지 확인
    """
    ret, frame = webcam.read()

    if not ret : # 비디오 프레임을 제대로 읽어오지 않으면 break
        break

    # frame에서 얼굴들 인식
    faces = face_detector(frame)
    result = orange_img.copy()

    """
    어노잉 오렌지는 frame에서 눈과 입을 읽어와서
    오렌지 이미지 위에 올리기!
    """

    try :
        face = faces[0]

        # 얼굴 영역 인식
        x1, x2 = face.left(), face.right()
        y1, y2 = face.top(), face.bottom()
        face_frame = frame[y1:y2, x1:x2].copy()

        shape = shape_predictor(frame, face)
        shape = face_utils.shape_to_np(shape)

        """
        Eye
        """
        le_x1, le_x2 = shape[36, 0], shape[39, 0]
        le_y1, le_y2 = shape[37, 1], shape[41, 1]
        le_margin = int((le_x2 - le_x1) * 0.18)

        re_x1, re_x2 = shape[42, 0], shape[45, 0]
        re_y1, re_y2 = shape[43, 1], shape[47, 1]
        re_margin = int((re_x2 - re_x1) * 0.18)

        left_eye_frame = frame[le_y1 - le_margin : le_y2 + le_margin ,
                         le_x1 - le_margin : le_x2 + le_margin].copy()
        left_eye_frame = resize(left_eye_frame, width=100)

        right_eye_frame = frame[re_y1 - re_margin : re_y2 + re_margin ,
                          re_x1 - re_margin : re_x2 + re_margin].copy()
        right_eye_frame = resize(right_eye_frame, width=100)

        result = cv2.seamlessClone(
            left_eye_frame,
            result,
            np.full(left_eye_frame.shape[:2], 255, left_eye_frame.dtype),
            (100, 200),
            cv2.MIXED_CLONE
        )

        result = cv2.seamlessClone(
            right_eye_frame,
            result,
            np.full(right_eye_frame.shape[:2], 255, right_eye_frame.dtype),
            (250, 200),
            cv2.MIXED_CLONE
        )


        """
        Mouth
        """

        mouth_x1, mouth_x2 = shape[48, 0], shape[54, 0]
        mouth_y1, mouth_y2 = shape[50, 1], shape[57, 1]
        mouth_margin = int((mouth_x2 - mouth_x1) * 0.1)

        mouth_frame = frame[mouth_y1 - mouth_margin : mouth_y2 + mouth_margin,
                      mouth_x1 - mouth_margin : mouth_x2 + mouth_margin]
        mouth_frame = resize(mouth_frame, width=250)

        result = cv2.seamlessClone(
            mouth_frame,
            result,
            np.full(mouth_frame.shape[:2], 255, mouth_frame.dtype),
            (180, 320),
            cv2.MIXED_CLONE
        )

        cv2.imshow("Annoying_you", result)

    except :
        pass

    # 키보드 esc 누르면 종료
    if cv2.waitKey(1) == 27:
        break

# 메모리 해제 : 카메라 장치에서 받아온 메모리를 해제한다.
webcam.release()
# 모든 윈도우 창을 닫는다.
cv2.destroyAllWindows()




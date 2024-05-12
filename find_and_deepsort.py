import cv2 as cv
import numpy as np
import datetime
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

CONFIDENCE_THRESHOLD = 0.6
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)

# YOLO 모델 초기화
model = YOLO("yolov8n.pt")

# DeepSort 추적기 초기화
tracker = DeepSort(max_age=50)

hsv = 0
lower_red1 = np.array([0, 0, 0])  # 초기값 설정
upper_red1 = np.array([0, 0, 0])  # 초기값 설정
lower_red2 = np.array([0, 0, 0])  # 초기값 설정
upper_red2 = np.array([0, 0, 0])  # 초기값 설정
lower_red3 = np.array([0, 0, 0])  # 초기값 설정
upper_red3 = np.array([0, 0, 0])  # 초기값 설정

# 마우스 이벤트 처리 함수
def mouse_callback(event, x, y, flags, param):
    global hsv, lower_red1, upper_red1, lower_red2, upper_red2, lower_red3, upper_red3

    # 마우스 왼쪽 버튼 누를시 위치에 있는 픽셀값을 읽어와서 HSV로 변환합니다.
    if event == cv.EVENT_LBUTTONDOWN:
        color = img_color[y, x]

        one_pixel = np.uint8([[color]])
        hsv = cv.cvtColor(one_pixel, cv.COLOR_BGR2HSV)
        hsv = hsv[0][0]

        # HSV 색공간에서 마우스 클릭으로 얻은 픽셀값과 유사한 필셀값의 범위를 정합니다.
        if hsv[0] < 10:
            lower_red1 = np.array([hsv[0]-10+180, 30, 30])
            upper_red1 = np.array([180, 255, 255])
            lower_red2 = np.array([0, 30, 30])
            upper_red2 = np.array([hsv[0], 255, 255])
            lower_red3 = np.array([hsv[0], 30, 30])
            upper_red3 = np.array([hsv[0]+10, 255, 255])

        elif hsv[0] > 170:
            lower_red1 = np.array([hsv[0], 30, 30])
            upper_red1 = np.array([180, 255, 255])
            lower_red2 = np.array([0, 30, 30])
            upper_red2 = np.array([hsv[0]+10-180, 255, 255])
            lower_red3 = np.array([hsv[0]-10, 30, 30])
            upper_red3 = np.array([hsv[0], 255, 255])

        else:
            lower_red1 = np.array([hsv[0], 30, 30])
            upper_red1 = np.array([hsv[0]+10, 255, 255])
            lower_red2 = np.array([hsv[0]-10, 30, 30])
            upper_red2 = np.array([hsv[0], 255, 255])
            lower_red3 = np.array([hsv[0]-10, 30, 30])
            upper_red3 = np.array([hsv[0], 255, 255])

# 마우스 이벤트 콜백 등록
cv.namedWindow('img_color')
cv.setMouseCallback('img_color', mouse_callback)

# 웹캠 캡처 초기화
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_color = cv.resize(frame, (640, 480))

    # 원본 영상을 HSV 영상으로 변환합니다.
    img_hsv = cv.cvtColor(img_color, cv.COLOR_BGR2HSV)

    # 범위 값으로 HSV 이미지에서 마스크를 생성합니다.
    img_mask1 = cv.inRange(img_hsv, lower_red1, upper_red1)
    img_mask2 = cv.inRange(img_hsv, lower_red2, upper_red2)
    img_mask3 = cv.inRange(img_hsv, lower_red3, upper_red3)
    img_mask = img_mask1 | img_mask2 | img_mask3

    # 마스크 이미지로 원본 이미지에서 범위값에 해당되는 영상 부분을 획득합니다.
    img_result = cv.bitwise_and(img_color, img_color, mask=img_mask)

    # YOLO를 통해 객체 검출
    detection = model.predict(source=[img_result], save=False, classes=[32])[0]
    results = []

    for data in detection.boxes.data.tolist(): # data : [xmin, ymin, xmax, ymax, confidence_score, class_id]
        confidence = float(data[4])
        if confidence < CONFIDENCE_THRESHOLD or int(data[5]) != 32:  # 클래스 32에 대한 조건 추가
            continue

        xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
        label = int(data[5])

        results.append([[xmin, ymin, xmax-xmin, ymax-ymin], confidence, label])

    # 추적 업데이트
    tracks = tracker.update_tracks(results, frame=img_color)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()

        xmin, ymin, xmax, ymax = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
        cv.rectangle(img_color, (xmin, ymin), (xmax, ymax), GREEN, 2)
        cv.rectangle(img_color, (xmin, ymin - 20), (xmin + 20, ymin), GREEN, -1)
        cv.putText(img_color, str(track_id), (xmin + 5, ymin - 8), cv.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)

    cv.imshow('img_color', img_color)
    cv.imshow('img_mask', img_mask)
    cv.imshow('img_result', img_result)

    # ESC 키를 누르면 종료합니다.
    if cv.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv.destroyAllWindows()

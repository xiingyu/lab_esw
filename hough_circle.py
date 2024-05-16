import cv2
import numpy as np

# 웹캠에서 영상 캡처
cap = cv2.VideoCapture(0)

# 트랙바 콜백 함수
def on_trackbar(pos):
    pass

# 트랙바 윈도우 생성
cv2.namedWindow('img')

# 트랙바 생성 및 초기화
cv2.createTrackbar('minDist', 'img', 1, 200, on_trackbar)
cv2.setTrackbarPos('minDist', 'img', 30)

cv2.createTrackbar('canny_max', 'img', 0, 500, on_trackbar)
cv2.setTrackbarPos('canny_max', 'img', 240)

cv2.createTrackbar('thres', 'img', 0, 200, on_trackbar)
cv2.setTrackbarPos('thres', 'img', 27)

cv2.createTrackbar('minRadius', 'img', 0, 200, on_trackbar)
cv2.setTrackbarPos('minRadius', 'img', 20)

cv2.createTrackbar('maxRadius', 'img', 0, 200, on_trackbar)
cv2.setTrackbarPos('maxRadius', 'img', 40)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 이미지 리사이즈
    image = cv2.resize(frame, (400, 400))
    
    # 그레이 스케일 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 노이즈 제거를 위한 가우시안 블러
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # 트랙바 값 읽기
    minDist = cv2.getTrackbarPos('minDist', 'img')
    canny_max = cv2.getTrackbarPos('canny_max', 'img')
    thres = cv2.getTrackbarPos('thres', 'img')
    minRadius = cv2.getTrackbarPos('minRadius', 'img')
    maxRadius = cv2.getTrackbarPos('maxRadius', 'img')

    # Hough Circle Transform을 사용한 원 탐지
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, minDist, param1=canny_max, param2=thres, minRadius=minRadius, maxRadius=maxRadius)

    dst = image.copy()
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # 원 둘레에 초록색 원 그리기
            cv2.circle(dst, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # 원 중심점에 빨강색 원 그리기
            cv2.circle(dst, (i[0], i[1]), 2, (0, 0, 255), 5)

    # 결과 출력
    cv2.imshow('img', dst)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 모든 창 닫기
cap.release()
cv2.destroyAllWindows()

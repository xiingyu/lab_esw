import numpy as np
import cv2

# HSV 값 확인을 위한 함수
def print_hsv_values(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # 마우스 왼쪽 버튼 클릭 시
        hsv_value = hsv_image[y, x]
        print(f"HSV 값: {hsv_value}")

# 메인 함수
def main():
    # 카메라를 통해 실시간 영상 읽어오기
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    cv2.namedWindow('HSV Image')
    cv2.setMouseCallback('HSV Image', print_hsv_values)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        global hsv_image
        hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        cv2.imshow('HSV Image', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()


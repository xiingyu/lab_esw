import cv2
import numpy as np

class LKAS:
    def __init__(self):
        pass

    def detect_color(self, img):
        # 이미지의 높이 계산
        height = img.shape[0]
        width = img.shape[1]
        
        # 아래에서 1/3의 영역 설정
        roi = img[int(height * 2/3):, :]
        
        # ROI에 대해 HSV 변환 및 마스크 적용
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        yellow_lower = np.array([15, 35, 70])
        yellow_upper = np.array([22, 255, 255])
        white_lower = np.array([0, 30, 120])
        white_upper = np.array([179, 50, 255])

        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
        white_mask = cv2.inRange(hsv, white_lower, white_upper)
        blend_mask = cv2.bitwise_or(yellow_mask, white_mask)
        blend_color = cv2.bitwise_and(roi, roi, mask=blend_mask)
        
        return blend_color, yellow_mask, white_mask

if __name__ == "__main__":
    lkas = LKAS()
    cap = cv2.VideoCapture("C:/Users/nrk52/downloads/lkas_test.mov")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        blend_color, yellow_mask, white_mask = lkas.detect_color(frame)

        cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Filtered Image", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Yellow Mask", cv2.WINDOW_NORMAL)
        cv2.namedWindow("White Mask", cv2.WINDOW_NORMAL)
        
        cv2.resizeWindow("Original Image", 400, 300)
        cv2.resizeWindow("Filtered Image", 400, 300)
        cv2.resizeWindow("Yellow Mask", 400, 300)
        cv2.resizeWindow("White Mask", 400, 300)
        
        cv2.imshow("Original Image", frame)
        cv2.imshow("Filtered Image", blend_color)
        cv2.imshow("Yellow Mask", yellow_mask)
        cv2.imshow("White Mask", white_mask)

        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

import cv2
import numpy as np

class FlagDetector:
    def __init__(self):
        # 노란색 깃발 HSV 
        self.lower_yellow = np.array([20, 100, 100])
        self.upper_yellow = np.array([30, 255, 255])
        
        # 회색 막대  HSV
        self.lower_gray = np.array([0, 0, 100])
        self.upper_gray = np.array([179, 50, 200])
        
    def detect(self, frame):

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        yellow_mask = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)

        gray_mask = cv2.inRange(hsv, self.lower_gray, self.upper_gray)

        flag_mask = cv2.bitwise_or(yellow_mask, gray_mask)

        result = np.zeros_like(frame)
        result[np.where(flag_mask == 255)] = frame[np.where(flag_mask == 255)]

        contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:

            max_contour = max(contours, key=cv2.contourArea)
            epsilon = 0.02 * cv2.arcLength(max_contour, True)
            approx = cv2.approxPolyDP(max_contour, epsilon, True)

            if len(approx) == 3:

                cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)

                points = approx.reshape(-1, 2)
                centroid = points.mean(axis=0)
                centroid = tuple(map(int, centroid))

                leftmost = tuple(max_contour[max_contour[:, :, 0].argmin()][0])
                rightmost = tuple(max_contour[max_contour[:, :, 0].argmax()][0])
                cv2.circle(frame, leftmost, 8, (255, 0, 0), -1)
                cv2.circle(frame, rightmost, 8, (255, 0, 0), -1)
                
                direction = "RIGHT" if centroid[0] < (leftmost[0] + rightmost[0]) / 2 else "LEFT"
                cv2.putText(frame, direction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                print("FLAG DETECT SUCCESS")
            else:
                # 삼각형이 아닐 때 처리
                print("FLAG DETECT FAIL")
        
        return frame

cap = cv2.VideoCapture(4)


detector = FlagDetector()

while True:
    ret, frame = cap.read()
    
    if not ret:
        break

    detected_frame = detector.detect(frame)

    cv2.imshow('Flag Detection', detected_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

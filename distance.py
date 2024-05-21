import cv2
import numpy as np
import math

class DistanceCalculator:
    def __init__(self, W_View_size=1920, H_View_size=1080, KNOWN_WIDTH=12, FOV=140):
        self.W_View_size = W_View_size
        self.H_View_size = H_View_size
        self.KNOWN_WIDTH = KNOWN_WIDTH
        self.FOV = FOV

        # 노란색 HSV 범위
        self.yellow_lower = np.array([20, 140, 100], np.uint8)
        self.yellow_upper = np.array([30, 255, 255], np.uint8)

        self.cap = cv2.VideoCapture(1)

    def find_yellow_circle_center(self, frame):
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        yellow_mask = cv2.inRange(hsv_frame, self.yellow_lower, self.yellow_upper)
        contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            c = max(contours, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            if radius > 10:  # 최소 반지름 크기 조건
                return (x, y), radius
        return None, None

    def calculate_distance(self, perceived_width):
        return self.KNOWN_WIDTH * self.W_View_size / (2 * perceived_width * math.tan(math.radians(self.FOV / 2)))

    def process_frame(self, frame):
        center, radius = self.find_yellow_circle_center(frame)
        if center is not None:
            x, y = center
            distance = self.calculate_distance(radius * 2)  # 지름을 사용하여 거리 계산
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.putText(frame, f"Distance: {distance:.2f} cm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return frame

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue

            processed_frame = self.process_frame(frame)

            cv2.imshow("Frame", processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    distance_calculator = DistanceCalculator()
    distance_calculator.run()

import cv2
import numpy as np
import math

class DistanceCalculator:
    def __init__(self, W_View_size=1920, H_View_size=1080, KNOWN_WIDTH_HOLE=20, FOV=150):
        self.W_View_size = W_View_size
        self.H_View_size = H_View_size
        self.KNOWN_WIDTH_HOLE = KNOWN_WIDTH_HOLE
        self.FOV = FOV

        # 노란색 HSV 범위
        self.yellow_lower = np.array([18, 140, 100], np.uint8)
        self.yellow_upper = np.array([36, 255, 255], np.uint8)

        self.cap = cv2.VideoCapture(1)

    def find_circle_center(self, frame, lower_bound, upper_bound):
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            c = max(contours, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            if radius > 10:
                return (x, y), radius
        return None, None

    def calculate_distance(self, perceived_width, known_width):
        return (known_width * self.W_View_size) / (2 * perceived_width * math.tan(math.radians(self.FOV / 2)))

    def process_frame(self, frame):
        # 노란색 홀의 중심과 반지름 찾기
        hole_center, hole_radius = self.find_circle_center(frame, self.yellow_lower, self.yellow_upper)
        if hole_center is not None:
            x_hole, y_hole = hole_center
            distance_hole = self.calculate_distance(hole_radius * 2, self.KNOWN_WIDTH_HOLE)
            cv2.circle(frame, (int(x_hole), int(y_hole)), int(hole_radius), (0, 255, 255), 2)
            cv2.putText(frame, f"Hole find: {x_hole:.2f}, {y_hole:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Distance: {distance_hole:.2f} units", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame, hole_center

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue

            processed_frame, hole_center = self.process_frame(frame)

            cv2.imshow("Frame", processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    distance_calculator = DistanceCalculator()
    distance_calculator.run()

import cv2
import numpy as np
from hall_video import DistanceCalculator

class BallDetector:
    def __init__(self, actual_diameter_meters=0.05, focal_length=403, neck_angle=60, area_threshold=4000):
        self.actual_diameter_meters = actual_diameter_meters
        self.actual_area_meters = np.pi * (self.actual_diameter_meters / 2) ** 2
        self.focal_length = focal_length
        self.neck_angle = neck_angle
        self.area_threshold = area_threshold
        self.distance_calculator = DistanceCalculator()

    def detect_red_ball(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_red = np.array([170, 100, 150])
        upper_red = np.array([180, 255, 255])
        mask = cv2.inRange(hsv, lower_red, upper_red)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)

            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0

            ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)
            center = (int(cX), int(cY))
            radius = int(radius)
            cv2.circle(frame, center, radius, (0, 255, 0), 2)
            cv2.drawContours(frame, [largest_contour], -1, (255, 0, 0), 2)

            cv2.circle(frame, center, 5, (0, 0, 255), -1)
            cv2.putText(frame, "Red Ball Center", (center[0] - 20, center[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            pixel_area = np.pi * (radius ** 2)
            distance = self.focal_length * np.sqrt(self.actual_area_meters / pixel_area)
            return frame, center, distance

        return frame, None, None

    def run(self):
        cap = cv2.VideoCapture(1)

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame, red_ball_center, _ = self.detect_red_ball(frame)
            _, hole_center = self.distance_calculator.process_frame(frame)

            if red_ball_center and hole_center:
                # Convert to integer tuples
                red_ball_center = (int(red_ball_center[0]), int(red_ball_center[1]))
                hole_center = (int(hole_center[0]), int(hole_center[1]))
                cv2.line(frame, red_ball_center, hole_center, (0, 0, 255), 2)
                
                # Check if the centers are aligned on the x-axis
                if abs(red_ball_center[0] - hole_center[0]) < 10:  # 허용 오차를 10 픽셀로 설정
                    cv2.putText(frame, "Aligned on X-axis", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                else:
                    cv2.putText(frame, "Not Aligned on X-axis", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    ball_detector = BallDetector()
    ball_detector.run()

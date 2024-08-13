import cv2
import numpy as np

class RedBallDetector:
    def __init__(self, actual_diameter_meters=0.04267, focal_length=403, neck_angle=60):
        self.actual_diameter_meters = actual_diameter_meters
        self.actual_area_meters = np.pi * (self.actual_diameter_meters / 2) ** 2
        self.focal_length = focal_length
        self.neck_angle = neck_angle
        self.neck_angle_radians = np.radians(self.neck_angle)
        
    def detect_and_calculate_distance(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_red = np.array([170, 100, 150])
        upper_red = np.array([180, 255, 255])
        mask = cv2.inRange(hsv, lower_red, upper_red)

        cv2.imshow("Mask", mask)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)

            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0

            frame_height, frame_width = frame.shape[:2]
            segment_width = frame_width // 3
            segment_height = frame_height // 3

            case_number = self.determine_case_number(cX, cY, segment_width, segment_height)

            ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)
            center = (int(cX), int(cY))
            radius = int(radius)
            cv2.circle(frame, center, radius, (0, 255, 0), 2)
            cv2.drawContours(frame, [largest_contour], -1, (255, 0, 0), 2)

            cv2.circle(frame, center, 5, (0, 0, 255), -1)
            cv2.putText(frame, "Center", (center[0] - 20, center[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            pixel_area = np.pi * (radius ** 2)

            if pixel_area == 0:
                return frame, None, None, None, None, case_number

            distance = self.focal_length * np.sqrt(self.actual_area_meters / pixel_area)
            walk_dist = distance * np.cos(np.pi / 2 - self.neck_angle_radians)

            bottom_center = (frame_width // 2, frame_height)
            cv2.line(frame, bottom_center, center, (255, 255, 0), 2)

            angle_degrees = self.calculate_angle(bottom_center, center)
            cv2.putText(frame, f"Angle: {angle_degrees:.2f} degrees", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            return frame, distance, center, pixel_area, walk_dist, case_number

        return frame, None, None, None, None, None

    def determine_case_number(self, cX, cY, segment_width, segment_height):
        if cX < segment_width and cY < segment_height:
            return 1
        elif segment_width <= cX < 2 * segment_width and cY < segment_height:
            return 2
        elif cX >= 2 * segment_width and cY < segment_height:
            return 3
        elif cX < segment_width and segment_height <= cY < 2 * segment_height:
            return 4
        elif segment_width <= cX < 2 * segment_width and segment_height <= cY < 2 * segment_height:
            return 5
        elif cX >= 2 * segment_width and segment_height <= cY < 2 * segment_height:
            return 6
        elif cX < segment_width and cY >= 2 * segment_height:
            return 7
        elif segment_width <= cX < 2 * segment_width and cY >= 2 * segment_height:
            return 8
        elif cX >= 2 * segment_width and cY >= 2 * segment_height:
            return 9
        return None

    def calculate_angle(self, bottom_center, center):
        angle = np.arctan2(center[1] - bottom_center[1], center[0] - bottom_center[0])
        return np.degrees(angle)

def main():
    cap = cv2.VideoCapture(1)

    detector = RedBallDetector()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame, distance, center, pixel_area, walk_dist, case_number = detector.detect_and_calculate_distance(frame)
        if distance:
            cv2.putText(frame, f"Distance: {distance:.2f} m", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Area: {pixel_area:.2f} pixels^2", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Walk Dist: {walk_dist:.2f} m", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Case: {case_number}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

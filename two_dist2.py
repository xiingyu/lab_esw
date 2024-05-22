import cv2
import numpy as np
import math

class DistanceCalculator:
    def __init__(self, W_View_size=1920, H_View_size=1080, KNOWN_WIDTH_HOLE=12, KNOWN_WIDTH_BALL=6, FOV=140):
        self.W_View_size = W_View_size
        self.H_View_size = H_View_size
        self.KNOWN_WIDTH_HOLE = KNOWN_WIDTH_HOLE
        self.KNOWN_WIDTH_BALL = KNOWN_WIDTH_BALL
        self.FOV = FOV

        # 노란색 HSV 범위
        self.yellow_lower = np.array([20, 140, 100], np.uint8)
        self.yellow_upper = np.array([30, 255, 255], np.uint8)

        '''
        [  0 240 255]
        case3
        [  0 218 246]
        case3
        [  9 232 255]
        '''
        # 빨간색 HSV 범위
        self.red_lower = np.array([170, 140, 150], np.uint8)
        self.red_upper = np.array([180, 255, 255], np.uint8)

        '''
        [100  47 245]
        case2
        [ 97  45 235]
        case2 hsv[0] > 170:
        lower = 100 | (100+10-180) | 100-10
            lower_blue1 = np.array([hsv[0], 30, 30])
            upper_blue1 = np.array([180, 255, 255])
            lower_blue2 = np.array([0, 30, 30])
            upper_blue2 = np.array([hsv[0]+10-180, 255, 255])
            lower_blue3 = np.array([hsv[0]-10, 30, 30])
            upper_blue3 = np.array([hsv[0], 255, 255])
        '''

        self.cap = cv2.VideoCapture(1)

    def find_circle_center(self, frame, lower_bound, upper_bound):
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            c = max(contours, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            if radius > 10:  # 최소 반지름 크기 조건
                return (x, y), radius
        return None, None

    def calculate_distance(self, perceived_width, known_width):
        return known_width * self.W_View_size / (2 * perceived_width * math.tan(math.radians(self.FOV / 2)))

    def process_frame(self, frame):
        # 노란색 홀의 중심과 반지름 찾기
        hole_center, hole_radius = self.find_circle_center(frame, self.yellow_lower, self.yellow_upper)
        if hole_center is not None:
            x_hole, y_hole = hole_center
            distance_hole = self.calculate_distance(hole_radius * 2, self.KNOWN_WIDTH_HOLE)
            cv2.circle(frame, (int(x_hole), int(y_hole)), int(hole_radius), (0, 255, 255), 2)
            cv2.putText(frame, f"Hole Distance: {distance_hole:.2f} cm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 빨간색 공의 중심과 반지름 찾기
        ball_center, ball_radius = self.find_circle_center(frame, self.red_lower, self.red_upper)
        if ball_center is not None:
            x_ball, y_ball = ball_center
            distance_ball = self.calculate_distance(ball_radius * 2, self.KNOWN_WIDTH_BALL)
            cv2.circle(frame, (int(x_ball), int(y_ball)), int(ball_radius), (0, 0, 255), 2)
            cv2.putText(frame, f"Ball Distance: {distance_ball:.2f} cm", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 노란색 홀과 빨간색 공 사이의 거리 계산 및 그리기
        if hole_center is not None and ball_center is not None:
            cv2.line(frame, (int(x_hole), int(y_hole)), (int(x_ball), int(y_ball)), (255, 0, 0), 2)
            distance_between_centers = math.sqrt((x_hole - x_ball) ** 2 + (y_hole - y_ball) ** 2)
            cv2.putText(frame, f"Center Distance: {distance_between_centers:.2f} pixels", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
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

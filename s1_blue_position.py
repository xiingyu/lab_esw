## 파란색 배열의 경우의 수 파악 
## update :: 24.08.08

import cv2
import numpy as np

class DirectionDetector: # 파랑 빨강 원형 배열에 따른 방향 결정
    def __init__(self, camera):
        self.camera = camera # 카메라 객체를 외부에서 주입 > 리소스 절약된다고,,함,,, 그러면은 전처리에 대한 것 먼저 해서 그걸 다른 코드에 주입,,? 해야하는 것 같다
        self.blue_lower = np.array([100, 150, 50])  # 파란색 하한값
        self.blue_upper = np.array([140, 255, 255])  # 파란색 상한값
        self.red_lower = np.array([170, 140, 150])  # 빨간색 하한값
        self.red_upper = np.array([180, 255, 255])  # 빨간색 상한값

    def get_image(self):
        ret, frame = self.camera.read()
        if not ret:
            print("Failed to capture image")
            return None
        return frame

    def preprocess_image(self, image, lower, upper):
        if image is None:
            return None
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        return mask

    def find_dots(self, mask):
        if mask is None:
            return [], []
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dots = []
        for contour in contours:
            if cv2.contourArea(contour) > 10:  # 작은 노이즈 제거
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    cX = int(M['m10'] / M['m00'])
                    cY = int(M['m01'] / M['m00'])
                    dots.append((cX, cY))
        return dots, contours

    def determine_direction(self, blue_dots, red_dot):
        if len(blue_dots) < 2 or not red_dot:
            return "Cannot determine direction. Not enough dots detected."

        # 모든 점들을 포함한 직선 검출
        dots = blue_dots + [red_dot]

        # x 좌표 차이와 y 좌표 차이를 계산하여 방향 결정
        x_coords = [dot[0] for dot in dots]
        y_coords = [dot[1] for dot in dots]

        x_diff = max(x_coords) - min(x_coords)
        y_diff = max(y_coords) - min(y_coords)

        if x_diff > y_diff:
            return "2번 방향"  # 가로로 정렬됨
        else:
            return "1번 방향"  # 세로로 정렬됨

    def find_red_position_in_1st_direction(self, blue_dots, red_dot):
        if len(blue_dots) < 2:
            return "Cannot determine position. Not enough blue dots."
        blue_dots = sorted(blue_dots, key=lambda x: x[1])  # y 좌표 기준으로 정렬

        if red_dot[1] < blue_dots[0][1]:
            return "case1"
        elif red_dot[1] < blue_dots[1][1]:
            return "case2"
        else:
            return "case3"

    def find_red_position_in_2nd_direction(self, blue_dots, red_dot):
        if len(blue_dots) < 2:
            return "Cannot determine position. Not enough blue dots."
        blue_dots = sorted(blue_dots, key=lambda x: x[0])  # x 좌표 기준으로 정렬

        if red_dot[0] < blue_dots[0][0]:
            return "case1"
        elif red_dot[0] < blue_dots[1][0]:
            return "case2"
        else:
            return "case3"

    def draw_dots(self, image, blue_dots, red_dot, blue_contours, red_contours):
        if image is None:
            return
        # Draw contours around the blue and red dots
        cv2.drawContours(image, blue_contours, -1, (0, 255, 0), 2)
        cv2.drawContours(image, red_contours, -1, (0, 0, 255), 2)
        
        for dot in blue_dots:
            cv2.circle(image, dot, 5, (0, 255, 0), -1)
        
        if red_dot:
            cv2.circle(image, red_dot, 5, (0, 0, 255), -1)
        
        cv2.imshow('Direction', image)
        cv2.imshow('Blue Mask', self.preprocess_image(image, self.blue_lower, self.blue_upper))
        cv2.imshow('Red Mask', self.preprocess_image(image, self.red_lower, self.red_upper))


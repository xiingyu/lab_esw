## update :: 24.09.04
## 테스트 했던 각도(목 각도 여러개 함)_distance(mm) :: 500 // 700 // 900 //1080
## distance로 구하고자 했던 walk_dist(자로 쟀을 땐) :: 350 // 600 // 800 // 1000 

## focal length :: 3.04mm

import cv2
import numpy as np

class ObjectDetector:
    def __init__(self, lower_red, upper_red, lower_yellow, upper_yellow, tolerance=20, focal_length=500, center_tolerance=50):
        self.lower_red = lower_red
        self.upper_red = upper_red
        self.lower_yellow = lower_yellow
        self.upper_yellow = upper_yellow
        self.state = "FIND_CIRCLES"
        self.tolerance = tolerance  # x 좌표 차이 허용 오차 (픽셀 단위)
        self.focal_length = focal_length  # 카메라의 초점 거리 (픽셀 단위)
        self.center_tolerance = center_tolerance  # 중앙 하단 부분 확인을 위한 허용 오차
    
    def find_circles(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 빨간색 마스크 생성
        red_mask = cv2.inRange(hsv, self.lower_red, self.upper_red)
        
        # 노란색 마스크 생성
        yellow_mask = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)
        
        result = frame.copy()
        red_circle = None
        yellow_circle = None
        
        # 빨간색 마스크에서 가장 큰 컨투어 찾기
        contours_red, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours_red:
            largest_contour_red = max(contours_red, key=cv2.contourArea)
            if len(largest_contour_red) > 0:
                (x_red, y_red), radius_red = cv2.minEnclosingCircle(largest_contour_red)
                center_red = (int(x_red), int(y_red))
                radius_red = int(radius_red)
                red_circle = (center_red, radius_red)
                cv2.circle(result, center_red, radius_red, (0, 255, 0), 2)  # 둘러싸는 원 표시
        
        return result, red_circle
    
    def calculate_distance(self, radius_red):
        # 공의 실제 직경(미터)
        actual_diameter_meters = 0.05
        actual_area_meters = np.pi * (actual_diameter_meters / 2) ** 2  # 공의 실제 면적
        
        # 픽셀 면적 계산
        pixel_area = np.pi * (radius_red ** 2)
        
        # 거리를 계산하여 반환 (거리 = 초점 거리 * √(실제 면적 / 픽셀 면적))
        if pixel_area > 0:
            distance = self.focal_length * np.sqrt(actual_area_meters / pixel_area)
            return distance
        else:
            return None
    
    def calculate_walk_distance(self, distance):
        # walk_dist^2 = distance^2 - (0.333m)^2
        height_offset = 0.33
        if distance > height_offset:
            walk_dist = np.sqrt(distance**2 - height_offset**2)
        else:
            walk_dist = 0  # 거리 계산이 불가능한 경우 0으로 반환
        
        #print(f"[DEBUG] Calculated Walk Distance: {walk_dist:.2f} meters")
        return walk_dist
    
    def process_frame(self, frame):
        result = frame.copy()
        result, red_circle = self.find_circles(frame)
        
        # 정렬 여부와 관계없이 거리 계산
        if red_circle:
            distance = self.calculate_distance(red_circle[1])
            if distance is not None:
                # walk_dist 계산
                walk_dist = self.calculate_walk_distance(distance)
                
                # 거리를 화면에 표시
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                color = (255, 0, 0)
                thickness = 3
                distance_text = f"Distance: {distance:.2f} meters"
                cv2.putText(result, distance_text, (10, 50), font, font_scale, color, thickness)
                
                # walk_dist 표시
                walk_dist_text = f"Walk Distance: {walk_dist:.2f} meters"
                cv2.putText(result, walk_dist_text, (10, 100), font, font_scale, color, thickness)
        
        return result

# 동영상 파일 처리
detector = ObjectDetector(
    lower_red=np.array([170, 80, 200]),
    upper_red=np.array([180, 255, 255]),
    
    lower_yellow=np.array([10, 70, 70]),
    upper_yellow=np.array([25, 255, 255]),
    tolerance=30,  # x 좌표 차이 허용 오차
    focal_length=765.6,  # 초점 거리
    center_tolerance=50  # 중앙 하단 확인을 위한 허용 오차
)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    processed_frame = detector.process_frame(frame)
    
    # 화면에 프레임 표시
    cv2.imshow('Processed Frame', processed_frame)
    
    # 'q' 키를 누르면 종료
    if cv2.waitKey(33) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

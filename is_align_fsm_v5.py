import cv2
import numpy as np

class ObjectDetector:
    def __init__(self, lower_red, upper_red, lower_yellow, upper_yellow, tolerance=20, focal_length=500, center_tolerance=50):
        self.lower_red = lower_red
        self.upper_red = upper_red
        self.lower_yellow = lower_yellow
        self.upper_yellow = upper_yellow
        self.state = "FIND_CIRCLES"
        self.tolerance = tolerance  # y 좌표 차이 허용 오차 (픽셀 단위)
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
        
        # 노란색 마스크에서 가장 큰 컨투어 찾기
        contours_yellow, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours_yellow:
            largest_contour_yellow = max(contours_yellow, key=cv2.contourArea)
            if len(largest_contour_yellow) > 0:
                (x_yellow, y_yellow), radius_yellow = cv2.minEnclosingCircle(largest_contour_yellow)
                center_yellow = (int(x_yellow), int(y_yellow))
                radius_yellow = int(radius_yellow)
                yellow_circle = (center_yellow, radius_yellow)
                cv2.circle(result, center_yellow, radius_yellow, (0, 255, 255), 2)  # 둘러싸는 원 표시
        
        return result, red_circle, yellow_circle
    
    def check_alignment(self, red_circle, yellow_circle):
        if red_circle and yellow_circle:
            red_center = red_circle[0]
            yellow_center = yellow_circle[0]
            
            # x 좌표를 기준으로 두 원이 정렬되었는지 확인
            sorted_dots = sorted([red_center, yellow_center], key=lambda x: x[0])
            y_diff = abs(sorted_dots[1][1] - sorted_dots[0][1])
            
            if y_diff < self.tolerance:  # y 좌표의 차이가 허용 오차 이하이면 동일 선상에 있다고 간주
                return True, "center-yesssssssss"
            else:
                return False, "no alignment"
        return False, ""
    
    def calculate_distance(self, radius_red):
        # 공의 실제 직경(미터)
        actual_diameter_meters = 0.04267  
        actual_area_meters = np.pi * (actual_diameter_meters / 2) ** 2  # 공의 실제 면적
        
        # 픽셀 면적 계산
        pixel_area = np.pi * (radius_red ** 2)
        
        # 거리를 계산하여 반환 (거리 = 초점 거리 * √(실제 면적 / 픽셀 면적))
        if pixel_area > 0:
            distance = self.focal_length * np.sqrt(actual_area_meters / pixel_area)
            return distance
        else:
            return None
    
    def is_red_circle_centered_bottom(self, red_circle, frame_shape):
        center_red = red_circle[0]
        frame_height, frame_width = frame_shape[:2]
        
        # 중앙 하단 위치 계산
        expected_x = frame_width // 2
        expected_y = int(frame_height * 0.75)  # 화면 하단 1/4 지점
        
        # 허용 오차 범위 내에 있는지 확인
        if abs(center_red[0] - expected_x) < self.center_tolerance and abs(center_red[1] - expected_y) < self.center_tolerance:
            return True, "Red circle is centered at the bottom."
        else:
            return False, "Red circle is not centered."
    
    def draw_guidelines(self, frame, yellow_circle):
        height, width, _ = frame.shape
        center_x = width // 2

        # 화면 중앙에 회색 가상선
        cv2.line(frame, (center_x, 0), (center_x, height), (192, 192, 192), 2)

        if yellow_circle:
            center_yellow = yellow_circle[0]
            # 노란색 공의 중심에서 회색 선까지 수평 핑크색 선 그리기
            cv2.line(frame, (center_yellow[0], center_yellow[1]), (center_x, center_yellow[1]), (255, 105, 180), 2)
            
            # 핑크색 선의 길이 계산
            pink_line_length = abs(center_x - center_yellow[0])
            
            # 핑크색 선의 길이를 화면에 표시
            cv2.putText(frame, f"Length: {pink_line_length} px", (center_yellow[0] + 10, center_yellow[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 105, 180), 2)
            
            # 각도 계산 (회색 가상선과 노란 원의 중심에서 회색 가상선 아래점을 잇는 선)
            bottom_center = np.array([center_x, height])  # 회색선의 가장 아래점
            yellow_center = np.array([center_yellow[0], center_yellow[1]])  # 노란 원의 중심

            dx = bottom_center[0] - yellow_center[0]
            dy = bottom_center[1] - yellow_center[1]

            angle = np.degrees(np.arctan2(dy, dx))

            # 회색 가상선이 0도이므로, 수평 오른쪽(양수), 수평 왼쪽(음수)로 각도를 표시
            angle_from_vertical = 90 - angle if angle >= 0 else - (90 + angle)
            
            # 각도를 화면에 표시
            cv2.putText(frame, f"Angle: {angle_from_vertical:.2f} degrees", (center_yellow[0] + 10, center_yellow[1] + 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 105, 180), 2)
    
    def process_frame(self, frame):
        result = frame.copy()
        if self.state == "FIND_CIRCLES":
            result, red_circle, yellow_circle = self.find_circles(frame)
            if red_circle and yellow_circle:
                self.state = "CHECK_ALIGNMENT"
            return result
        
        elif self.state == "CHECK_ALIGNMENT":
            result, red_circle, yellow_circle = self.find_circles(frame)
            aligned, message = self.check_alignment(red_circle, yellow_circle)
            
            # 화면 중앙에 메시지 출력
            height, width, _ = result.shape
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            color = (0, 0, 255) if not aligned else (0, 255, 0)
            thickness = 2
            text_size = cv2.getTextSize(message, font, font_scale, thickness)[0]
            text_x = int((width - text_size[0]) / 2)
            text_y = int((height + text_size[1]) / 2)
            cv2.putText(result, message, (text_x, text_y), font, font_scale, color, thickness)
            
            if aligned and red_circle:
                distance = self.calculate_distance(red_circle[1])
                if distance:
                    distance_text = f"Distance: {distance:.2f} meters"
                    cv2.putText(result, distance_text, (text_x, text_y + 40), font, font_scale, color, thickness)
                
            # 임시로 다음 단계로 넘어가기 위한 대기
            key = cv2.waitKey(33)
            if key == ord('t'):
                self.state = "CHECK_RED_POSITION"  # t를 누르면 다음 단계로 전환
                print("다음 단계로 넘어갑니다.")
            
            return result
        
        elif self.state == "CHECK_RED_POSITION":
            result, red_circle, _ = self.find_circles(frame)
            if red_circle:
                centered, message = self.is_red_circle_centered_bottom(red_circle, frame.shape)
                
                # 화면에 메시지 출력
                height, width, _ = result.shape
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                color = (0, 255, 0) if centered else (0, 0, 255)
                thickness = 2
                text_size = cv2.getTextSize(message, font, font_scale, thickness)[0]
                text_x = int((width - text_size[0]) / 2)
                text_y = int((height + text_size[1]) / 2)
                cv2.putText(result, message, (text_x, text_y), font, font_scale, color, thickness)
                
                # y 버튼으로 다음 단계로 넘어가기
                key = cv2.waitKey(33)
                if key == ord('y'):
                    self.state = "DRAW_LINES"  # 다음 단계로 전환
                    print("다음 단계로 넘어갑니다.")
            
            return result
        
        elif self.state == "DRAW_LINES":
            result, _, yellow_circle = self.find_circles(frame)
            self.draw_guidelines(result, yellow_circle)
            return result

# 동영상 파일 처리
detector = ObjectDetector(
    lower_red=np.array([170, 90, 220]),
    upper_red=np.array([180, 255, 255]),
    
    lower_yellow=np.array([10, 70, 70]),
    upper_yellow=np.array([25, 255, 255]),
    tolerance=157,  # y 좌표 차이 허용 오차
    focal_length=500,  # 초점 거리
    center_tolerance=50  # 중앙 하단 확인을 위한 허용 오차
)

cap = cv2.VideoCapture('./re_alignment_case1.mp4')

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

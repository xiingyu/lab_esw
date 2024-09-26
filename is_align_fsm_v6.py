## update :: 24.09.26
## Resolution: 928(h)x724(w) <<- frame.shape로 구한 것임 얘를 사용해서 구한 초점 거리 값이 765.6
## 초점 거리--> focal_length_pixels= (sensor_width_mm * sensor_width_pixels) / focal_length_mm
## 라즈베리파이 카메라 v2.1( Sony IMX219 )의 물리적 초점 거리는 약 3.04mm, 센서의 크기는 3.68mm x 2.76mm
## 19일 이후 모션 제어 추가하려면 :: self.serial_comm.send_data(28)으로 번호만 수정하면 Motion.py(준렬코드)에 숫자 입력으로 들어감 (아직 모션에 대한 fsm 구분을 안함)
## 26일부터 v6로 파일 변경! >> FIND_CIRCLES > CHECK_ALIGNMENT 이 다음에 빨간색 공까지의 거리를 추정할 것임(RED_DIST)


from Motion import SerialCommunication

import cv2
import numpy as np
import time

class ObjectDetector:
    def __init__(self, lower_red, upper_red, lower_yellow, upper_yellow, tolerance=20, focal_length=500, center_tolerance=50):
        self.lower_red = lower_red
        self.upper_red = upper_red
        self.lower_yellow = lower_yellow
        self.upper_yellow = upper_yellow

        self.state = "FIND_CIRCLES" # 초기 state 지정한 것!!
        
        self.tolerance = tolerance  # x 좌표 차이 허용 오차 (픽셀 단위)
        self.focal_length = focal_length  # 카메라의 초점 거리 (픽셀 단위)
        self.center_tolerance = center_tolerance  # 중앙 하단 부분 확인을 위한 허용 오차
        self.serial_comm = SerialCommunication() # 준렬 코드를 클래스로 변경하고 이걸 추가함
        self.last_movement_time = None  # 마지막으로 모션이 수행된 시간을 저장
    
    '''수정할 것 :: 공이 없으면 어떻게 할 지?'''
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
            x_diff = abs(red_center[0] - yellow_center[0])
            
            if x_diff < self.tolerance:  # x 좌표의 차이가 허용 오차 이하이면 정렬되었다고 간주
                print("정렬됨을 확인함. go_3stet으로 이동")
                self.state = "RED_DIST" # RED_DIST는 빨간공까지의 거리만 추정하는 것
                return True, "Align!!!!!!!!!!!!!!!!!!!!"
            else:
                print("틀어짐을 확인함. try_alignment로 이동")
                self.state = "TRY_ALIGNMENT" # TRY_ALIGNMENT 상태로 이동하면 모션만 있음
                return False, "Not aligned..."
        return False, ""
    
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

    def go_3step(self):
        self.serial_comm.send_data(28)
        time.sleep(1)
        self.serial_comm.send_data(30)
        time.sleep(1)
        self.serial_comm.send_data(21)
        self.last_movement_time = time.time()  # 모션이 끝났음을 알리기 위해 시간을 기록
        self.state = "WAIT"  # 일시적으로 대기 상태로 전환

        return None
    
    def try_alignment(self): # 여기에서 정렬을 맞추려는 동작 몇개 하다가 다시 정렬확인스테이트로 넘기기
        self.serial_comm.send_data(15)
        time.sleep(1)
        self.serial_comm.send_data(20)
        time.sleep(1)
        self.serial_comm.send_data(15)
        self.last_movement_time = time.time()  # 모션이 끝났음을 알리기 위해 시간을 기록
        self.state = "WAIT"  # 일시적으로 대기 상태로 전환

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

            # 노란색 공의 중심에서 회색 선의 가장 끝지점까지 선 그리기
            cv2.line(frame, (center_yellow[0], center_yellow[1]), (center_x, height), (255, 255, 0), 2)
            
            # 각도 계산 (노란 원 중심과 회색 가상선이 만나는 지점까지의 빨간색 각도)
            dy = center_yellow[1] - height  # y 좌표 차이
            dx = center_yellow[0] - center_x  # x 좌표 차이

            angle = 180 - np.degrees(np.arctan2(dx, dy))  # 각도를 degree로 변환

            # 각도를 화면에 표시
            cv2.putText(frame, f"Angle: {angle:.2f} degrees", (center_yellow[0] + 10, center_yellow[1] + 30), 
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

            return result
        
        elif self.state == "RED_DIST":
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

                    self.state = "GO_3STEP"
            
            return result
        
        elif self.state == "GO_3STEP":
            self.go_3step()
            return result
        
        elif self.state == "TRY_ALIGNMENT":
            self.try_alignment()
            return result

        elif self.state == "WAIT":
            # last_movement_time이 None이 아니어야 시간을 비교할 수 있음
            if self.last_movement_time is not None and time.time() - self.last_movement_time > 2:
                self.state = "FIND_CIRCLES"
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
    tolerance=30,  # x 좌표 차이 허용 오차
    focal_length=765.6,  # 초점 거리--> focal_length_pixels= (sensor_width_mm * sensor_width_pixels) / focal_length_mm

    center_tolerance=50  # 중앙 하단 확인을 위한 허용 오차
)

#cap = cv2.VideoCapture('./alignment_case1.mov')
cap = cv2.VideoCapture('./re_alignment_case1.mp4')
#cap = cv2.VideoCapture('./dist_test.mp4')

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

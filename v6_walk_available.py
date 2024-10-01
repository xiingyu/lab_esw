## update :: 24.10.01
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
    
    def find_circles(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 빨간색 마스크 생성
        red_mask = cv2.inRange(hsv, self.lower_red, self.upper_red)
        
        result = frame.copy()
        red_circle = None
        
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
                self.state = "RED_DIST"

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

        return walk_dist

    def go_3step(self):
        print("Executing go_3step")
        self.serial_comm.send_data(28)
        time.sleep(1)
        self.serial_comm.send_data(30)
        time.sleep(1)
        self.serial_comm.send_data(21)
        self.last_movement_time = time.time()  # 모션이 끝났음을 알리기 위해 시간을 기록
        self.state = "WAIT"  # 일시적으로 대기 상태로 전환

        return None
    
    def try_alignment(self): # 여기에서 정렬을 맞추려는 동작 몇개 하다가 다시 정렬확인스테이트로 넘기기
        print("Executing try_alignment")
        self.serial_comm.send_data(15)
        time.sleep(1)
        self.serial_comm.send_data(20)
        time.sleep(1)
        self.serial_comm.send_data(15)
        self.last_movement_time = time.time()  # 모션이 끝났음을 알리기 위해 시간을 기록
        self.state = "WAIT"  # 일시적으로 대기 상태로 전환

        return None
    
    def far_ball_motion(self):
        print("Executing far_ball_motion")
        self.serial_comm.send_data(15)
        time.sleep(1)
        self.serial_comm.send_data(18)
        time.sleep(1)
        self.serial_comm.send_data(19)
        self.last_movement_time = time.time()  # 모션이 끝났음을 알리기 위해 시간을 기록

        self.state = "CHECK_RED_POSITION"
 
    def process_frame(self, frame):
        result = frame.copy()

        if self.state == "FIND_CIRCLES":
            result, red_circle = self.find_circles(frame)
            if red_circle:
                self.state = "RED_DIST"
            return result
        
        elif self.state == "RED_DIST":
            result, red_circle = self.find_circles(frame)
            if red_circle:
                distance = self.calculate_distance(red_circle[1])
                if distance is not None:
                    walk_dist = self.calculate_walk_distance(distance)
                    print(f"Walk Distance: {walk_dist}")
                    self.state = "THRES_DIST"
            return result
        
        elif self.state == "THRES_DIST":
            result, red_circle = self.find_circles(frame)  # find_circles 호출 추가
            if red_circle:
                distance = self.calculate_distance(red_circle[1])
                if distance is not None:
                    walk_dist = self.calculate_walk_distance(distance)
                    if walk_dist > 1.5:
                        self.state = "GO_3STEP"
                    else:
                        self.state = "FAR_BALL_MOTION"
            return result
        
        elif self.state == "GO_3STEP":
            self.go_3step()
            return result
        
        elif self.state == "TRY_ALIGNMENT":
            self.try_alignment()
            return result

        elif self.state == "WAIT":
            if self.last_movement_time is not None and time.time() - self.last_movement_time > 2:
                self.state = "FIND_CIRCLES"
            return result
        
        elif self.state == "FAR_BALL_MOTION":
            self.far_ball_motion()
            return result

# 동영상 파일 처리
detector = ObjectDetector(
    lower_red=np.array([170, 90, 140]),
    upper_red=np.array([180, 255, 255]),
    
    lower_yellow=np.array([10, 70, 70]),
    upper_yellow=np.array([25, 255, 255]),
    tolerance=30,  # x 좌표 차이 허용 오차
    focal_length=765.6,  # 초점 거리--> focal_length_pixels= (sensor_width_mm * sensor_width_pixels) / focal_length_mm

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

import cv2
import numpy as np
import math
import threading
import time
#from HSVConfig import * 

class FieldProcessing:
    def __init__(self):
        # 초록색과 회색 영역 인식을 위한 HSV 범위 설정
        self.green_lower = np.array([35, 50, 50]) # 색깔 마스킹 값 사전 저장 
        self.green_upper = np.array([75, 255, 255])
        self.gray_lower = np.array([0, 0, 40])
        self.gray_upper = np.array([180, 50, 220])
        self.yellow_lower = np.array([20, 140, 100], np.uint8)
        self.yellow_upper = np.array([30, 255, 255], np.uint8)
    
    def process_green_field(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv, self.green_lower, self.green_upper)
        gray_mask = cv2.inRange(hsv, self.gray_lower, self.gray_upper)

        # 회색 영역을 초록색 마스크에서 제외
        combined_mask = cv2.bitwise_and(green_mask, green_mask, mask=~gray_mask)

        # 경기장의 초록색 영역의 경계 찾기
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
            cv2.drawContours(frame, [approx], 0, (0, 255, 0), 5)

        # 회색 영역(벙커)의 경계 찾기
        gray_contours, _ = cv2.findContours(gray_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in gray_contours:
            cv2.drawContours(frame, [cnt], -1, (0, 0, 255), 2)

        return frame

    def find_yellow_circle_center(self, frame):
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        yellow_mask = cv2.inRange(hsv_frame, self.yellow_lower, self.yellow_upper)
        contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                print(cX, cY)
                return (cX, cY)
        return None

class Vision:
    def __init__(self, W_View_size=600, H_View_size=400, hsv_Lower=0, hsv_Upper=0, KNOWN_WIDTH=4.5, FOV=62.2, h=30, alpha=45, h_min=[146], h_max=[179], s_min=[110], s_max=[255], v_min=[169], v_max=[255]):
        self.W_View_size = W_View_size
        self.H_View_size = H_View_size
        self.hsv_Lower = hsv_Lower
        self.hsv_Upper = hsv_Upper
        self.KNOWN_WIDTH = KNOWN_WIDTH
        self.FOV = FOV # 시야각
        self.h = h # 로봇 키
        self.alpha = alpha
        self.h_min = h_min
        self.h_max = h_max
        self.s_min = s_min
        self.s_max = s_max
        self.v_min = v_min
        self.v_max = v_max
        self.ball = False
        self.R = None
        self.rotation_angle = None
        self.x = 0
        self.y = 0
        self.flag_x = 0
        self.flag_y = 0
        self.set_hsv_values()
        self.field_processor = FieldProcessing()

        # OpenCV 비디오 캡처 설정
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.W_View_size)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.H_View_size)
        time.sleep(0.5)  # 카메라가 초기화될 시간을 기다림

        self.thread = threading.Thread(target=self.run)
        self.thread.start()
        self.frame = None
        self.lock = threading.Lock()

    def set_hsv_values(self):
        self.hsv_Lower = (self.h_min[0], self.s_min[0], self.v_min[0])
        self.hsv_Upper = (self.h_max[0], self.s_max[0], self.v_max[0])

    def set_alpha(self, alpha):
        with self.lock:
            self.alpha = alpha

    def get_alpha(self):
        with self.lock:
            return self.alpha

    def get_R(self):
        with self.lock:
            return self.R
            
    def get_ball(self):
        with self.lock:
            return self.ball

    def get_xy(self):
        with self.lock:
            return (self.x, self.y)

    def tilt_point_around_x_axis(self, point, tilt_angle_degrees):
        # 각도를 라디안으로 변환
        tilt_angle_radians = np.radians(tilt_angle_degrees)
        
        # x축을 중심으로 기울이는 회전 행렬
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(tilt_angle_radians), -np.sin(tilt_angle_radians)],
            [0, np.sin(tilt_angle_radians), np.cos(tilt_angle_radians)]
        ])
        
        # 주어진 점의 좌표를 확장하여 3D 벡터로 변환
        point_vector = np.array([point[0], point[1], point[2]])
        
        # 회전 행렬을 점에 적용하여 새 좌표 계산
        new_point = rotation_matrix.dot(point_vector)
        
        return new_point

    def calculate_angle_between_lines(self, ball_x, ball_y, flag_x, flag_y, center_x):
        vector_ball = np.array([ball_x - flag_x, ball_y - flag_y])
        vector_flag = np.array([center_x - ball_x, self.H_View_size - ball_y])

        # 두 벡터의 내적 계산
        dot_product = np.dot(vector_ball, vector_flag)

        # 두 벡터의 크기 계산
        magnitude_ball = np.linalg.norm(vector_ball)
        magnitude_flag = np.linalg.norm(vector_flag)

        # 코사인 각도 계산
        cos_angle = dot_product / (magnitude_ball * magnitude_flag)

        # 각도를 라디안에서 도(degree)로 변환
        angle = math.degrees(np.arccos(cos_angle))

        return angle

    def set_strike_angle(self): # 두 개의 사이각 계산, 골대 안잡히면 진행 방향에 있는 좌표로 깃발 좌표 대체 
        center_x = self.W_View_size / 2
        center_y = self.H_View_size / 2
        x, y = self.get_xy()
        try:
            angle = self.calculate_angle_between_lines(x, y, self.flag_x, self.flag_y, center_x)
        except:
            return None
        return angle

    def calculate_angle(self, v1, v2, theta):
        """
        v1, v2: 2차원 이미지 상의 두 벡터
        theta: z축을 중심으로 회전시키고 싶은 각도
        """
        # 2차원 벡터를 3차원 공간으로 변환
        v1_3d = np.array([v1[0], v1[1], 0])
        v2_3d = np.array([v2[0], v2[1], 0])
        
        # z축을 중심으로 회전 변환 행렬 생성
        rotation_matrix = np.array([
            [math.cos(math.radians(theta)), -math.sin(math.radians(theta)), 0],
            [math.sin(math.radians(theta)), math.cos(math.radians(theta)), 0],
            [0, 0, 1]
        ])
        
        # 회전 변환 행렬을 사용하여 3차원 공간에서의 두 벡터를 회전시켜 보정
        v1_corrected_3d = np.dot(rotation_matrix, v1_3d)
        v2_corrected_3d = np.dot(rotation_matrix, v2_3d)
        
        # 보정된 3차원 벡터를 2차원으로 투영
        v1_corrected_2d = v1_corrected_3d[:2]
        v2_corrected_2d = v2_corrected_3d[:2]
        
        # 두 벡터의 내적을 사용하여 코사인 값을 구함
        dot_product = np.dot(v1_corrected_2d, v2_corrected_2d)
        norm_v1 = np.linalg.norm(v1_corrected_2d)
        norm_v2 = np.linalg.norm(v2_corrected_2d)
        cos_theta = dot_product / (norm_v1 * norm_v2)
        
        # 코사인 값을 사용하여 두 벡터 사이의 각도를 구함
        angle = math.degrees(math.acos(cos_theta))
        
        return angle

    def get_rotation_angle(self):
        with self.lock:
            return self.rotation_angle

    def is_point_inside_boundary(self, x, y, boundary):
        """
        x, y: 점의 좌표
        boundary: 경계의 다각형 좌표 리스트
        """
        # 점과 경계 사이의 상대적 위치를 결정하기 위해 cv2.pointPolygonTest 함수 사용
        result = cv2.pointPolygonTest(np.array(boundary), (x, y), False)
        
        # 점이 경계 내부에 있으면 1, 경계 위에 있으면 0, 경계 외부에 있으면 -1 반환
        if result >= 0:
            return True
        else:
            return False

    def process_frame(self, frame):
        self.ball = False
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_frame, self.hsv_Lower, self.hsv_Upper)
        res = cv2.bitwise_and(frame, frame, mask=mask)

        # 컨투어 찾기
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            self.x = x + w / 2
            self.y = y + h / 2
            self.ball = True
            
            # 거리 계산 (픽셀에서 실제 거리로 변환)
            perceived_width = w
            self.R = self.KNOWN_WIDTH * self.W_View_size / (2 * perceived_width * math.tan(math.radians(self.FOV / 2)))
            
            # 중심점과 공 사이의 회전 각도 계산
            self.rotation_angle = self.calculate_angle((self.W_View_size/2, self.H_View_size), (self.x, self.y), self.alpha)
        
        # 경기장과 벙커 처리
        frame = self.field_processor.process_green_field(frame)
        
        # 노란색 원(깃발) 위치 찾기
        yellow_center = self.field_processor.find_yellow_circle_center(frame)
        if yellow_center is not None:
            self.flag_x, self.flag_y = yellow_center
        
        return frame

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            processed_frame = self.process_frame(frame)
            
            if self.R and self.rotation_angle is not None:
                cv2.putText(processed_frame, f"Distance: {self.R:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(processed_frame, f"Angle: {self.rotation_angle:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Frame", processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    vision = Vision()
    while True:
        R = vision.get_R()
        rotation_angle = vision.get_rotation_angle()
        print("[*] goal and ball angle :" + str(180 - vision.set_strike_angle()))

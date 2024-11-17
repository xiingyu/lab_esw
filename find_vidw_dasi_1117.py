### update :: 2024.11.16 규현오빠 코드에 내가 한 v7만 일부 추가할 것 ###

from Motion import SerialCommunication

import cv2
import numpy as np
import time
import threading

class MainAlgo() :
    def __init__(self) :
        self.red_lower1 = np.array([0, 30, 30])     # 대회 공 (첫 번째 빨간색 범위)
        self.red_upper1 = np.array([10, 240, 240])
        
        self.red_lower2 = np.array([165, 30,30])   # 대회 공 (두 번째 빨간색 범위)
        self.red_upper2 = np.array([180, 255, 255])
                
        self.yellow_lower = np.array([20, 120, 165]) # 깃발, 화살표
        self.yellow_upper = np.array([40, 255, 255])
        self.green_lower = np.array([59, 80, 60])
        self.green_upper = np.array([85, 240, 240])
        self.brown_lower = np.array([60, 0, 170])   # 벙커
        self.brown_upper = np.array([140, 35, 255])
        self.serial_comm = SerialCommunication() # 준렬 코드를 클래스로 변경하고 이걸 추가함
        self.last_movement_time = None  # 마지막으로 모션이 수행된 시간을 저장
        
        ### object flags ###
        self.red_ball_existance = 0 
            # 1 2 3
            # 4 5 6 
        self.red_ball_detection = False
        
        ####################

        self.position_reached = False  # Center 도달 여부를 확인하기 위한 플래그
        self.position_started = None  # Left 또는 Right 명령 시작 방향을 저장
        
        #### states and flags ####
        self.robot_moving = False
        self.head_v_align = 16100
        self.head_h_align = 11100
            ## 1 2 3 4 5
            ##   6 7 8
        self.high_view_flag = False ## 90도로 봤을때 없었으면 True
        self.head_h_cnt = 0
        self.main_state = "find_ball"
        self.ready_to_putt = 0 ## 0, 1
        ##########################
                
        ### const parameters ###
        self.img_size_x = 640
        self.img_size_y = 480
        self.frame_rate = 10
        self.pixel_size = 0.00000112 #meters
        self.ball_ROI = ([int(self.img_size_x * 0.59), int(self.img_size_y * 0.62)],[int(self.img_size_x * 0.63), int(self.img_size_y * 0.68)])
        
        self.focal_length=0.00304#meters
        
        self.center_x = int(self.img_size_x/2)
        self.center_y = int(self.img_size_y/2)
        
        self.robot_height = 0.45 #meters
        ########################
                
        ### params declare ####
        self.img = np.zeros((self.img_size_y, self.img_size_x, 3), dtype=np.uint8)
        self.field_img = np.zeros((self.img_size_y, self.img_size_x, 3), dtype=np.uint8)
        self.green_filterd = np.zeros_like(self.img)
        self.red_filterd = np.zeros_like(self.img)
        self.yellow_filterd = np.zeros_like(self.img)
        
        self.running = True  # 스레드 실행을 제어하는 플래그
        self.red_circle=((0,0),0)
        self.yellow_circle=(0.,0.)
        self.head_v_align = 16100
        self.head_h_align = 11100
                
        ########################## 
                
        self.serial_comm.send_data(11100)
        self.serial_comm.send_data(16100)
          
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, self.img_size_x)
        self.cap.set(4, self.img_size_y)
        self.cap.set(5, self.frame_rate)
        self.thread = threading.Thread(target=self.update_frame, daemon=True)
        self.thread.start()  # 스레드 시작
        
    def update_frame(self) :
        while self.running:
            ret, self.img = self.cap.read()  # 프레임을 계속해서 업데이트
            if not ret:
                print("Cannot find camera")
                break
            
            blurred = cv2.GaussianBlur(self.img, (5, 5), 1)
            self.hsv_img = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
            green_mask = cv2.inRange(self.hsv_img, self.green_lower, self.green_upper)
            ret_green, green_mask_thresh = cv2.threshold(green_mask, 100, 255, cv2.THRESH_BINARY)
            
            if ret_green :
                contours, _ = cv2.findContours(green_mask_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                max_area = 0
                max_contour = None
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > max_area:
                        max_area = area
                        max_contour = contour

                if max_contour is not None:
                    max_contour_mask = np.zeros_like(green_mask_thresh)
                    hull = cv2.convexHull(max_contour)
                    cv2.drawContours(max_contour_mask, [hull], -1, 255, -1)
                    cv2.drawContours(max_contour_mask, [max_contour], -1, (255, 255, 255), thickness=cv2.FILLED)
                    
                    self.green_filterd = cv2.bitwise_and(self.img, self.img, mask=max_contour_mask)
                else :
                    self.green_filterd = np.zeros_like(self.img)
                    
            else :
                self.green_filterd = np.zeros_like(self.img)
                
            if (self.main_state != "ball") :
                self.hsv_img = cv2.cvtColor(self.green_filterd, cv2.COLOR_BGR2HSV)
                red_mask1 = cv2.inRange(self.hsv_img, self.red_lower1, self.red_upper1)
                red_mask2 = cv2.inRange(self.hsv_img, self.red_lower2, self.red_upper2)
                
                red_mask = cv2.bitwise_or(red_mask1,red_mask2)
                
                ret_red, red_mask_thresh = cv2.threshold(red_mask, 120, 255, cv2.THRESH_BINARY)
                
                if ret_red :
                    contours, _ = cv2.findContours(red_mask_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    max_area = 0
                    max_contour = None
                    for contour in contours:
                        area = cv2.contourArea(contour)
                        if area > max_area:
                            max_area = area
                            max_contour = contour

                    if max_contour is not None:
                        max_contour_mask = np.zeros_like(red_mask_thresh)
                        cv2.drawContours(max_contour_mask, [max_contour], -1, (255, 255, 255), thickness=cv2.FILLED)
                           
                        self.red_filterd = cv2.bitwise_and(self.img, self.img, mask=max_contour_mask)
                    
                        (x_red, y_red), radius_red = cv2.minEnclosingCircle(max_contour)
                        center_red = (int(x_red), int(y_red))
                        radius_red = int(radius_red)
                        self.red_circle = (center_red, radius_red)
                        cv2.circle(self.red_filterd, center_red, radius_red, (0, 255, 0), 2)  # 둘러싸는 원 표시
                        self.red_ball_detection = True
                        if max_area < 10 :
                            self.red_ball_existance = False
                            self.red_ball_detection = False
                    else :
                        self.red_ball_detection = False
                        self.red_filterd = np.zeros_like(self.img)
                        self.red_circle = ((0, 0),0)
                        #print(f"ball state talker : {self.red_circle}")
                else :
                    self.red_filterd = np.zeros_like(self.img)
            else : 
                self.red_ball_detection = False

### 여기에 추가한 내용 ###
    def check_position(self):
        """ 빨간 공이 화면의 좌/우/중앙 어디에 있는지 확인 """
		# 공의 x좌표 기준으로 판단(왼쪽 30%에 있는지)
        if self.red_circle[0][0] < int(self.img_size_x * 0.2) : 
            return "Left"
            # 공의 x좌표 기준으로 판단(오른쪽 30%에 있는지)
        elif self.red_circle[0][0] > int(self.img_size_x * 0.8) :
            return "Right"
        else :
            return "Center"

    def process_frame(self):
        """ 공의 위치를 조정하기 위한 프로세스 """
        if not self.red_ball_detection:
            print("No ball detected in current direction")
            #self.position_reached = False  # 탐지 실패 시 초기화
            self.main_state = "find_ball"  # 공을 찾도록 상태 변경
            return  # 탐지되지 않은 경우 바로 반환

        position = self.check_position()

        if position == "Left":
            if self.position_started != "Left":
                self.position_started = "Left"
                print(f"Ball detected at Left. Starting adjustments. Current command: {self.head_h_align}")

            # 왼쪽에서 감지된 경우 몸체를 왼쪽으로 10도씩 조정
            self.head_h_align -= 10
            self.serial_comm.send_data(self.head_h_align)
            self.delay(0.8)
            print(self.red_ball_detection)
            self.head_h_cnt += 1 # 왼쪽으로 움직인 횟수 증가

        elif position == "Right":
            if self.position_started != "Right":
                self.position_started = "Right"
                print(f"Ball detected at Right. Starting adjustments. Current command: {self.head_h_align}")

            # 오른쪽에서 감지된 경우 몸체를 오른쪽으로 10도씩 조정
            self.head_h_align += 10
            self.serial_comm.send_data(self.head_h_align)
            self.delay(0.8)
            print(self.red_ball_detection)
            self.head_h_cnt += 1  # 오른쪽으로 움직인 횟수 증가

        elif position == "Center":
            print("Ball detected at Center. Holding position.")
            self.position_reached = True

            
        # 고개를 정면으로 돌리기
        self.serial_comm.send_data(11100)
        self.head_h_align = 11100
        self.delay(0.8)
        
        # 고개를 돌린 횟수만큼 몸체를 반대 방향으로 회전
        print(f"Rotating body to realign with head adjustments: {abs(self.head_h_cnt)} steps.")
        for _ in range(abs(self.head_h_cnt)):
            if self.position_started == "Left":
                self.serial_comm.send_data(4)  # 왼쪽 턴
            elif self.position_started == "Right":
                self.serial_comm.send_data(6)  # 오른쪽 턴
            self.delay(0.8)
                
        self.head_h_cnt = 0 # 머리 조정 횟수 초기화
        self.position_started = None  # 방향 정보 초기화

        self.main_state = "adjust_position"  # 위치 조정 상태로 복귀

 ######################################################################

    def show_frame(self):
        cv2.imshow("img", self.img)
        # cv2.imshow("green", self.green_filterd)
        # cv2.imshow("red", self.red_filterd)
        cv2.waitKey(1)  
        
    def delay(self, sec) :
        start = time.time()
        
        while(True) :
            current = time.time()
            if current - start > sec :
                break
            else :
                time.sleep(1/self.frame_rate)
                self.show_frame()
                
        return 1
                
    def finite_statemachine(self) :
        if self.main_state == "find_ball" :
            self.position_reached = False  # 초기화            
            if self.high_view_flag == False :
                ##view foward
                self.delay(0.1)
                self.serial_comm.send_data(1670)
                self.head_v_align = 1670
                self.delay(0.8)
                self.serial_comm.send_data(11100)
                self.head_h_align = 11100
                self.delay(0.1)
                
                self.delay(1)
                print(self.red_ball_detection)

	            # 11100(정면)일 때 공이 인식되었나 확인
                if self.red_ball_detection == True :
                    self.check_position()
                    print(f"Detected ball position: {self.head_h_align}")
                    self.process_frame()
                    return  # 공 탐지 후 실행 종료

                ##제일 왼쪽 끝으로 목 돌리기(정면에서 인식되지 않았을 때)
                self.serial_comm.send_data(1110)
                self.head_h_align = 1110
                self.delay(2)
                if self.red_ball_detection :
                    self.check_position()
                    print(f"Detected ball position: {self.head_h_align}")
                    self.process_frame()
                    return  # 공 탐지 후 실행 종료

                ##55
                self.serial_comm.send_data(1155)
                self.head_h_align = 1155
                self.delay(2)
                if self.red_ball_detection == True :
                    self.check_position()
                    print(f"Detected ball position: {self.head_h_align}")
                    self.process_frame()
                    return  # 공 탐지 후 실행 종료

                ##145
                self.serial_comm.send_data(11145)
                self.head_h_align = 11145
                self.delay(2)
                if self.red_ball_detection == True :
                    self.check_position()
                    print(f"Detected ball position: {self.head_h_align}")
                    self.process_frame()
                    return  # 공 탐지 후 실행 종료
                
                ##right
                self.serial_comm.send_data(11180)
                self.delay(0.8)
                self.serial_comm.send_data(40) # 골프채 내리기
                self.delay(2)
                print(f"골프채내리고 공 확인하는지: {self.red_ball_detection}")
                
                if self.red_ball_detection == True :
                    self.check_position()
                    print(f"Detected ball position: {self.head_h_align}")
                    self.process_frame()
                    return  # 공 탐지 후 실행 종료
                self.delay(5)
                self.serial_comm.send_data(41) # 골프채 올리기
                self.head_h_align = 11180
                self.delay(2)

                self.high_view_flag = True

            else :
                self.delay(3)
                self.serial_comm.send_data(1640)
                self.head_v_align = 1640
                self.delay(1)
                self.serial_comm.send_data(11100)
                self.head_h_align = 11100
                self.delay(1)
                if self.red_ball_detection == True :
                    self.check_position()
                    print(f"Detected ball position: {self.head_h_align}")
                    self.process_frame()
                    return  # 공 탐지 후 실행 종료
                       
                self.serial_comm.send_data(1155)
                self.head_h_align = 1155
                self.delay(2)
                if self.red_ball_detection == True :
                    self.check_position()
                    print(f"Detected ball position: {self.head_h_align}")
                    self.process_frame()
                    return  # 공 탐지 후 실행 종료

                
                self.serial_comm.send_data(11145)
                self.head_h_align = 11145
                self.delay(2)
                if self.red_ball_detection == True :
                    self.check_position()
                    print(f"Detected ball position: {self.head_h_align}")
                    self.process_frame()
                    return  # 공 탐지 후 실행 종료
                
        elif self.main_state == "adjust_position":
            self.process_frame()  # 공 위치 조정 수행

            print("Rechecking ball position after body alignment...")

            # 공이 여전히 탐지되었는지 확인
            if not self.red_ball_detection:
                print("Ball lost during adjustment. Switching to find_ball state.")
                self.main_state = "find_ball"
                self.position_reached = False  # 초기화
                return
                
            
            # 정면 상태에서 다시 Center 여부 확인
            new_position = self.check_position()
            if new_position == "Center":
                print("Ball is still Center after body rotation.")
                self.main_state = "goto_ball"  # 공으로 이동 상태로 전환
                return  # goto_ball로 전환 후 함수 종료
            else:
                print(f"Ball is no longer Center (new position: {new_position}). Re-adjusting.")
                self.position_reached = False
                self.main_state = "adjust_position"  # 위치 조정 상태로 복귀
                return  # adjust_position으로 전환 후 함수 종료

        elif self.main_state == "goto_ball" :
            if self.red_circle[0][1] < int(self.img_size_y * 0.8) :
                self.serial_comm.send_data(18)
                self.delay(1)
            elif self.red_ball_detection == False :
                self.serial_comm.send_data(1660)
                self.head_v_align = 1660
                time.sleep(0.1)
            else :
                self.delay(1)
                self.serial_comm.send_data(1660)
                self.head_v_align = 1660
                self.delay(1)
                self.main_state= "goto_ball2"
                print("set goto_ball2")
                
        elif self.main_state == "goto_ball2" :
            self.delay(1)
            if self.red_circle[0][1] < int(self.img_size_y * 0.6) :
                self.serial_comm.send_data(18)
                self.delay(1)
            else :
                self.main_state = "position_ball"
                self.delay(1)
                print("set position_ball")

if __name__ == "__main__" :
    try :
        main = MainAlgo()
        while True :
            main.show_frame()
            main.finite_statemachine()
        
    except Exception as e :
        print(f"raised errors {e} ")

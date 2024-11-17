## update :: 2024.11.15 3도로 햇을때 느린 것 대비한 방법

import cv2
import numpy as np
import time
import threading
from Motion import SerialCommunication

class MainAlgo:
    def __init__(self):
        # 빨간색 공 탐지를 위한 HSV 범위
        self.red_lower = np.array([165, 50, 50])
        self.red_upper = np.array([180, 255, 255])

        self.serial_comm = SerialCommunication()  # 시리얼 통신 초기화
        self.red_ball_detection = False
        self.first_frame = True  # 첫 프레임인지 확인하기 위한 플래그
        self.position_reached = False  # Center 도달 여부를 확인하기 위한 플래그
        self.position_started = None  # Left 또는 Right 명령 시작 방향을 저장
        
        self.frame_rate = 10
        self.current_command = 11100  # 초기 명령

        self.main_state = "find_ball"  # main_state 초기화

        ### 이미지 설정 ###
        self.img_size_x = 640
        self.img_size_y = 480
        self.img = np.zeros((self.img_size_y, self.img_size_x, 3), dtype=np.uint8)
        self.red_filterd = np.zeros_like(self.img)
        self.red_circle = (0, 0)
        self.running = True  # 스레드 실행을 제어하는 플래그

        ### 카메라 설정 ###
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, self.img_size_x)
        self.cap.set(4, self.img_size_y)
        self.cap.set(5, self.frame_rate)
        self.thread = threading.Thread(target=self.update_frame, daemon=True)
        self.thread.start()  # 스레드 시작


    def update_frame(self):
        """ 프레임을 읽고 빨간색 공을 탐지하여 표시 """
        ret, self.img = self.cap.read()  # 프레임을 읽기
        if not ret:
            print("Cannot find camera")
            return

        # 이미지 블러 처리 후 HSV 색 공간으로 변환
        blurred = cv2.GaussianBlur(self.img, (5, 5), 1)
        hsv_img = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        red_mask = cv2.inRange(hsv_img, self.red_lower, self.red_upper)
        ret_red, red_mask_thresh = cv2.threshold(red_mask, 150, 255, cv2.THRESH_BINARY)

        if ret_red:
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

                # 빨간색 공의 중심 좌표와 반지름 계산
                (x_red, y_red), radius_red = cv2.minEnclosingCircle(max_contour)
                center_red = (int(x_red), int(y_red))
                radius_red = int(radius_red)
                self.red_circle = (center_red, radius_red)
                cv2.circle(self.red_filterd, center_red, radius_red, (0, 255, 0), 2)
                self.red_ball_detection = True
            else:
                self.red_ball_detection = False
                self.red_filterd = np.zeros_like(self.img)
                self.red_circle = (0, 0)

    def check_position(self, red_circle, frame_width):
        """ 빨간 공이 화면의 좌/우/중앙 어디에 있는지 확인 """
        center_red_x = red_circle[0][0]
        frame_center_x = frame_width // 2
        tolerance = 30

        if center_red_x < frame_center_x - tolerance:
            return "Left"
        elif center_red_x > frame_center_x + tolerance:
            return "Right"
        else:
            return "Center"

    def process_frame(self):
        """ 공의 위치를 조정하기 위한 프로세스 """
        if not self.red_ball_detection:
            print("No ball detected in current direction")
            return  # 탐지되지 않은 경우 바로 반환

        position = self.check_position(self.red_circle, self.img_size_x)

        if position == "Left":
            if self.position_started != "Left":
                self.position_started = "Left"
                print(f"Ball detected at Left. Starting adjustments. Current command: {self.current_command}")

            # 왼쪽에서 감지된 경우 몸체를 왼쪽으로 5도씩 조정
            self.current_command -= 5
            self.serial_comm.send_data(self.current_command)
            self.delay(0.8)

        elif position == "Right":
            if self.position_started != "Right":
                self.position_started = "Right"
                print(f"Ball detected at Right. Starting adjustments. Current command: {self.current_command}")

            # 오른쪽에서 감지된 경우 몸체를 오른쪽으로 5도씩 조정
            self.current_command += 5
            self.serial_comm.send_data(self.current_command)
            self.delay(0.8)

        elif position == "Center":
            print("Ball detected at Center. Holding position.")
            self.position_reached = True

    def show_frame(self):
        cv2.imshow("img", self.img)
        cv2.waitKey(1)
        
    def delay(self, sec):
        """ 지정된 시간 동안 대기하며 화면 표시 """
        start = time.time()
        while True:
            current = time.time()
            if current - start > sec:
                break
            else:
                time.sleep(1 / self.frame_rate)
                self.show_frame()
        return 1
        
    def finite_statemachine(self):
        """ 상태 머신을 사용하여 공 탐지 동작 수행 """
        if self.main_state == "find_ball":
            # 첫 프레임 처리
            if self.first_frame:
                # 정면으로 초기화
                self.serial_comm.send_data(1660)  # 목 높이 초기화
                self.delay(0.1)
                self.serial_comm.send_data(11100)  # 정면(11100)으로 초기화
                self.delay(0.1)
                self.update_frame()  # 첫 프레임 탐지 상태 갱신
                self.delay(2)
                self.first_frame = False  # 첫 프레임 처리 완료
                self.red_ball_detection = False  # 탐지 상태 초기화
                self.position_started = None  # 탐지 위치 초기화
                print("First frame initialized to 11100 (front)")
                return  # 첫 프레임 처리 후 함수 종료

            # 공 탐지 여부에 따라 동작
            if self.red_ball_detection:
                position = self.check_position(self.red_circle, self.img_size_x)

                # 감지된 경우 방향에 따라 조정
                if position == "Center" and self.position_started:
                    self.position_reached = True
                    print(f"{self.position_started.lower()}2center")
                elif position == "Left" and not self.position_reached:
                    self.process_frame()
                elif position == "Right" and not self.position_reached:
                    self.process_frame()
            else:
                # 이전 명령에 따라 다음 명령 설정
                commands = {
                    11100: 1110,    # 정면 → 왼쪽 90도
                    1110: 1155,     # 왼쪽 90도 → 왼쪽 45도
                    1155: 11190,    # 왼쪽 45도 → 오른쪽 90도
                    11190: 11145,   # 오른쪽 90도 → 오른쪽 45도
                    11145: 11100,   # 오른쪽 45도 → 정면
                }

                # 다음 명령어 결정
                next_command = commands.get(self.current_command, 11100)  # 기본값은 정면
                print(f"No ball detected at {self.current_command}. Rotating to {next_command}")
                self.current_command = next_command

                # 명령 전송
                self.serial_comm.send_data(self.current_command)
                self.delay(0.8)

                # 프레임 업데이트 및 공 탐지 여부 확인
                self.update_frame()
                self.delay(0.8)
                if self.red_ball_detection:
                    print(f"Ball detected after rotating to {self.current_command}!")
                    return


if __name__ == "__main__":
    main = MainAlgo()
    try:
        while True:
            #main.update_frame()
            main.show_frame()
            main.finite_statemachine()
    except Exception as e:
        print(f"raised errors {e}")
    finally:
        main.cap.release()
        cv2.destroyAllWindows()

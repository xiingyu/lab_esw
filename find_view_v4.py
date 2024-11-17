import cv2
import numpy as np
import time
from Motion import SerialCommunication

class MainAlgo:
    def __init__(self):
        # 빨간색 공 탐지를 위한 HSV 범위
        self.red_lower = np.array([165, 50, 50])
        self.red_upper = np.array([180, 255, 255])

        self.serial_comm = SerialCommunication()  # 시리얼 통신 초기화
        self.first_frame = True  # 첫 프레임인지 확인하기 위한 플래그
        self.position_reached = False  # Center 도달 여부를 확인하기 위한 플래그
        self.position_started = None  # Left 또는 Right 명령 시작 방향을 저장
        self.red_ball_detection = False
        self.frame_rate = 10
        self.current_command = 1190  # 초기 명령

        self.main_state = "find_ball"  # main_state 초기화

        ### 이미지 설정 ###
        self.img_size_x = 640
        self.img_size_y = 480
        self.img = np.zeros((self.img_size_y, self.img_size_x, 3), dtype=np.uint8)
        self.red_filterd = np.zeros_like(self.img)
        self.red_circle = (0, 0)

        ### 카메라 설정 ###
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, self.img_size_x)
        self.cap.set(4, self.img_size_y)
        self.cap.set(5, self.frame_rate)

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
        tolerance = 50

        if center_red_x < frame_center_x - tolerance:
            return "Left"
        elif center_red_x > frame_center_x + tolerance:
            return "Right"
        else:
            return "Center"

    def process_frame(self):
        """ 공의 위치를 조정하기 위한 프로세스 """
        if not self.red_ball_detection:
            return  # 공이 없으면 바로 리턴

        position = self.check_position(self.red_circle, self.img_size_x)

        if self.initial_commands_complete and not self.position_reached:
            if position == "Left":
                if self.position_started != "Left":
                    self.position_started = "Left"
                    self.current_command = 1190  # 초기 명령 값 설정

                # Left 상태에서 명령 전송
                self.serial_comm.send_data(self.current_command)
                print(f"Sent command: {self.current_command} to adjust leftward")

                # 프레임 업데이트 및 위치 재확인
                self.update_frame()
                position = self.check_position(self.red_circle, self.img_size_x)
                print(f"Current position after adjustment: {position}")

                # 위치에 따라 추가 명령 전송
                if position == "Left":
                    self.current_command -= 10
                    return  # 다음 명령을 위해 루프를 유지

                elif position == "Center":
                    print("Set :: left 2 center")
                    self.position_reached = True
                    return  # 목표에 도달하면 종료

            elif position == "Right":
                # Right 상태에서 프로세스 추가 필요 시 여기에 추가
                pass

    def show_frame(self):
        """ 현재 프레임과 빨간색 필터링된 이미지를 화면에 표시 """
        cv2.imshow("img", self.img)
        cv2.waitKey(1)

    def finite_statemachine(self):
        """ 상태 머신을 사용하여 공 탐지 동작 수행 """
        if self.main_state == "find_ball":
            if self.first_frame:
                self.serial_comm.send_data(1670)
                time.sleep(0.1)
                self.serial_comm.send_data(11100)
                time.sleep(0.1)
                self.first_frame = False
                self.initial_commands_complete = True

            # 공 탐지 후 위치 확인
            if self.red_ball_detection:
                position = self.check_position(self.red_circle, self.img_size_x)

                # Center 여부 확인 및 Left/Right 명령 전송
                if self.initial_commands_complete and not self.position_reached:
                    if position == "Center" and self.position_started:
                        self.position_reached = True
                        print(f"{self.position_started.lower()}2center")
                    elif position == "Left" and not self.position_reached:
                        self.process_frame()
                    elif position == "Right" and not self.position_reached:
                        self.process_frame()

if __name__ == "__main__":
    main = MainAlgo()
    try:
        while True:
            main.update_frame()
            main.show_frame()
            main.finite_statemachine()
    except Exception as e:
        print(f"raised errors {e}")
    finally:
        # "q" 누르거나 예외 발생 시 자원 해제
        main.cap.release()
        cv2.destroyAllWindows()

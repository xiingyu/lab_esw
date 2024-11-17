import cv2
import numpy as np
import time
from Motion import SerialCommunication

class ObjectDetector:
    def __init__(self, lower_red, upper_red, focal_length=500):
        self.lower_red = lower_red
        self.upper_red = upper_red
        self.serial_comm = SerialCommunication()  # Serial communication initialization
        self.first_frame = True
        self.position_reached = False
        self.initial_commands_complete = False
        self.left_commands = [1180, 1170, 1160, 1150, 1140, 1130, 1120]
        self.right_commands = [11110, 11120, 11130, 11140, 11150, 11160]

    def find_red_circle(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        red_mask = cv2.inRange(hsv, self.lower_red, self.upper_red)
        result = frame.copy()
        red_circle = None

        contours_red, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours_red:
            largest_contour_red = max(contours_red, key=cv2.contourArea)
            (x_red, y_red), radius_red = cv2.minEnclosingCircle(largest_contour_red)
            center_red = (int(x_red), int(y_red))
            radius_red = int(radius_red)
            red_circle = (center_red, radius_red)
            cv2.circle(result, center_red, radius_red, (0, 255, 0), 2)

        return result, red_circle

    def check_position(self, red_circle, frame_width):
        center_red_x = red_circle[0][0]
        frame_center_x = frame_width // 2
        tolerance = 100

        if center_red_x < frame_center_x - tolerance:
            return "Left"
        elif center_red_x > frame_center_x + tolerance:
            return "Right"
        else:
            return "Center"

    def process_frame(self, frame, cap):
        if self.first_frame:
            self.serial_comm.send_data(1670)
            time.sleep(1)
            self.serial_comm.send_data(11100)
            time.sleep(1)
            self.first_frame = False
            self.initial_commands_complete = True

        result, red_circle = self.find_red_circle(frame)
        
        if red_circle:
            position = self.check_position(red_circle, frame.shape[1])

            if self.initial_commands_complete and not self.position_reached:
                # Adjust leftward
                if position == "Left":
                    current_angle = 100
                    while position == "Left" and current_angle < 190:
                        current_angle -= 10
                        self.serial_comm.send_data(11000 + current_angle)
                        print(f"Adjusted neck angle to right: {11000 + current_angle}")
                        time.sleep(0.2)

                        # Capture new frame, display, and re-check position
                        ret, frame = cap.read()
                        if not ret:
                            break
                        result, red_circle_check = self.find_red_circle(frame)
                        cv2.imshow("Processed Frame", result)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                        
                        if red_circle_check:
                            position = self.check_position(red_circle_check, frame.shape[1])
                            if position == "Center":
                                self.position_reached = True
                                print("left2center")
                                break

                # Adjust rightward
                elif position == "Right":
                    current_angle = 100
                    while position == "Right" and current_angle > 10:
                        current_angle += 10
                        self.serial_comm.send_data(11000 + current_angle)
                        print(f"Adjusted neck angle to left: {11000 + current_angle}")
                        time.sleep(0.2)

                        # Capture new frame, display, and re-check position
                        ret, frame = cap.read()
                        if not ret:
                            break
                        result, red_circle_check = self.find_red_circle(frame)
                        cv2.imshow("Processed Frame", result)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

                        if red_circle_check:
                            position = self.check_position(red_circle_check, frame.shape[1])
                            if position == "Center":
                                self.position_reached = True
                                print("right2center")
                                break

        return result

# Video capture and processing loop
detector = ObjectDetector(
    lower_red=np.array([170, 90, 140]),
    upper_red=np.array([180, 255, 255]),
    focal_length=765.6
)

cap = cv2.VideoCapture(0)
cap.set(3, 1024)
cap.set(4, 768)
cap.set(5, 15)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    processed_frame = detector.process_frame(frame, cap)
    cv2.imshow("Processed Frame", processed_frame)

    if cv2.waitKey(33) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

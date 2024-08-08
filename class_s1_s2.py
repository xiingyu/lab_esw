import cv2
from s1_blue_position import DirectionDetector
from s2_ball_detection import RedBallDetector

class CombinedDetector:
    def __init__(self, camera_index=1):
        self.camera = cv2.VideoCapture(camera_index)
        if not self.camera.isOpened():
            raise ValueError(f"Camera at index {camera_index} could not be opened.")
        self.direction_detector = DirectionDetector(self.camera)
        self.ball_detector = RedBallDetector(camera_index)

    def run(self):
        while True:
            image = self.direction_detector.get_image()
            if image is None:
                break

            # 방향 및 점 찾기
            blue_mask = self.direction_detector.preprocess_image(image, self.direction_detector.blue_lower, self.direction_detector.blue_upper)
            blue_dots, blue_contours = self.direction_detector.find_dots(blue_mask)
            
            red_mask = self.direction_detector.preprocess_image(image, self.direction_detector.red_lower, self.direction_detector.red_upper)
            red_dots, red_contours = self.direction_detector.find_dots(red_mask)
            red_dot = red_dots[0] if red_dots else None

            direction = self.direction_detector.determine_direction(blue_dots, red_dot)

            if direction == "1번 방향" and red_dot:
                red_position = self.direction_detector.find_red_position_in_1st_direction(blue_dots, red_dot)
                print(f"1번 방향, 빨간 점 위치: {red_position}")
            elif direction == "2번 방향" and red_dot:
                red_position = self.direction_detector.find_red_position_in_2nd_direction(blue_dots, red_dot)
                print(f"2번 방향, 빨간 점 위치: {red_position}")
            else:
                print(f"Robot is facing the {direction} of the field.")
            
            self.direction_detector.draw_dots(image, blue_dots, red_dot, blue_contours, red_contours)
            
            # 공 탐지 및 거리 계산
            ball_frame, distance, center, pixel_area, walk_dist, case_number = self.ball_detector.detect_and_calculate_distance(image)
            if distance:
                cv2.putText(ball_frame, f"Distance: {distance:.2f} m", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(ball_frame, f"Area: {pixel_area:.2f} pixels^2", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(ball_frame, f"Walk Dist: {walk_dist:.2f} m", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(ball_frame, f"Case: {case_number}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Direction", image)
            cv2.imshow("Ball Detection", ball_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.camera.release()
        cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    combined_detector = CombinedDetector(camera_index=1)
    combined_detector.run()

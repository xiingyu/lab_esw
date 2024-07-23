import cv2
import numpy as np

# 카메라 내부 매개변수 설정 (실제 매개변수는 해당 카메라의 매뉴얼에 따라 설정)
focal_length = 1000  # 초점 거리
camera_matrix = np.array([[focal_length, 0, 320],
                          [0, focal_length, 240],#첫 번째 요소는 y 축에 대한 초점 거리 0으로 설정 두 번째 요소 x축 초점 거리
                          [0, 0, 1]], dtype=np.float64)

def detect_yellow_arrow(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_yellow = np.array([20, 100, 150])
    upper_yellow = np.array([30, 255, 255])

    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # 노란색 화살표를 제외한 부분은 모두 검정색으로 처리
    frame[mask != 255] = [0, 0, 0]

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)

        # 적절한 epsilon 값을 설정하여 7개의 점으로 근사
        epsilon = 0.03 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)

        if len(approx) == 7:
            arrow_points = approx.reshape(7, 2)

            # 화살표 중심 계산
            center = np.mean(arrow_points, axis=0).astype(int)

            # 화살표 방향 벡터 계산
            v1 = arrow_points[1] - arrow_points[0]
            v2 = arrow_points[6] - arrow_points[0]
            direction_vector = (v1 + v2) / 2

            # 방향 벡터를 반대로 변경
            direction_vector = -direction_vector
            
            print(direction_vector)
            # 카메라 기준 방향 벡터 계산
            #1. 방향 벡터 카메라 좌표계로 변환하기 : [x, y, 1] 형태
            direction_camera_space = np.array([direction_vector[0], direction_vector[1], 1], dtype=np.float64)
            #2. 카메라 매트릭스의 역행렬 이용하여 카메라 좌표계에서 세계 좌표계로 방향 벡터 변환
            direction_world_space = np.dot(np.linalg.inv(camera_matrix), direction_camera_space)#카메라 matrix랑 내적
            #3. 방향벡터 정규화(단위벡터로 만들기)
            direction_world_space /= np.linalg.norm(direction_world_space)


            # 방향 벡터 각도 계산 (카메라 기준)
            angle_camera_frame_rad = np.arctan2(direction_world_space[1], direction_world_space[0])

            angle_camera_frame_deg = angle_camera_frame_rad * 180/ np.pi
            #print(angle_camera_frame_rad)

            # 각도 보정하기..

            angle_correction = 0  # 보정값
            ANGLE_F = -(angle_camera_frame_deg - angle_correction)-250

            # 수정된 각도 표시

            cv2.putText(frame, f"Angle (camera frame): {ANGLE_F:.2f} degrees", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            #cv2.putText(frame, f"Angle (camera frame): {angle_camera_frame_deg:.2f} degrees", (50, 50),
            #            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            



         # 각도에 따라 방향 결정
            if ANGLE_F > 0:
                cv2.putText(frame, "LEFT", (50, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            else:
                cv2.putText(frame, "RIGHT", (50, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                               

            # 카메라 방향 파란색 화살표로 표시
            #cv2.arrowedLine(frame, (320, 240), (320 + int(direction_world_space[0] * 50), 240 + int(direction_world_space[1] * 50)), (255, 0, 0), 2)

            # direction vector를 빨간색 화살표로 표시
            cv2.arrowedLine(frame, center, (center[0] + int(direction_vector[0]), center[1] + int(direction_vector[1])), (0, 0, 255), 2)

            # 프레임에 화살표 점과 각도를 그리기
            for point in arrow_points:
                cv2.circle(frame, tuple(point), 5, (0, 255, 255), -1)
            cv2.putText(frame, f"Angle (camera frame): {ANGLE_F:.2f} degrees", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    return frame


cap = cv2.VideoCapture(4)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_with_arrow = detect_yellow_arrow(frame)

    cv2.imshow('Yellow Arrow Detection', frame_with_arrow)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

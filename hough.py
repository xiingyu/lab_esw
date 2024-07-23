import cv2
import numpy as np

def detect_yellow_arrow(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # HSV 설정
    lower_yellow = np.array([20, 100, 100])  
    upper_yellow = np.array([30, 255, 255]) 

    # HSV 이미지에서 노란색 영역만을 추출하기 위한 마스크 생성
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # 노란색 영역 외의 나머지 부분을 모두 검정색으로 만들기
    mask_inv = cv2.bitwise_not(mask)
    frame[mask_inv > 0] = [0, 0, 0]  # 검정색으로 처리

    # 마스크된 이미지에 캐니 엣지 검출 적용
    edges = cv2.Canny(mask, 50, 150, apertureSize=3)# sobel kernel 3으로 설정

    # 이미지에서 허프 변환 사용하여 선 찾기
    lines = cv2.HoughLines(edges, 1, np.pi/180, 20)

    # 선이 검출되지 않았을 경우
    if lines is None:
        return frame, "No arrow detected"

    # 화살표 방향 계산을 위한 배열 초기화
    left = [0, 0]
    right = [0, 0]
    up = [0, 0]
    down = [0, 0]

    # 허프 변환으로 검출된 선들 순회
    for line in lines:
        rho, theta = line[0]

        # 왼쪽/오른쪽 화살표를 판별하는 조건들
        if ((np.round(theta, 2)) >= 1.0 and (np.round(theta, 2)) <= 1.1) or ((np.round(theta, 2)) >= 2.0 and (np.round(theta, 2)) <= 2.1):
            if (rho >= 20 and rho <= 30):
                left[0] += 1
            elif (rho >= 60 and rho <= 65):
                left[1] += 1
            elif (rho >= -73 and rho <= -57):
                right[0] += 1
            elif (rho >= 148 and rho <= 176):
                right[1] += 1

        # 위쪽/아래쪽 화살표를 판별하는 조건들
        elif ((np.round(theta, 2)) >= 0.4 and (np.round(theta, 2)) <= 0.6) or ((np.round(theta, 2)) >= 2.6 and (np.round(theta, 2)) <= 2.7):
            if (rho >= -63 and rho <= -15):
                up[0] += 1
            elif (rho >= 67 and rho <= 74):
                down[1] += 1
                up[1] += 1
            elif (rho >= 160 and rho <= 171):
                down[0] += 1

    # 검출된 방향을 저장할 변수 초기화
    direction = "No arrow detected"
    angle = 0.0

    # 검출된 방향 기반으로 화살표 방향 설정
    if left[0] >= 1 and left[1] >= 1:
        direction = "left"
    elif right[0] >= 1 and right[1] >= 1:
        direction = "right"
    elif up[0] >= 1 and up[1] >= 1:
        direction = "up"
    elif down[0] >= 1 and down[1] >= 1:
        direction = "down"

    # 검출된 선들과 각도를 프레임에 표시
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            #cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # 각도 계산
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi

    # 화살표 방향과 각도를 프레임에 표시
    cv2.putText(frame, f"Direction: {direction}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Angle: {angle:.2f} degrees", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    return frame, direction

if __name__ == "__main__":
    cap = cv2.VideoCapture(4)  # 웹캠에서 입력 받음

    if not cap.isOpened():
        print("Error: 웹캠을 열 수 없습니다.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame, direction = detect_yellow_arrow(frame)

        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

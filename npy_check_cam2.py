import numpy as np
import cv2
from time import sleep

def preprocess(frame):
    # 가우시안 블러 적용
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    return blurred

def get_min_max_hsv(hsv, mask):
    # 마스크에서 유효한 픽셀 추출
    points = cv2.findNonZero(mask)  # 마스크에서 0이 아닌 모든 픽셀(흰색) 좌표 찾기
    if points is not None:
        hsv_values = hsv[points[:, 0, 1], points[:, 0, 0]]  # 흰색 픽셀의 y 좌표 배열, 흰색 픽셀의 x 좌표 배열
        min_hsv = hsv_values.min(axis=0)  # 최소값
        max_hsv = hsv_values.max(axis=0)  # 최대값
        return min_hsv, max_hsv
    else:
        return None, None

def load_hsv_ranges():
    # HSV 범위를 npy 파일에서 불러오기
    green_hsv_range = np.load('green_hsv_range.npy')
    brown_hsv_range = np.load('brown_hsv_range.npy')
    red_hsv_range1 = np.load('red_hsv_range1.npy')
    red_hsv_range2 = np.load('red_hsv_range2.npy')
    yellow_hsv_range = np.load('yellow_hsv_range.npy')

    # 각 범위의 lower와 upper 값을 분리
    green_lower, green_upper = green_hsv_range
    brown_lower, brown_upper = brown_hsv_range
    red_lower1, red_upper1 = red_hsv_range1
    red_lower2, red_upper2 = red_hsv_range2
    yellow_lower, yellow_upper = yellow_hsv_range

    return green_lower, green_upper, brown_lower, brown_upper, red_lower1, red_upper1, red_lower2, red_upper2, yellow_lower, yellow_upper

def detect(frame, green_lower, green_upper, brown_lower, brown_upper, red_lower1, red_upper1, red_lower2, red_upper2, yellow_lower, yellow_upper):
    # 전처리
    preprocessed_frame = preprocess(frame)
    
    # HSV 색 공간으로 변환
    hsv = cv2.cvtColor(preprocessed_frame, cv2.COLOR_BGR2HSV)

    # 마스크 생성
    green_mask = cv2.inRange(hsv, green_lower, green_upper)
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)

    # 컨투어 찾기
    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # 최종 마스크 초기화
    final_mask = np.zeros_like(green_mask)

    for contour in contours:
        # Convex Hull 계산
        hull = cv2.convexHull(contour)

        # Convex Hull을 따라 마스크 생성
        contour_mask = np.zeros_like(green_mask)
        cv2.drawContours(contour_mask, [hull], -1, 255, -1)

        # 초록색 컨투어 내에서 갈색 영역 추출
        brown_mask = cv2.inRange(hsv, brown_lower, brown_upper)
        brown_in_green_mask = cv2.bitwise_and(brown_mask, contour_mask)
        
        # 초록색 컨투어 내에서 첫 번째 빨간색 영역 추출
        red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
        red_in_green_mask1 = cv2.bitwise_and(red_mask1, contour_mask)
        
        # 초록색 컨투어 내에서 두 번째 빨간색 영역 추출
        red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
        red_in_green_mask2 = cv2.bitwise_and(red_mask2, contour_mask)

        # 최종 마스크에 초록색 마스크 추가
        final_mask = cv2.bitwise_or(final_mask, green_mask)
        final_mask = cv2.bitwise_or(final_mask, brown_in_green_mask)
        final_mask = cv2.bitwise_or(final_mask, red_in_green_mask1)
        final_mask = cv2.bitwise_or(final_mask, red_in_green_mask2)

    # 노란색 마스크를 최종 마스크에 추가
    final_mask = cv2.bitwise_or(final_mask, yellow_mask)
    
    # 모폴로지 연산으로 경계 다듬기
    kernel = np.ones((5, 5), np.uint8)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)
    
    # 최종 마스크 블러링
    final_mask = cv2.GaussianBlur(final_mask, (5, 5), 0)

    # 최종 마스크를 사용하여 원본 이미지에서 해당 영역만 남기기
    result = cv2.bitwise_and(frame, frame, mask=final_mask)

    # 결과 출력
    cv2.imshow('Result', result)
    
    return result

def main():
    # HSV 범위 로드
    green_lower, green_upper, brown_lower, brown_upper, red_lower1, red_upper1, red_lower2, red_upper2, yellow_lower, yellow_upper = load_hsv_ranges()

    # 웹캠 열기
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to open webcam.")
        return
    
    while True:
        # 프레임 캡처
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame from webcam.")
            break
        
        # 프레임 처리
        processed_frame = detect(frame, green_lower, green_upper, brown_lower, brown_upper, red_lower1, red_upper1, red_lower2, red_upper2, yellow_lower, yellow_upper)

        # 'q' 키를 누르면 종료
        key = cv2.waitKey(1) & 0xff
        if key == ord('q'):
            break

    # 웹캠 및 모든 창 닫기
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()


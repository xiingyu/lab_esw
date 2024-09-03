import cv2
import numpy as np
from this_pixel_dist_class import RedBallDetector

## FSM 상태 정의 (일단 기본만)
STATE_INIT = 0 # 초기상태
STATE_CHECK_5TH_SECTION = 1 # 지금은 이게 공 찾는 것이긴 함
#STATE_CALCULATE_DIST = 2 # 픽셀 거리 계산
STATE_CHECK_8TH_SECTION = 2 # 8번째 섹션인지 확인
STATE_IS_HOLE = 3 # YELLOW_HOLE인지 확인
STATE_IS_ALIGN = 4 # align인지 확인

class Alignment_FSM_1000:
    def __init__(self):
        self.state = STATE_INIT
        self.detector = RedBallDetector()
        self.cap = cv2.VideoCapture('./re_alignment_case1.MP4') # 1000정도의 거리, 완전상태 영상

    def process_frame(self, frame): # FSM 넘기는 것
        if self.state == STATE_CHECK_5TH_SECTION:
            return self.check_center(frame)
        #elif self.state == STATE_CALCULATE_DIST:
            #return self.
        elif self.state == STATE_CHECK_8TH_SECTION:
            return self.check_8th_section(frame)
        elif self.state == STATE_IS_HOLE:
            return self.find_hole(frame)
        elif self.state == STATE_IS_ALIGN:
            return self.check_align(frame)
        else:
            return frame
    
    def find_ball(self, frame):
        frame, _, _, _, _, case_number = self.detector.detect_and_calculate_distance(frame)
        if case_number:
            print("공이 탐지되었습니다.")
            self.state = STATE_CHECK_8TH_SECTION
        else:
            print("공이 없습니다. 공 찾기 단계로 돌아갑니다.")
            self.state = STATE_INIT
        return frame
        
    # [영상처리] 9분할 화면 중 가운데에 있는지 확인 및 픽셀 거리 측정
    def check_center(self, frame):
        frame, _, _, _, _, case_number = self.detector.detect_and_calculate_distance(frame) # is not self.detector.determine_case_number
        if case_number == 5:  # 가운데에 위치한 경우
            print("공이 가운데에 있습니다.")
            #self.state = STATE_CALCULATE_DIST
            self.state = STATE_CHECK_8TH_SECTION
        else:
            print(f"공이 {case_number}번 섹션에 있습니다. 방향 조정 필요.")
            # 여기서 필요한 방향 조정 로직을 구현해야 함
            self.state = STATE_CHECK_5TH_SECTION
        return frame
    
    # 여기에 이제 픽셀 거리 추정하는 것 넣을까 했는데, self.detector.detect_and_calculate_distance을 애초에 나누는 게 나을 것 같다
    def pixel_dist(self, frame): # return frame, distance, center, pixel_area, walk_dist, case_number
        frame, _, _, _, _, case_number = self.detector.detect_and_calculate_distance(frame)

        return frame

    
    # 공이 탐지되고, 게걸음 및 직진 후, 고개 아래로 내리면 8번째 섹션임
    def check_8th_section(self, frame):
        frame, _, _, _, _, case_number = self.detector.detect_and_calculate_distance(frame)
        if case_number == 8:
            print("공이 8번째 섹션에 있습니다. 앞으로 이동.")
            self.state = STATE_IS_HOLE
        else:
            print("공이 8번째 섹션에 없습니다. 다시 확인합니다.")
            self.state = STATE_CHECK_8TH_SECTION # 이 부분은 STATE_CHECK_5TH_SECTION으로 돌아가서 공을 다시 찾는 것이 더 자연스러울 것 같긴함
        return frame
    def find_hole(self, frame):
        print("일단 홀찾기 스테이트로 넘어왔는지 확인")
        self.state = STATE_IS_ALIGN
        return frame
    
    def check_align(self, frame):
        print("일단 정렬 확인 스테이트로 넘어왔는지 확인")
        return frame

    def run(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            processed_frame = self.process_frame(frame)
            cv2.imshow('Processed Frame', processed_frame)

            if cv2.waitKey(33) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    fsm = Alignment_FSM_1000()
    fsm.state = STATE_CHECK_5TH_SECTION  # 초기 상태 설정
    fsm.run()

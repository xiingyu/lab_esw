import cv2
import numpy as np
import os
from math import atan

class LKAS:
    def __init__(self):
        self.nwindows = 10
        self.window_height = None
        self.nothing_flag = False
        self.start_time = cv2.getTickCount() / cv2.getTickFrequency()

    def detect_color(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        yellow_lower = np.array([15, 80, 0])
        yellow_upper = np.array([45, 255, 255])
        white_lower = np.array([0, 0, 200])
        white_upper = np.array([179, 64, 255])

        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
        white_mask = cv2.inRange(hsv, white_lower, white_upper)
        blend_mask = cv2.bitwise_or(yellow_mask, white_mask)
        blend_color = cv2.bitwise_and(img, img, mask=blend_mask)
        return blend_color

    def img_warp(self, img):
        self.img_x, self.img_y = img.shape[1], img.shape[0]

        src_center_offset = [200, 315]
        src = np.float32([
            [0, 479],
            [src_center_offset[0], src_center_offset[1]],
            [640 - src_center_offset[0], src_center_offset[1]],
            [639, 479]
        ])
        dst_offset = [round(self.img_x * 0.125), 0]
        dst = np.float32([
            [dst_offset[0], self.img_y],
            [dst_offset[0], 0],
            [self.img_x - dst_offset[0], 0],
            [self.img_x - dst_offset[0], self.img_y]
        ])

        matrix = cv2.getPerspectiveTransform(src, dst)
        warp_img = cv2.warpPerspective(img, matrix, (self.img_x, self.img_y))
        return warp_img

    def img_binary(self, blend_line):
        bin = cv2.cvtColor(blend_line, cv2.COLOR_BGR2GRAY)
        binary_line = np.zeros_like(bin)
        binary_line[bin != 0] = 1
        return binary_line

    def detect_nothing(self):
        self.nothing_left_x_base = int(round(self.img_x * 0.140625))
        self.nothing_right_x_base = int(self.img_x - round(self.img_x * 0.140625))

        self.nothing_pixel_left_x = np.array(np.zeros(self.nwindows) + round(self.img_x * 0.140625))
        self.nothing_pixel_right_x = np.array(np.zeros(self.nwindows) + self.img_x - round(self.img_x * 0.140625))
        self.nothing_pixel_y = np.array([round(self.window_height / 2) * index for index in range(0, self.nwindows)])

    def window_search(self, binary_line):
        bottom_half_y = binary_line.shape[0] / 2
        histogram = np.sum(binary_line[int(bottom_half_y):, :], axis=0)
        midpoint = int(histogram.shape[0] / 2)
        left_x_base = np.argmax(histogram[:midpoint])
        right_x_base = np.argmax(histogram[midpoint:]) + midpoint

        left_x_current = self.nothing_left_x_base if left_x_base == 0 else left_x_base
        right_x_current = self.nothing_right_x_base if right_x_base == midpoint else right_x_base

        out_img = np.dstack((binary_line, binary_line, binary_line)) * 255

        nwindows = self.nwindows
        window_height = self.window_height
        margin = 80
        min_pix = int(round((margin * 2 * window_height) * 0.0031))

        lane_pixel = binary_line.nonzero()
        lane_pixel_y = np.array(lane_pixel[0])
        lane_pixel_x = np.array(lane_pixel[1])

        left_lane_idx = []
        right_lane_idx = []

        for window in range(nwindows):
            win_y_low = int(binary_line.shape[0] - (window + 1) * window_height)
            win_y_high = int(binary_line.shape[0] - window * window_height)

            win_x_left_low = int(left_x_current - margin)
            win_x_left_high = int(left_x_current + margin)
            win_x_right_low = int(right_x_current - margin)
            win_x_right_high = int(right_x_current + margin)

            if left_x_current != 0:
                cv2.rectangle(out_img, (win_x_left_low, win_y_low), (win_x_left_high, win_y_high), (0, 255, 0), 2)
            if right_x_current != midpoint:
                cv2.rectangle(out_img, (win_x_right_low, win_y_low), (win_x_right_high, win_y_high), (0, 0, 255), 2)

            good_left_idx = (
                (lane_pixel_y >= win_y_low)
                & (lane_pixel_y < win_y_high)
                & (lane_pixel_x >= win_x_left_low)
                & (lane_pixel_x < win_x_left_high)
            ).nonzero()[0]
            good_right_idx = (
                (lane_pixel_y >= win_y_low)
                & (lane_pixel_y < win_y_high)
                & (lane_pixel_x >= win_x_right_low)
                & (lane_pixel_x < win_x_right_high)
            ).nonzero()[0]

            left_lane_idx.append(good_left_idx)
            right_lane_idx.append(good_right_idx)

            if len(good_left_idx) > min_pix:
                left_x_current = int(np.mean(lane_pixel_x[good_left_idx]))
            if len(good_right_idx) > min_pix:
                right_x_current = int(np.mean(lane_pixel_x[good_right_idx]))

        left_lane_idx = np.concatenate(left_lane_idx)
        right_lane_idx = np.concatenate(right_lane_idx)

        left_x = lane_pixel_x[left_lane_idx]
        left_y = lane_pixel_y[left_lane_idx]
        right_x = lane_pixel_x[right_lane_idx]
        right_y = lane_pixel_y[right_lane_idx]

        if len(left_x) == 0 and len(right_x) == 0:
            left_x = self.nothing_pixel_left_x
            left_y = self.nothing_pixel_y
            right_x = self.nothing_pixel_right_x
            right_y = self.nothing_pixel_y
        else:
            if len(left_x) == 0:
                left_x = right_x - self.img_x / 2
                left_y = right_y
            elif len(right_x) == 0:
                right_x = left_x + self.img_x / 2
                right_y = left_y

        left_fit = np.polyfit(left_y, left_x, 2)
        right_fit = np.polyfit(right_y, right_x, 2)
        plot_y = np.linspace(0, binary_line.shape[0] - 1, 5)
        left_fit_x = left_fit[0] * plot_y ** 2 + left_fit[1] * plot_y + left_fit[2]
        right_fit_x = right_fit[0] * plot_y ** 2 + right_fit[1] * plot_y + right_fit[2]
        center_fit_x = (right_fit_x + left_fit_x) / 2

        center = np.asarray(tuple(zip(center_fit_x, plot_y)), np.int32)
        right = np.asarray(tuple(zip(right_fit_x, plot_y)), np.int32)
        left = np.asarray(tuple(zip(left_fit_x, plot_y)), np.int32)

        cv2.polylines(out_img, [left], False, (0, 0, 255), thickness=5)
        cv2.polylines(out_img, [right], False, (0, 255, 0), thickness=5)
        sliding_window_img = out_img
        return sliding_window_img, left, right, center, left_x, left_y, right_x, right_y

    def meter_per_pixel(self):
        world_warp = np.array([[97, 1610], [109, 1610], [109, 1606], [97, 1606]], np.float32)
        meter_x = np.sum((world_warp[0] - world_warp[3]) ** 2)
        meter_y = np.sum((world_warp[0] - world_warp[1]) ** 2)
        meter_per_pix_x = meter_x / self.img_x
        meter_per_pix_y = meter_y / self.img_y
        return meter_per_pix_x, meter_per_pix_y

    def calc_curve(self, left_x, left_y, right_x, right_y):
        y_eval = self.img_x - 1
        meter_per_pix_x, meter_per_pix_y = self.meter_per_pixel()

        left_fit_cr = np.polyfit(left_y * meter_per_pix_y, left_x * meter_per_pix_x, 2)
        right_fit_cr = np.polyfit(right_y * meter_per_pix_y, right_x * meter_per_pix_x, 2)

        left_curve_radius = ((1 + (2 * left_fit_cr[0] * y_eval * meter_per_pix_y + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * left_fit_cr[0]
        )
        right_curve_radius = ((1 + (2 * right_fit_cr[0] * y_eval * meter_per_pix_y + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * right_fit_cr[0]
        )

        return left_curve_radius, right_curve_radius

    def calc_vehicle_offset(self, sliding_window_img, left_x, left_y, right_x, right_y):
        left_fit = np.polyfit(left_y, left_x, 2)
        right_fit = np.polyfit(right_y, right_x, 2)

        bottom_y = sliding_window_img.shape[0] - 1
        bottom_x_left = left_fit[0] * (bottom_y ** 2) + left_fit[1] * bottom_y + left_fit[2]
        bottom_x_right = right_fit[0] * (bottom_y ** 2) + right_fit[1] * bottom_y + right_fit[2]
        vehicle_offset = sliding_window_img.shape[1] / 2 - (bottom_x_left + bottom_x_right) / 2

        meter_per_pix_x, meter_per_pix_y = self.meter_per_pixel()
        vehicle_offset *= meter_per_pix_x

        return vehicle_offset

    def cam_cal_steer(self, left_curve_radius, right_curve_radius, vehicle_offset):
        curvature = 1 / ((left_curve_radius + right_curve_radius) / 2)
        cam_steer = (atan((1 * curvature) / 1 - (1 / 2) * curvature)) * 100
        if vehicle_offset > 0:
            cam_steer = -cam_steer
        return cam_steer

    def process_image(self, img):
        self.window_height = int(img.shape[0] / self.nwindows)
        warp_img = self.img_warp(img)
        blend_img = self.detect_color(warp_img)
        binary_img = self.img_binary(blend_img)
        if not self.nothing_flag:
            self.detect_nothing()
            self.nothing_flag = True

        sliding_window_img, left, right, center, left_x, left_y, right_x, right_y = self.window_search(binary_img)

        left_curve_radius, right_curve_radius = self.calc_curve(left_x, left_y, right_x, right_y)
        vehicle_offset = self.calc_vehicle_offset(sliding_window_img, left_x, left_y, right_x, right_y)
        cam_steer = self.cam_cal_steer(left_curve_radius, right_curve_radius, vehicle_offset)

        os.system("clear")
        print("------------------------------")
        print("left : {}".format(left))
        print("right : {}".format(right))
        print("center : {}".format(center))
        print("left_x : {}".format(left_x))
        print("left_y : {}".format(left_y))
        print("right_x : {}".format(right_x))
        print("right_y : {}".format(right_y))
        print("left_curve_radius : {}".format(left_curve_radius))
        print("right_curve_radius : {}".format(right_curve_radius))
        print("vehicle_offset : {}".format(vehicle_offset))
        print("steering angle : {}".format(cam_steer))
        print("time : {}".format(self.start_time))
        print("------------------------------")

        return sliding_window_img

if __name__ == "__main__":
    lkas = LKAS()
    cap = cv2.VideoCapture("C:/Users/nrk52/downloads/lkas_test.mov")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        sliding_window_img = lkas.process_image(frame)

        cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Sliding Window Image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Original Image", 800, 600)
        cv2.resizeWindow("Sliding Window Image", 800, 600)
        cv2.imshow("Original Image", frame)
        cv2.imshow("Sliding Window Image", sliding_window_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

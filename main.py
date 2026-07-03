import cv2 as cv
import numpy as np
import math

cap = cv.VideoCapture("sample.mp4")

feature_params = dict(maxCorners=100, 
                      qualityLevel=0.3, 
                      minDistance=7, 
                      blockSize=7)

lk_params = dict(winSize=(15, 15), 
                 maxLevel=2, 
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

isTrue, old_frame = cap.read()
old_frame = old_frame[100:-400, 200:-200]

old_hsv = cv.cvtColor(old_frame, cv.COLOR_BGR2HSV)
old_v = old_hsv[:, :, 2]

mask = np.zeros_like(old_v)
roicorners = np.array([[(0,85),(300,25),(625,25),(920,85)]]) 
cv.fillPoly(mask, roicorners, 255)

p0 = cv.goodFeaturesToTrack(old_v, mask=mask, **feature_params)

tracking_canvas = np.zeros_like(old_frame)

midpoint = old_frame.shape[1] // 2

num_slice = 10
smoothed_centers = [None] * num_slice

half_width = np.linspace(75,350,num_slice)

while True:
    isTrue, frame = cap.read()
    if not isTrue:
        print("Video ended.")
        break
        
    frame = frame[100:-400, 200:-200]

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    only_v = hsv[:, :, 2]
    
    p1, st, err = cv.calcOpticalFlowPyrLK(old_v, only_v, p0, None, **lk_params)

    tracking_canvas = cv.addWeighted(tracking_canvas, 0.9, tracking_canvas, 0, 0)

    if p1 is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        left_points = []
        right_points = []

        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            tracking_canvas = cv.line(tracking_canvas, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
            if a < midpoint:
                left_points.append((a, b))
                frame = cv.circle(frame, (int(a), int(b)), 5, (0, 0, 255), -1)
            else:
                right_points.append((a, b))
                frame = cv.circle(frame, (int(a), int(b)), 5, (255, 0, 0), -1)

        num_slice = 10
        slice_height = frame.shape[0] // num_slice

        center_points = []

        for i in range(num_slice):
            y_top = i * slice_height
            y_bottom = (i + 1) * slice_height

            left_x_inslice = [p[0] for p in left_points if y_top <= p[1] < y_bottom]
            right_x_inslice = [p[0] for p in right_points if y_top <= p[1] < y_bottom]

            center_x = None

            if len(left_x_inslice) > 0 and len(right_x_inslice) > 0:
                left_avg_x = sum(left_x_inslice) / len(left_x_inslice)
                right_avg_x = sum(right_x_inslice) / len(right_x_inslice)
                center_x = (left_avg_x + right_avg_x) / 2

            elif len(left_x_inslice) > 0:
                left_avg_x = sum(left_x_inslice) / len(left_x_inslice)
                center_x = left_avg_x + half_width[i]

            elif len(right_x_inslice) > 0:
                right_avg_x = sum(right_x_inslice) / len(right_x_inslice)
                center_x = right_avg_x - half_width[i]

            if center_x is not None:
                center_y = (y_top + y_bottom) / 2

                if smoothed_centers[i] is None:
                    smoothed_centers[i] = center_x
                else:
                    smoothed_centers[i] = (0.8 * smoothed_centers[i]) + (0.2 * center_x)

                frame = cv.circle(frame, (int(smoothed_centers[i]), int(center_y)), 5, (0, 255, 255), -1)

            valid_x = []
            valid_y = []

            for idx, center_x in enumerate(smoothed_centers):
                if center_x is not None:
                    slice_y = int((idx * slice_height) + (slice_height // 2))
                    valid_y.append(slice_y)
                    valid_x.append(center_x)
                
            if len(valid_y) >= 6:
                curve_fit = np.polyfit(valid_y, valid_x, 2)
                line_fit = np.polyfit(valid_y, valid_x, 1)

                if abs(curve_fit[0]) < 0.001:
                    poly_fit = np.array([0, line_fit[0], line_fit[1]])
                else:
                    poly_fit = curve_fit

            elif len(valid_y) >= 2:
                linear_fit = np.polyfit(valid_y, valid_x, 1)
                poly_fit = np.array([0, linear_fit[0], linear_fit[1]])
            else:
                poly_fit = None

            if poly_fit is not None:
                top_y = min(valid_y)
                ploty = np.linspace(top_y, frame.shape[0] - 1, frame.shape[0] - top_y, dtype=np.int32)
                smooth_fitx = poly_fit[0] * ploty**2 + poly_fit[1] * ploty + poly_fit[2]
                # Keep the fitted centerline calculation for future use, but do not draw the filled path overlay.
                car_x = frame.shape[1] // 2
                lookahead_y = int(frame.shape[0] * 0.6)

                if lookahead_y < top_y:
                    lookahead_y = top_y

                target_x = poly_fit[0] * lookahead_y**2 + poly_fit[1] * lookahead_y + poly_fit[2]
                steering_angle = math.atan2(target_x - car_x, frame.shape[0] - lookahead_y) * (180.0 / math.pi)
                

    output = cv.add(frame, tracking_canvas)
    cv.imshow("Optical Flow Tracking", output)

    if cv.waitKey(20) & 0xFF == ord("d"):
        break

    # --- THE NEW REPLENISHMENT LOGIC ---
    old_v = only_v.copy()

    # Check if we are running out of points (starving)
    if len(good_new) < 20: 
        # Find a brand new batch of points using the current frame and your ROI mask
        new_points = cv.goodFeaturesToTrack(only_v, mask=mask, **feature_params)
        
        if new_points is not None:
            p0 = new_points
        else:
            p0 = good_new.reshape(-1, 1, 2)
    else:
        # We have enough points, just keep tracking the ones we have
        p0 = good_new.reshape(-1, 1, 2)

cap.release()
cv.destroyAllWindows()
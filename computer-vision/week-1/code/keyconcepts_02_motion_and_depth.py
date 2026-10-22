"""
KEY CONCEPTS - Script 2: Optical flow, Kalman filter, Object tracking, Stereo depth
=====================================================================================
From "CV-Week-1-KeyConcepts.pdf". No video file needed - we generate the frames.

  Optical flow   - how far each point moved between two frames (a motion arrow per point)
  Kalman filter  - predict where something will be next, then correct with the
                   (noisy) measurement -> a smooth, reliable track
  Object tracking- following an object across frames (detection + prediction)
  Stereo vision  - two cameras side by side; nearby things shift MORE between the two
                   views -> that shift ("disparity") tells you depth
  Epipolar geometry - the rule that says WHERE in the second image to search

Run:  python keyconcepts_02_motion_and_depth.py
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from cv_helpers import make_shapes_image, show, OUTPUT_DIR

# =================================================================
# 1. OPTICAL FLOW (Lucas-Kanade)
#    Assumption: a point keeps the SAME brightness as it moves.
#    Lucas-Kanade = solve for motion in a small LOCAL window around each point.
#    (Horn-Schunck = solve for the whole image with a global smoothness rule.)
# =================================================================
frame1 = make_shapes_image()
M = np.float32([[1, 0, 18], [0, 1, 10]])                       # everything moves 18 right, 10 down
frame2 = cv2.warpAffine(frame1, M, (frame1.shape[1], frame1.shape[0]), borderValue=255)

p0 = cv2.goodFeaturesToTrack(frame1, maxCorners=80, qualityLevel=0.05, minDistance=10)   # Shi-Tomasi corners
p1, status, err = cv2.calcOpticalFlowPyrLK(frame1, frame2, p0, None, winSize=(21, 21), maxLevel=3)

flow_img = cv2.cvtColor(frame2, cv2.COLOR_GRAY2BGR)
moves = []
for (a, b), (c, d), ok in zip(p0.reshape(-1, 2), p1.reshape(-1, 2), status.ravel()):
    if ok:
        moves.append((c - a, d - b))
        cv2.arrowedLine(flow_img, (int(a), int(b)), (int(c), int(d)), (0, 0, 255), 2, tipLength=0.4)
mean_move = np.mean(moves, axis=0)
print(f"Optical flow: tracked {len(moves)} points, average motion = ({mean_move[0]:.1f}, {mean_move[1]:.1f}) px "
      f"(true: 18, 10)")
show([frame1, flow_img], ["Frame 1 (+ corners to track)", "Frame 2 with motion arrows (Lucas-Kanade)"],
     filename="keyconcepts_02a_optical_flow.png")

# =================================================================
# 2. KALMAN FILTER - tracking a ball from noisy measurements
#    State = [x, y, vx, vy] (position + velocity). Each step:
#      PREDICT: where should it be now, given its last speed? (motion model)
#      UPDATE:  blend in the new noisy measurement -> corrected estimate
#    Works best for roughly linear motion with Gaussian noise
#    (Extended Kalman Filter handles curved/non-linear cases).
# =================================================================
rng = np.random.default_rng(0)
t = np.arange(60)
true_x, true_y = 20 + 8 * t, 50 + 4 * t + 0.08 * t ** 2              # a thrown ball
meas = np.stack([true_x + rng.normal(0, 15, t.size), true_y + rng.normal(0, 15, t.size)], 1)

kf = cv2.KalmanFilter(4, 2)                                          # 4 state numbers, 2 measured
kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.05
kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 225          # we tell it: measurements are noisy (15^2)
kf.errorCovPost = np.eye(4, dtype=np.float32) * 100
kf.statePost = np.array([[meas[0, 0]], [meas[0, 1]], [0], [0]], np.float32)

est = []
for z in meas:
    kf.predict()                                                    # step 1: predict
    s = kf.correct(z.reshape(2, 1).astype(np.float32))              # step 2: correct with measurement
    est.append(s[:2].ravel().copy())                                # .copy(): s is the filter's live state
est = np.array(est)
err_meas = np.mean(np.hypot(meas[:, 0] - true_x, meas[:, 1] - true_y))
err_kf = np.mean(np.hypot(est[10:, 0] - true_x[10:], est[10:, 1] - true_y[10:]))
print(f"Kalman: average error raw measurements = {err_meas:.1f} px, Kalman estimate = {err_kf:.1f} px")

plt.figure(figsize=(8, 5))
plt.plot(true_x, true_y, color="#1f2328", lw=2, label="true path")
plt.scatter(meas[:, 0], meas[:, 1], s=14, color="#d1242f", label="noisy measurements")
plt.plot(est[:, 0], est[:, 1], color="#0969da", lw=2, label="Kalman estimate")
plt.gca().invert_yaxis()                                            # image coordinates: y grows down
plt.legend(); plt.title("Kalman filter: predict -> correct each frame"); plt.xlabel("x"); plt.ylabel("y")
plt.savefig(os.path.join(OUTPUT_DIR, "keyconcepts_02b_kalman.png"), dpi=100, bbox_inches="tight")
plt.close()

# =================================================================
# 3. STEREO DEPTH - a "random-dot stereogram"
#    Left & right cameras see the same scene from slightly different positions.
#    A near object shifts MORE between views than the far background.
#      disparity = x_left - x_right       depth ~ (focal length * baseline) / disparity
#    Epipolar geometry: for rectified (aligned) cameras the matching point is on the
#    SAME ROW - so the search is 1-D. (Essential matrix = calibrated cameras,
#    Fundamental matrix = uncalibrated.)
#    Hard part: flat, texture-less regions have nothing to match.
# =================================================================
H, W = 240, 320
left = (rng.random((H, W)) * 255).astype(np.uint8)
right = left.copy()
right[:, :-8] = left[:, 8:]                           # background: shifted 8 px (far)
yy, xx = np.mgrid[:H, :W]
square = (abs(yy - 120) < 50) & (abs(xx - 150) < 60)
shift_near = 24
src = np.clip(xx + shift_near, 0, W - 1)
right[square] = left[yy[square], src[square]]         # square: shifted 24 px (near!)

stereo = cv2.StereoBM_create(numDisparities=48, blockSize=15)
disp = stereo.compute(left, right).astype(np.float32) / 16.0      # StereoBM returns disparity x16
print(f"Stereo: median disparity background = {np.median(disp[20:220, 230:300]):.0f} px, "
      f"near square = {np.median(disp[square]):.0f} px (bigger = closer)")
disp_vis = cv2.normalize(np.clip(disp, 0, None), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
show([left, right, disp_vis],
     ["Left camera (random dots)", "Right camera (looks the same!)", "Disparity map: bright = near"],
     filename="keyconcepts_02c_stereo.png")

# ---------------------------------------------------------------
# TRY THIS:
#   a) Optical flow: change the motion matrix M to a rotation (getRotationMatrix2D).
#   b) Kalman: make measurementNoiseCov small (e.g. 1) - the estimate follows the noise.
#   c) Stereo: change shift_near to 40 - the square gets brighter (closer).
# ---------------------------------------------------------------

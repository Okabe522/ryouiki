import cv2
import numpy as np
from ultralytics import YOLO
import sys


def main(video_path):
    model = YOLO("volleyball.pt")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] cannot open video: {video_path}")
        return

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) 
    out = cv2.VideoWriter("output_track.mp4",
                          cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    # --- カルマンの設定 ---
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
    kf.transitionMatrix  = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
    kf.processNoiseCov   = np.eye(4, dtype=np.float32) * 0.03
    #観測を優先→数字を小さく、kalmanの予測を優先→数字を大きく
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 80

    initialized = False
    predict_only_count = 0
    trail = []
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if initialized:
            pred = kf.predict()
            px, py = int(pred[0]), int(pred[1])
        else:
            px, py = None, None

        results = model(frame, conf=0.5)
        boxes = results[0].boxes

        volleyballs = [
            (float(b.conf[0]), *map(int, b.xyxy[0]))
            for b in boxes
            if model.names[int(b.cls[0])] == "volleyball"
        ]
        volleyballs.sort(reverse=True)

        found = False
        if volleyballs:
            conf, x1, y1, x2, y2 = volleyballs[0]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            if not initialized:
                kf.statePre = np.array([[cx],[cy],[0],[0]], np.float32)
                kf.correct(np.array([[cx],[cy]], np.float32))
                initialized = True
                tx, ty = cx, cy
                found = True
            else:
                dist = np.hypot(cx - px, cy - py)
                #観測値と予測値の距離の閾値、距離が遠すぎたら誤検出だと判断し観測値を採用しない
                if dist <= 50: 
                    est = kf.correct(np.array([[cx],[cy]], np.float32))
                    tx, ty = int(est[0]), int(est[1])
                    predict_only_count = 0
                    found = True
                else:
                    tx, ty = px, py

            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.circle(frame, (tx,ty), 5, (255,0,0), -1)
        else:
            if initialized:
                pred = kf.predict()
                tx, ty = int(pred[0]), int(pred[1])
                predict_only_count += 1
                #予測だけのフレームを何回まで許すか
                if predict_only_count >= 10:
                    initialized = False
            else:
                tx, ty = None, None

        if initialized and tx is not None:
            trail.append((tx,ty))
            trail = trail[-15:]
            for i in range(1, len(trail)):
                cv2.line(frame, trail[i-1], trail[i], (0,160,255), 3)

        cv2.putText(frame, f"Frame:{frame_id}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
     
        out.write(frame)
        cv2.imshow("Volleyball Tracking", frame)
        if cv2.waitKey(1) == 27:
            break

        frame_id += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使い方: python ball_tracking.py input_video.mp4")
    else:
        main(sys.argv[1])

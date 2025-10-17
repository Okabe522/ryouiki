# 合成データで「本当の位置→ノイズ観測→カルマン推定」を可視化して体験するサンプル。
# キー: ESC で終了

import cv2
import numpy as np

def main():
    # 画面サイズ
    W, H = 800, 450
    canvas = np.zeros((H, W, 3), dtype=np.uint8)

    # 本当の物体の初期位置・速度（右下へ等速直線運動）
    true_x, true_y = 100.0, 100.0
    vx, vy = 3.5, 2.0

   
  
    # OpenCVのカルマン（状態4次元: x,y,vx,vy / 観測2次元: x,y）
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1,0,0,0],
                                     [0,1,0,0]], dtype=np.float32)
    kf.transitionMatrix = np.array([[1,0,1,0],
                                    [0,1,0,1],
                                    [0,0,1,0],
                                    [0,0,0,1]], dtype=np.float32)

    # プロセスノイズ・観測ノイズの共分散（大きいほど信用しない）
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
     # 観測ノイズの標準偏差（大きいほど測定がブレる）
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * (10.0**2)

    # 初期化
    kf.statePre = np.array([[true_x], [true_y], [vx], [vy]], dtype=np.float32)
    initialized = True

    trail = []  

    frame_id = 0
    while True:
        frame = canvas.copy()
        frame[:] = 0

        # 真値の更新
        true_x += vx
        true_y += vy

        
        if true_x < 0 or true_x > W: vx *= -1; true_x = np.clip(true_x, 0, W)
        if true_y < 0 or true_y > H: vy *= -1; true_y = np.clip(true_y, 0, H)

        # 観測（たまに欠測させる）
        has_measurement = (frame_id % 15 != 0)  # 15フレごとに欠測
        if has_measurement:
            meas_x = true_x + np.random.randn() * 10.0
            meas_y = true_y + np.random.randn() * 10.0

        # 予測
        pred = kf.predict()
        px, py = float(pred[0,0]), float(pred[1,0])

        # 補正（観測があるときだけ）
        if has_measurement:
            m = np.array([[np.float32(meas_x)], [np.float32(meas_y)]], dtype=np.float32)
            est = kf.correct(m)
            ex, ey = float(est[0,0]), float(est[1,0])
        else:
            ex, ey = px, py  # 観測なし→予測のみ

        # 可視化：真値=白、観測=赤、推定=青
        cv2.circle(frame, (int(true_x), int(true_y)), 6, (255,255,255), -1)
        if has_measurement:
            cv2.circle(frame, (int(meas_x), int(meas_y)), 6, (0,0,255), 2)
        cv2.circle(frame, (int(ex), int(ey)), 6, (255,0,0), -1)

        # 軌跡（推定）を短く描画
        trail.append((int(ex), int(ey)))
        trail = trail[-25:]
        for i in range(1, len(trail)):
            cv2.line(frame, trail[i-1], trail[i], (255,0,0), 2)

        cv2.putText(frame, "white=true  red=measurement  blue=estimate",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)

        cv2.imshow("Kalman Demo", frame)
        key = cv2.waitKey(16) & 0xFF
        if key == 27:  # ESC
            break

        frame_id += 1

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

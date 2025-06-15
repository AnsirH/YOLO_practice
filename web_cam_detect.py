import cv2
import time
from ultralytics import YOLO

# 모델 로드
model = YOLO("yolov8n.pt")

# 클래스 이름 확인
CLASS_NAMES = model.names
# 감지할 클래스: "person"과 "cell phone"
TARGET_CLASSES = [cls_id for cls_id, name in CLASS_NAMES.items() if name in ['person', 'cell phone']]

# 웹캠 열기
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR]: 웹캠을 열 수 없습니다.")
    exit()

prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR]: 프레임을 읽을 수 없습니다.")
        break

    # YOLO 추론
    results = model(frame, verbose=False)[0]

    # 결과 복사
    annotated_frame = frame.copy()

    # 현재 시간
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time else 0
    prev_time = current_time

    # 감지 박스 처리
    if results.boxes is not None:
        for box in results.boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)
            if cls_id in TARGET_CLASSES and conf > 0.3:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = f"{CLASS_NAMES[cls_id]} {conf:.2f}"

                # 박스 그리기
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    annotated_frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0), 
                    2                    
                )

    # FPS 표시
    cv2.putText(
        annotated_frame,
        f"FPS: {fps:.2f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2
    )

    # 화면에 출력
    cv2.imshow("YOLOv8 Webcam Detection", annotated_frame)

    # "q" 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 종료 처리
cap.release()
cv2.destroyAllWindows()
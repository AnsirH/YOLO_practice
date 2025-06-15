from ultralytics import YOLO
import cv2
import os

# 1. 모델 불러오기( 사전 학습된 YOLO8 Nano 모델 )
model = YOLO("yolov8n.pt") # 사전 훈련된 detection 모델

# 2. 이미지 입력
input_path = "input.jpg" # 감지할 이미지 경로
output_path = "output_detected.jpg" # 추출할 이미지 경로

# 3. 이미지 추론
results = model(input_path)

# 결과 이미지 원본
img = results[0].orig_img.copy()

# 결과 박스 객체
boxes = results[0].boxes

# 클래스 필터: 사람( 0 ), 자동차( 2 )
target_classes = [0, 2]

if boxes is not None:
    for box in boxes:
        cls_id = int(box.cls)
        conf = float(box.conf)

        if cls_id in target_classes and conf > 0.3:
            # 박스 좌표: [x1, y1, x2, y2]
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

            # 클래스 이름
            label = model.names[cls_id]

            # 박스 그리기
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                img,
                f"{label} {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )
else:
    print("감지된 객체가 없습니다.")

# 이미지 저장
cv2.imwrite(output_path, img)
print(f"결과 이미지 저장 완료: {output_path}")
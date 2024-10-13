# your_project_code/predict.py

import torch
from pathlib import Path

def load_model(model_path='best.pt'):
    # 모델 로드
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
    return model

def predict_tomato(model, image_path):
    # 이미지 예측
    results = model(image_path)
    
    # 결과 파싱
    detections = results.xyxy[0]  # [x1, y1, x2, y2, confidence, class]
    
    tomato_info = []
    stem_info = []
    
    for *box, conf, cls in detections:
        cls = int(cls)
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        
        if cls == 0:  # 토마토
            color = 'red'  # 실제 색상 분석 필요 (예시로 고정)
            tomato_info.append({
                'bbox': (x1, y1, x2, y2),
                'width': width.item(),
                'height': height.item(),
                'color': color
            })
        elif cls == 1:  # 줄기
            stem_status = 'yellow'  # 실제 상태 분석 필요 (예시로 고정)
            stem_info.append({
                'bbox': (x1, y1, x2, y2),
                'status': stem_status
            })
    
    return tomato_info, stem_info

def should_harvest(tomato_info, stem_info, size_threshold=500, color_required='red', stem_required='yellow'):
    for tomato in tomato_info:
        if (tomato['color'] == color_required and
            tomato['width'] * tomato['height'] > size_threshold):
            # 해당 토마토에 대응하는 줄기 정보 확인 (간단화를 위해 첫 번째 줄기 정보 사용)
            if stem_info and stem_info[0]['status'] == stem_required:
                return True
    return False

if __name__ == "__main__":
    # 모델 경로
    model_path = 'runs/train/exp/weights/best.pt'  # 훈련된 모델 경로로 변경
    
    # 예측할 이미지 경로
    image_path = 'path/to/your/test_image.png'  # 실제 이미지 경로로 변경
    
    # 모델 로드
    model = load_model(model_path)
    
    # 예측 수행
    tomato_info, stem_info = predict_tomato(model, image_path)
    
    # 수확 가능 여부 판단
    if should_harvest(tomato_info, stem_info):
        print("토마토를 수확할 수 있습니다.")
    else:
        print("아직 수확할 수 없습니다.")
    
    # 결과 시각화 (옵션)
    results = model(image_path)
    results.show()  # 예측 결과를 창으로 표시

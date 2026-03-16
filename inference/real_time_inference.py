import onnxruntime as ort
import numpy as np
import cv2
import time

ONNX_MODEL_PATH = r"C:\Users\irona\Downloads\custom_yolov8_test\best.onnx" # CHANGE THIS if your model is named differently
CONF_THRESHOLD = 0.25           # Confidence threshold for filtering boxes
NMS_THRESHOLD = 0.45            # IoU threshold for Non-Max Suppression
INPUT_SIZE = 640              # YOLOv8 models typically use 640x640 input

# Default COCO classes (for standard YOLOv8 models)
# CLASS_NAMES = [
#     'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
#     'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
#     'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
#     'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
#     'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
#     'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
#     'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
#     'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
#     'hair drier', 'toothbrush'
# ]

CLASS_NAMES = [
    "forward", 'four', 'func', 'run', 'three', 'turn_lf', 'turn_rt', 'two'
]

try:
    session = ort.InferenceSession(ONNX_MODEL_PATH, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(f"ONNX Model '{ONNX_MODEL_PATH}' loaded successfully.")

except Exception as e:
    print(f"ERROR: Could not load model or find required libraries. Check your paths and installations.")
    print(e)
    exit()

cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

def preprocess(frame):
    img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    input_tensor = np.expand_dims(img, axis=0)
    
    return input_tensor

def postprocess(output, original_shape, input_size):
    pred = np.squeeze(output[0])
    if pred.ndim == 2 and pred.shape[0] < pred.shape[1]:
        pred = pred.T  # (N, D)

    if pred.ndim != 2:
        return [], [], []

    D = pred.shape[1]
    nc = len(CLASS_NAMES)

    if D == 4 + nc:
        has_obj = False
        cls_start = 4
    elif D == 5 + nc:
        has_obj = True
        cls_start = 5
    else:
        return [], [], []

    boxes = pred[:, :4].astype(np.float32)

    if has_obj:
        obj = pred[:, 4].astype(np.float32)
        cls_probs = pred[:, cls_start:cls_start + nc].astype(np.float32)
        class_ids = np.argmax(cls_probs, axis=1)
        cls_conf = cls_probs[np.arange(cls_probs.shape[0]), class_ids]
        scores = obj * cls_conf
    else:
        cls_probs = pred[:, cls_start:cls_start + nc].astype(np.float32)
        class_ids = np.argmax(cls_probs, axis=1)
        scores = cls_probs[np.arange(cls_probs.shape[0]), class_ids]

    mask = scores > CONF_THRESHOLD
    boxes = boxes[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]

    if boxes.shape[0] == 0:
        return [], [], []

    scale_x = original_shape[1] / input_size
    scale_y = original_shape[0] / input_size

    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]

    x1 = (x - w / 2) * scale_x
    y1 = (y - h / 2) * scale_y
    x2 = (x + w / 2) * scale_x
    y2 = (y + h / 2) * scale_y

    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

    boxes_nms = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).astype(int)

    indices = cv2.dnn.NMSBoxes(
        boxes_nms.tolist(),
        scores.tolist(),
        CONF_THRESHOLD,
        NMS_THRESHOLD
    )

    if len(indices) == 0:
        return [], [], []

    indices = np.array(indices).reshape(-1)

    final_boxes = boxes_xyxy[indices].astype(int)
    final_scores = scores[indices]
    final_classes = class_ids[indices]

    return final_boxes, final_scores, final_classes

print("Starting real-time detection. Press 'q' to quit...")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    H, W = frame.shape[:2]
    
    input_tensor = preprocess(frame)

    start_time = time.time()
    output = session.run([output_name], {input_name: input_tensor})
    end_time = time.time()

    fps = 1 / (end_time - start_time)
    
    final_boxes, final_scores, final_classes = postprocess(output, (H, W), INPUT_SIZE)

    for box, score, class_id in zip(final_boxes, final_scores, final_classes):
        x1, y1, x2, y2 = box
        label = CLASS_NAMES[class_id]
        
        color = (int((class_id * 37) % 255), int((class_id * 73) % 255), int((class_id * 109) % 255))
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        text = f'{label}: {score:.2f}'
        cv2.putText(frame, text, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('YOLOv8 ONNX Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
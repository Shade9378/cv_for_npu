import argparse
import os
import onnxruntime as ort
import numpy as np
import cv2

DEFAULT_CONF_THRESHOLD = 0.25
DEFAULT_NMS_THRESHOLD = 0.50
DEFAULT_OUT_PATH = "out.jpg"

DEFAULT_CLASS_NAMES = ["forward", "four", "func", "run", "three", "turn_lf", "turn_rt", "two"]

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def needs_sigmoid(x: np.ndarray) -> bool:
    return (x.min() < 0.0) or (x.max() > 1.0)


def get_model_input_hw(session: ort.InferenceSession, fallback_hw=(640, 640)):
    ishape = session.get_inputs()[0].shape
    h = ishape[2] if isinstance(ishape[2], int) else fallback_hw[0]
    w = ishape[3] if isinstance(ishape[3], int) else fallback_hw[1]
    return int(h), int(w)


def preprocess_bgr(img_bgr: np.ndarray, in_hw):
    in_h, in_w = in_hw
    img = cv2.resize(img_bgr, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img


def postprocess(output, original_hw, in_hw, class_names, conf_thres, nms_thres):
    H, W = original_hw
    in_h, in_w = in_hw

    pred = np.squeeze(output[0])
    if pred.ndim != 2:
        return [], [], []

    if pred.shape[0] < pred.shape[1]:
        pred = pred.T

    _, D = pred.shape
    known_nc = len(class_names)

    if D == 4 + known_nc:
        has_obj = False
        nc = known_nc
        cls_start = 4
    elif D == 5 + known_nc:
        has_obj = True
        nc = known_nc
        cls_start = 5
    else:
        if D > 5:
            has_obj = True
            nc = D - 5
            cls_start = 5
        elif D > 4:
            has_obj = False
            nc = D - 4
            cls_start = 4
        else:
            return [], [], []

    boxes = pred[:, :4].astype(np.float32)

    if has_obj:
        obj = pred[:, 4].astype(np.float32)
        cls = pred[:, cls_start:cls_start + nc].astype(np.float32)

        if needs_sigmoid(obj):
            obj = sigmoid(obj)
        if needs_sigmoid(cls):
            cls = sigmoid(cls)

        class_ids = np.argmax(cls, axis=1)
        cls_conf = cls[np.arange(cls.shape[0]), class_ids]
        scores = obj * cls_conf
    else:
        cls = pred[:, cls_start:cls_start + nc].astype(np.float32)
        if needs_sigmoid(cls):
            cls = sigmoid(cls)

        class_ids = np.argmax(cls, axis=1)
        scores = cls[np.arange(cls.shape[0]), class_ids]

    keep = scores > conf_thres
    boxes = boxes[keep]
    scores = scores[keep]
    class_ids = class_ids[keep].astype(int)

    if boxes.shape[0] == 0:
        return [], [], []

    scale_x = W / float(in_w)
    scale_y = H / float(in_h)

    x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = (x - w / 2) * scale_x
    y1 = (y - h / 2) * scale_y
    x2 = (x + w / 2) * scale_x
    y2 = (y + h / 2) * scale_y

    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

    boxes_xyxy[:, 0] = np.clip(boxes_xyxy[:, 0], 0, W - 1)
    boxes_xyxy[:, 1] = np.clip(boxes_xyxy[:, 1], 0, H - 1)
    boxes_xyxy[:, 2] = np.clip(boxes_xyxy[:, 2], 0, W - 1)
    boxes_xyxy[:, 3] = np.clip(boxes_xyxy[:, 3], 0, H - 1)

    final_boxes, final_scores, final_classes = [], [], []
    for c in np.unique(class_ids):
        idx = np.where(class_ids == c)[0]
        b = boxes_xyxy[idx]
        s = scores[idx]

        nms_boxes = np.stack([b[:, 0], b[:, 1], b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], axis=1).astype(int)

        kept = cv2.dnn.NMSBoxes(
            nms_boxes.tolist(),
            s.tolist(),
            conf_thres,
            nms_thres
        )
        if kept is None or len(kept) == 0:
            continue

        kept = np.array(kept).reshape(-1)
        final_boxes.append(b[kept])
        final_scores.append(s[kept])
        final_classes.append(np.full(len(kept), c, dtype=int))

    if not final_boxes:
        return [], [], []

    final_boxes = np.vstack(final_boxes).astype(int)
    final_scores = np.concatenate(final_scores)
    final_classes = np.concatenate(final_classes)

    return final_boxes, final_scores, final_classes


def draw_detections(img_bgr, boxes, scores, classes, class_names):
    for box, score, cid in zip(boxes, scores, classes):
        x1, y1, x2, y2 = map(int, box)
        cid = int(cid)
        label = class_names[cid] if 0 <= cid < len(class_names) else f"class_{cid}"

        color = (int((cid * 37) % 255), int((cid * 73) % 255), int((cid * 109) % 255))
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)

        text = f"{label}: {float(score):.2f}"
        y_text = max(y1 - 10, 0)
        cv2.putText(img_bgr, text, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return img_bgr


def get_unique_path(path: str) -> str:
    path = os.path.abspath(path)
    out_dir = os.path.dirname(path)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(path):
        return path

    base, ext = os.path.splitext(path)
    counter = 1
    while True:
        candidate = f"{base}_{counter}{ext}"
        if not os.path.exists(candidate):
            return candidate
        counter += 1


def parse_args():
    p = argparse.ArgumentParser(description="Run YOLOv8 ONNX inference on a single image and save visualization.")
    p.add_argument("--model", required=True, help="Path to .onnx model")
    p.add_argument("--image", required=True, help="Path to input image")

    p.add_argument(
        "--out",
        default=DEFAULT_OUT_PATH,
        help="Output image path (default: out.jpg). If it exists, auto-saves as out_1.jpg, out_2.jpg, ..."
    )

    p.add_argument("--conf", type=float, default=DEFAULT_CONF_THRESHOLD, help="Confidence threshold (default: 0.25)")
    p.add_argument("--nms", type=float, default=DEFAULT_NMS_THRESHOLD, help="NMS IoU threshold (default: 0.5)")

    p.add_argument("--provider", default="CPUExecutionProvider",
                   help='ONNXRuntime provider (e.g., "CPUExecutionProvider", "CUDAExecutionProvider")')
    p.add_argument("--classes", default=None,
                   help='Optional: comma-separated class names, e.g. "a,b,c". If omitted, uses defaults in script.')
    p.add_argument("--no-show", action="store_true", help="Do not open an OpenCV window")
    p.add_argument("--fallback-hw", type=int, nargs=2, default=[640, 640],
                   metavar=("H", "W"), help="Fallback input H W if model uses dynamic shapes")
    p.add_argument("--verbose", action="store_true", help="Print extra debug info")
    return p.parse_args()


def main():
    args = parse_args()

    class_names = DEFAULT_CLASS_NAMES
    if args.classes:
        class_names = [c.strip() for c in args.classes.split(",") if c.strip()]
        if len(class_names) == 0:
            raise ValueError("--classes was provided but parsed to an empty list.")

    if not os.path.isfile(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")
    if not os.path.isfile(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")
    H, W = img.shape[:2]

    session = ort.InferenceSession(args.model, providers=[args.provider])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    in_hw = get_model_input_hw(session, fallback_hw=tuple(args.fallback_hw))

    if args.verbose:
        print(f"Loaded: {args.model}")
        print(f"Provider: {args.provider}")
        print(f"Input tensor: {input_name} shape={session.get_inputs()[0].shape}")
        print(f"Output tensor: {output_name} shape={session.get_outputs()[0].shape}")
        print(f"Using input HW: {in_hw}")
        print(f"Conf/NMS: {args.conf}/{args.nms}")
        print(f"Classes ({len(class_names)}): {class_names}")

    inp = preprocess_bgr(img, in_hw)
    out = session.run([output_name], {input_name: inp})

    if args.verbose:
        pred = np.squeeze(out[0])
        if pred.ndim == 2 and pred.shape[0] < pred.shape[1]:
            pred = pred.T
        if pred.size > 0:
            print("Raw output (N,D):", pred.shape, " min/max:", float(pred.min()), float(pred.max()))

    boxes, scores, classes = postprocess(
        out, (H, W), in_hw, class_names,
        conf_thres=args.conf,
        nms_thres=args.nms,
    )

    if args.verbose:
        print(f"Detections after NMS: {len(boxes)}")

    out_img = draw_detections(img.copy(), boxes, scores, classes, class_names)

    final_out_path = get_unique_path(args.out)
    cv2.imwrite(final_out_path, out_img)
    print(f"Saved: {final_out_path}")

    if not args.no_show:
        cv2.imshow("Detections", out_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

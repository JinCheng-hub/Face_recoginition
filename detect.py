import os
import pickle
import cv2
from skimage.feature import hog
import argparse


def load_model(model_path):
    try:
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)
        classifier = model_data['classifier']
        label_names = model_data['labels']
        scaler = model_data['scaler']
        return classifier, label_names, scaler
    except Exception as e:
        print(f"Loading model error: {e}")
        return None, None, None


def detect(model_path, source, threshold):
    classifier, label_names, scaler = load_model(model_path)
    model_path = "deploy.prototxt"
    weights_path = "res10_300x300_ssd_iter_140000.caffemodel"
    net = cv2.dnn.readNetFromCaffe(model_path, weights_path)

    try:
        source = int(source)
    except ValueError:
        pass

    cap = cv2.VideoCapture(source)
    cv2.namedWindow("Face Recognition", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Face Recognition", 1080, 1920)
    if not cap.isOpened():
        print(f"Can't open: {source}")
        return False

    os.makedirs("detect", exist_ok=True)
    exist_ids = [int(folder) for folder in os.listdir("detect") if folder.isdigit()]
    exp_id = str(max(exist_ids) + 1 if exist_ids else 1)
    exp_path = os.path.join("detect", exp_id)
    os.mkdir(exp_path)

    output_path = os.path.join(exp_path, "detection.avi")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) - 10

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Start face recognition， 'q' to quit...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        output_frame = frame.copy()
        h, w = frame.shape[:2]
        
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.6:
                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                (x1, y1, x2, y2) = box.astype("int")

                face = frame[y1:y2, x1:x2]
                face_resized = cv2.resize(face, (128, 128))
                face_hog = hog(
                    cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY),
                    orientations=9,
                    pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2),
                    block_norm='L2-Hys',
                    transform_sqrt=True
                )
                face_vector = scaler.transform([face_hog])
                prediction = classifier.predict(face_vector)[0]
                probabilities = classifier.predict_proba(face_vector)[0]
                max_prob = probabilities.max()

                name = label_names.inverse_transform([prediction])[0]

                color = (0, 255, 0) if max_prob > threshold else (0, 0, 255)
                cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
                text = f"{name} ({max_prob:.2f})"
                cv2.putText(
                    output_frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2
                )

        cv2.imshow("Face Recognition", output_frame)
        out.write(output_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Result saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Classifier model to recognition")
    parser.add_argument("--source", default="0", help="webcam ID or video path")
    parser.add_argument("--threshold", default=0.65, help="confidence threshold")
    args = parser.parse_args()

    detect(args.model, args.source, args.threshold)

if __name__ == "__main__":
    main()

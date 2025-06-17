import os
import cv2
import random
import numpy as np
from skimage.feature import hog
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from pytubefix import YouTube
from pytubefix.cli import on_progress
from googleapiclient.discovery import build
import pickle


def yt_download(query, num_results=10, save_dir="videos"):
    """Download videos from YouTube based on a search query"""
    api_key = "YOUR_YOUTUBE_API_KEY"

    youtube = build("youtube", "v3", developerKey=api_key)
    search_response = youtube.search().list(
        q=query,
        part="snippet",
        type="video",
        maxResults=num_results,
        videoDuration="medium"
    ).execute()

    video_links = [
        f"https://www.youtube.com/watch?v={item["id"]["videoId"]}"
        for item in search_response["items"] if item["id"]["kind"] == "youtube#video"
    ]
    
    os.makedirs(save_dir, exist_ok=True)
    for url in video_links:
        try:
            yt = YouTube(url, on_progress_callback=on_progress)
            print(f"Downloading: {yt.title}")
            stream = yt.streams.get_highest_resolution()
            stream.download(output_path=save_dir)
            print(f"Downloaded {yt.title} to {save_dir}")
        except Exception as e:
            print(f"Error downloading video: {e}")


def capture(person_name: str, source: str, limit: int):
    try:
        source = int(source)
    except ValueError:
        pass
    if source.isdigit() or source.endswith("mp4"):
        cap = cv2.VideoCapture(source)
    else:
        cap = None
    if cap and not cap.isOpened():
        print(f"Cannot open {source}")
        return

    save_dir = os.path.join("faces", person_name)
    os.makedirs(save_dir, exist_ok=True)
    ids = [int(file.split(".")[0]) for file in os.listdir(save_dir)]
    count = max(ids) if ids else 0
    index = 0

    net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

    print(f"Start capture {person_name}'s face...")
    
    while count < limit:
        if cap:
            ret, frame = cap.read()
            if not ret:
                break
        else:
            files = os.listdir(source)
            if index >= len(files):
                break
            frame = cv2.imread(os.path.join(source, files[index]))
            index += 1
            if frame is None:
                print("Failed to read:", files[index])
                continue

        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.7:
                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                (x1, y1, x2, y2) = box.astype("int")
                x1 = max(0, x1 - int((x2 - x1) * 0.1))
                y1 = max(0, y1 - int((y2 - y1) * 0.1))
                x2 = min(w, x2 + int((x2 - x1) * 0.1))
                y2 = min(h, y2 + int((y2 - y1) * 0.1))
                face = frame[y1:y2, x1:x2].copy()

                # cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                face = cv2.resize(face, (128, 128))
                count += 1
                filepath = os.path.join(save_dir, f"{count}.jpg")
                cv2.imwrite(filepath, face)
                print("Saved face to", filepath)

        # Uncomment to display the frame during capture
        # cv2.imshow("Face Capture", frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        
    if cap:
        cap.release()
    # cv2.destroyAllWindows()
    print(f"Saved {count} {person_name}'s faces")


def load_model(model_path):
    """Load a pre-trained model."""
    try:
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)
        classifier = model_data['classifier']
        label_names = model_data['labels']
        scaler = model_data['scaler']
        return classifier, label_names, scaler
    except Exception as e:
        print(f"Loading model error: {e}")
        return None, None, scaler


def classify_images(folder, model_path, threshold=0.8):
    """Classify images using a pre-trained SVM model."""
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')
    image_files = [f for f in os.listdir(folder) if f.lower().endswith(image_extensions)]
    model, label_names, scaler = load_model(model_path)
    score = []

    for image_file in image_files:
        image_path = os.path.join(folder, image_file)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None or img.shape[0] < 100 or img.shape[1] < 100:
            continue
        img = cv2.resize(img, (128, 128))
        img = hog(
            img,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm='L2-Hys',
            transform_sqrt=True
        )
        img = scaler.transform([img])
        max_prob = model.predict_proba(img)[0].max()
        print(image_file, max_prob)
        score.append(max_prob)
        if max_prob >= threshold:
            prediction = model.predict(img)[0]
            target = label_names.inverse_transform([prediction])[0]
            os.makedirs(target, exist_ok=True)
            os.rename(image_path, os.path.join(target, image_file))
            print(f"Moved {image_file} to {target}")


def filter_similar_images(folder, threshold=0.9):
    """Filter out similar images using SSIM."""
    os.makedirs(f"Similar_Images", exist_ok=True)
    image_files = [f for f in os.listdir(folder) if f.endswith(('png', 'jpg', 'jpeg'))]
    removed = 0

    for i in range(len(image_files)):
        for j in range(i + 1, len(image_files)):
            img1_path = os.path.join(folder, image_files[i])
            img2_path = os.path.join(folder, image_files[j])
            if not os.path.exists(img1_path) or not os.path.exists(img2_path):
                continue
            try:
                img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
                img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
                if img1 is None or img2 is None:
                    continue
                img1 = cv2.resize(img1, (128, 128))
                img2 = cv2.resize(img2, (128, 128))
                img1 = cv2.normalize(img1, None, 0, 255, cv2.NORM_MINMAX)
                img2 = cv2.normalize(img2, None, 0, 255, cv2.NORM_MINMAX)

                score, _ = ssim(img1, img2, full=True)
                if score > threshold:
                    print(f"Similar images detected: {image_files[i]} and {image_files[j]} (SSIM: {score:.2f})")

                    # plot similar images
                    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                    axes[0].imshow(img1, cmap='gray')
                    axes[0].set_title(f"Image 1: {image_files[i]}")
                    axes[0].axis('off')

                    axes[1].imshow(img2, cmap='gray')
                    axes[1].set_title(f"Image 2: {image_files[j]}")
                    axes[1].axis('off')

                    plt.suptitle(f"SSIM: {score:.2f}")
                    plt.savefig(f"Similar_Images/{os.path.basename(folder.rstrip('/'))}_{removed}.png")
                    plt.close()

                    removed += 1
                    os.remove(img2_path)
                    print(f"Removed {image_files[j]} (similar to {image_files[i]})")
            except Exception as e:
                print(f"Error comparing {img1_path} and {img2_path}: {e}")
                continue
    print(f"Removed {removed} similar images.")


def augment_image(image_path):
    """Create augmented versions of a single image."""
    base_name = image_path.split(".")[0]
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image could not be read. Please check the path.")

    h, w = image.shape[:2]

    # Rotation or Mask
    def rotation_mask():
        if random.choice(["rotation", "mask"]) == "rotation":
            angle = random.uniform(-15, 15)
            rotation_matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))
            cv2.imwrite(f"{base_name}_rotation.jpg", rotated_image)
        else:
            masked_image = image.copy()
            for _ in range(random.randint(1, 3)):
                mask_size = (random.randint(20, 50), random.randint(20, 50))
                mask_h, mask_w = mask_size
                top_left_x = random.randint(0, w - mask_w)
                top_left_y = random.randint(0, h - mask_h)
                mask_value = random.randint(0, 255)
                masked_image[top_left_y:top_left_y + mask_h, top_left_x:top_left_x + mask_w] = mask_value
            cv2.imwrite(f"{base_name}_mask.jpg", masked_image)

    # Brightness_contrast or Sharpen
    def contrast_sharpen():
        if random.choice(["contrast", "sharpen"]) == "contrast":
            brightness = random.randint(-30, 30)
            contrast = random.randint(-30, 30)
            mod = np.int16(image) * (contrast / 127 + 1) - contrast + brightness
            mod = np.clip(mod, 0, 255).astype(np.uint8)
            cv2.imwrite(f"{base_name}_brightness_contrast.jpg", mod)
        else:
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            sharpened_image = cv2.filter2D(image, -1, kernel)
            cv2.imwrite(f"{base_name}_sharpen.jpg", sharpened_image)

    # Noise or Blur
    def noise_blur():
        if random.choice(["noise", "blur"]) == "noise":
            stddev = random.uniform(30, 70)
            noise = np.random.normal(0, stddev, image.shape).astype(np.int16)
            noisy_image = image.astype(np.int16) + noise
            noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
            cv2.imwrite(f"{base_name}_noise.jpg", noisy_image)
        else:
            kernel_size = random.choice([(5, 5), (9, 9), (13, 13)])
            blurred_image = cv2.GaussianBlur(image, kernel_size, 0)
            cv2.imwrite(f"{base_name}_blur.jpg", blurred_image)

    selected = random.sample([rotation_mask, contrast_sharpen, noise_blur], 2)
    for augmentation in selected:
        augmentation()
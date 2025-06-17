# Face Recognition

本專案是一個完整的人臉辨識流程，涵蓋資料蒐集、預處理、特徵擷取、資料增強、模型訓練與推論。適合用於辨識多位指定人物，並可自動化處理影像與影片來源。

## 安裝需求

1. 安裝 Python 3.7+
2. 安裝必要套件：
   ```sh
   pip install -r requirements.txt
   ```
3. 下載 deploy.prototxt 與 res10_300x300_ssd_iter_140000.caffemodel 至專案根目錄。

## 使用流程

### 1. 下載圖片

可使用 `bing_image_downloader` 或其他工具下載指定人物頭像，參考 runs.ipynb 範例。

### 2. 擷取臉部與去除相似影像

使用 `collect.capture` 擷取臉部，並用 `collect.filter_similar_images` 移除重複或相似影像。

### 3. HOG特徵擷取

將臉部影像轉換為HOG特徵，並可視覺化儲存。

### 4. 影像增強

利用 `collect.augment_image` 對影像進行旋轉、遮罩、亮度對比、銳化、雜訊、模糊等隨機增強。

### 5. 模型訓練

執行：
```sh
python train.py faces
```
訓練完成後，模型與報告會儲存於 `train/` 目錄。

### 6. 模型推論

執行：
```sh
python detect.py train/1/model.pkl --source "test.mp4"
```
或使用攝影機：
```sh
python detect.py train/1/model.pkl --source 0
```

## 參考

- [OpenCV DNN Face Detector](https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector)
- [scikit-image HOG](https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_hog.html)
- [scikit-learn SVM](https://scikit-learn.org/stable/modules/svm.html)

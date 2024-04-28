import cv2
from paddleocr import PaddleOCR, draw_ocr
from collections import deque
from statistics import mode

# 创建 OCR 对象
ocr = PaddleOCR(use_angle_cls=True, lang='ch')  # 启用角度分类

def preprocess_image(frame):
    # 转为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 提高对比度
    enhanced = cv2.equalizeHist(gray)
    # 应用高斯模糊
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    # 二值化
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary

# 打开视频文件
cap = cv2.VideoCapture('D:/workspaceTech/ultralytics/OCR/4.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        break  # 如果没有获取到帧，说明视频已经结束，退出循环

    processed_frame = preprocess_image(frame)
    ocr_results = ocr.ocr(processed_frame, cls=True)

    # 输出识别结果
    print(ocr_results)  # 打印文本
    # 对每一个识别结果，在帧上绘制文本
    # for result in ocr_results:
    #     text = result[1][0]  # 提取识别到的文本
    #     position = result[0][0]  # 文本框的位置
    #     # 在帧上绘制识别的文本
    #     cv2.putText(frame, text, (int(position[0]), int(position[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36, 255, 12), 2)

    # 显示带有 OCR 文本的图像
    cv2.imshow('OCR Result', processed_frame)

    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

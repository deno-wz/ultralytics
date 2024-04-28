import psutil
import GPUtil
from ultralytics import YOLO
from tabulate import tabulate
images_path = './tests/huaxiangroad/4.mp4'
model = YOLO('./weights/obbText/best_obb_text.pt')
# model = YOLO("yolov8n.pt")
# res = model(images_path,save=True,stream=True)
res = model(images_path,save=False,stream=True)

# 使用字典初始化计数器
defect_counts = {0: 0, 1: 0, 2: 0}
class_names = {0: "RoadText", 1: "RobotText", 2: "TimeText"}

for r in res:
    if r.obb is not None:
        for obb in r.obb:
            class_id = int(obb.cls.item())
            if class_id in defect_counts:
                defect_counts[class_id] += 1

# 输出结果
for class_id, count in defect_counts.items():
    print(f"检测到 {class_names[class_id]}: {count}个")

def print_gpu_usage():
    gpus = GPUtil.getGPUs()
    list_gpus = []
    for gpu in gpus:
        # 获取GPU核心的使用率
        gpu_usage = f"{gpu.load*100}%"
        # 获取GPU显存的使用率
        memory_usage = f"{gpu.memoryUsed}/{gpu.memoryTotal} MB ({gpu.memoryUtil*100}%)"
        # 获取GPU的温度（如果有传感器数据）
        gpu_temp = f"{gpu.temperature} C"
        list_gpus.append((
            gpu.id, gpu.name, gpu_usage, memory_usage, gpu_temp
        ))

    print(tabulate(list_gpus, headers=("id", "name", "GPU Util", "Memory Usage", "Temperature")))

print_gpu_usage()
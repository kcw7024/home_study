from unittest import result
import torch


model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

img = 'D:\home_study\Yolo/test3.jpg'

results = model(img)
results.print()
'''
image 1/1: 480x640 1 cup, 6 bowls, 2 broccolis, 1 pizza, 1 cake, 1 dining table
Speed: 47.0ms pre-process, 228.0ms inference, 3.0ms NMS per image at shape (1, 3, 480, 640)

'''
results.xyxy[0]
results.show()
results.pandas().xyxy[0]

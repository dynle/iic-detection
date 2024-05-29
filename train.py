# ! Keep showing the predefined classes

from ultralytics import YOLO

# Load a model
# model = YOLO('yolov8n.yaml')  # build a new model from YAML
model = YOLO('./weights/yolov8n.pt')  # load a pretrained model (recommended for training)
# model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

# Train the model
results = model.train(data='./dataset/data.yaml', epochs=20, imgsz=640, device='mps')

# Save the model
model.save('./weights/yolov8n-iic-trained.pt')
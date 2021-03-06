import torch
import argparse
import cv2
import time
import detect_utils
from PIL import Image
from detect_utils import get_model

# construct the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='path to input image/video')
parser.add_argument('-m', '--min-size', dest='min_size', default=800,
                    help='minimum input size for the FasterRCNN network')
args = vars(parser.parse_args())

# download or load the model from disk
model = get_model(num_classes=2)
model.load_state_dict(torch.load("model_v8"))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

image = Image.open(args['input'])
model.eval().to(device)
# get the start time
start_time = time.time()
boxes, classes, labels, score = detect_utils.predict(image, model, device, 0.8)
image = detect_utils.draw_boxes(boxes, classes, labels, score, image)
# get the end time
end_time = time.time()
# get the fps
fps = 1 / (end_time - start_time)
cv2.imshow('Image', image)
print(fps)
save_name = f"{args['input'].split('/')[-1].split('.')[0]}_{args['min_size']}"
cv2.imwrite(f"outputs/{save_name}.jpg", image)
cv2.waitKey(0)
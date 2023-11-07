import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2 


from segment_anything import SamPredictor, sam_model_registry


image = cv2.imread('./download.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)
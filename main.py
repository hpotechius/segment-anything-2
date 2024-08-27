import torch
import cv2
import numpy as np
import os
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


#checkpoint = "./checkpoints/sam2_hiera_large.pt"
#model_cfg = "sam2_hiera_l.yaml"
checkpoint = "./checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"
#predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint, device="cpu"))
predictor = SAM2AutomaticMaskGenerator(build_sam2(model_cfg, checkpoint, device="cpu"),
                                       points_per_side=64,
                                       points_per_batch=128,
                                       pred_iou_thresh=0.7,
                                       stability_score_thresh=0.92,
                                       stability_score_offset=0.7,
                                       crop_n_layers=1,
                                       box_nms_thresh=0.7)

image = cv2.imread("assets/examples_MIT/warehouse_0020.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
masks = predictor.generate(image)
print(len(masks))

# Assuming masks is a numpy array of shape (Y, 256, 256)
height, width, _ = image.shape

# Create a blank image to hold the combined masks
combined_image = np.zeros((height, width, 3), dtype=np.uint8)

# Generate a unique color for each mask
colors = [tuple(np.random.randint(100, 255, 3).tolist()) for _ in range(len(masks))]

# Overlay each mask onto the combined image with the corresponding color
for i in range(len(masks)):
    mask = masks[i]["segmentation"]
    color = colors[i]
    colored_mask = np.zeros((height, width, 3), dtype=np.uint8)
    for c in range(3):
        colored_mask[:, :, c] = mask * color[c]
    combined_image = cv2.addWeighted(combined_image, 1.0, colored_mask, 0.5, 0)

# Save the resulting image
cv2.imwrite('assets/results/combined_masks.png', combined_image)
print("Combined image saved as 'combined_masks2.png'")

#with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):

    # predictor.set_image(image)
    # masks, _, _ = predictor.predict()

    # # Assuming masks is a numpy array of shape (Y, 256, 256)
    # Y, height, width = masks.shape

    # # Create a blank image to hold the combined masks
    # combined_image = np.zeros((height, width, 3), dtype=np.uint8)

    # # Generate a unique color for each mask
    # colors = [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(Y)]

    # # Overlay each mask onto the combined image with the corresponding color
    # for i in range(Y):
    #     mask = masks[i]
    #     color = colors[i]
    #     colored_mask = np.zeros((height, width, 3), dtype=np.uint8)
    #     for c in range(3):
    #         colored_mask[:, :, c] = mask * color[c]
    #     combined_image = cv2.addWeighted(combined_image, 1.0, colored_mask, 0.5, 0)

    # # Save the resulting image
    # cv2.imwrite('assets/results/combined_masks.png', combined_image)
    # print("Combined image saved as 'combined_masks.png'")
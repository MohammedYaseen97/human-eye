import sys
sys.path.append("omni_parser")

from omni_parser.util.utils import check_ocr_box, get_yolo_model, get_caption_model_processor, get_som_labeled_img
import torch
from PIL import Image
import io
import base64
from typing import Optional

DEVICE = torch.device('cuda')

yolo_model = get_yolo_model(model_path='omni_parser/weights/icon_detect/model.pt')
caption_model_processor = get_caption_model_processor(model_name="florence2", model_name_or_path="omni_parser/weights/icon_caption_florence")

def process(
    image_input,
    box_threshold = 0.05,
    iou_threshold = 0.1,
    use_paddleocr = True,
    imgsz = 640
):

    image_save_path = 'omni_parser/imgs/saved_image_demo.png'
    image_input.save(image_save_path)
    image = Image.open(image_save_path)
    box_overlay_ratio = image.size[0] / 3200
    draw_bbox_config = {
        'text_scale': 0.8 * box_overlay_ratio,
        'text_thickness': max(int(2 * box_overlay_ratio), 1),
        'text_padding': max(int(3 * box_overlay_ratio), 1),
        'thickness': max(int(3 * box_overlay_ratio), 1),
    }

    ocr_bbox_rslt, is_goal_filtered = check_ocr_box(image_save_path, display_img = False, output_bb_format='xyxy', goal_filtering=None, easyocr_args={'paragraph': False, 'text_threshold':0.9}, use_paddleocr=use_paddleocr)
    text, ocr_bbox = ocr_bbox_rslt
    dino_labled_img, label_coordinates, parsed_content_list = get_som_labeled_img(image_save_path, yolo_model, BOX_TRESHOLD = box_threshold, output_coord_in_ratio=True, ocr_bbox=ocr_bbox,draw_bbox_config=draw_bbox_config, caption_model_processor=caption_model_processor, ocr_text=text,iou_threshold=iou_threshold, imgsz=imgsz,)  
    image = Image.open(io.BytesIO(base64.b64decode(dino_labled_img)))
    print('finished processing..')
    # parsed_content_list = '\n'.join([f'icon {i}: ' + str(v) for i,v in enumerate(parsed_content_list)])
    
    return image, parsed_content_list

# Display the image using matplotlib
import matplotlib.pyplot as plt

def display_image(image):
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    plt.axis('off')
    plt.show()

image = Image.open('ui_cropped.jpg')

parsed_image, parsed_content_list = process(image)
display_image(parsed_image)

from ui_attention_predictor import UIAttentionPredictor, Platform
import json

# Initialize predictor
attention_predictor = UIAttentionPredictor(
    platform=Platform.ANDROID,
    tech_savviness=9
)

elements_data = [
    # {
    #     "type": "button",
    #     "text": "Settings",
    #     "bounds": {
    #         "x1": 0.1,  # These are normalized coordinates (0-1)
    #         "y1": 0.1,
    #         "x2": 0.2,
    #         "y2": 0.2
    #     }
    # },
    # {
    #     "type": "icon",
    #     "text": "menu",
    #     "bounds": {
    #         "x1": 0.8,
    #         "y1": 0.1,
    #         "x2": 0.9,
    #         "y2": 0.2
    #     }
    # }
    {
        "type": item["type"],
        "text": item["content"],
        "bounds": {
            "x1": min(item["bbox"][0], item["bbox"][2]),
            "x2": max(item["bbox"][0], item["bbox"][2]),
            "y1": min(item["bbox"][1], item["bbox"][3]),
            "y2": max(item["bbox"][1], item["bbox"][3])
        }
    }
    for item in parsed_content_list
]

task_description = "Find the home button"
image = Image.open("ui_cropped.jpg")

# Get prediction
result = attention_predictor.predict_attention(
    ui_image=image,
    task=task_description,
    elements_data=elements_data
)

print(json.dumps(result, indent=4))
attention_predictor.visualize_attention(result, Image.open("ui_cropped.jpg"), alpha=0.9)
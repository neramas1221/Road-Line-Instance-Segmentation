import matplotlib.pyplot as plt
import torch
import json
import numpy as np
import cv2
import utils
import torchvision
import random
import torchvision.transforms.functional as TF
from torchvision.io import read_image
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from engine import train_one_epoch, evaluate
    

img_path_train = "road-lane-instance-segmentation/versions/1/road_lane_instance_segmentation/train2017/"
json_path_train =  "road-lane-instance-segmentation/versions/1/road_lane_instance_segmentation/annotations/instances_train2017_fixed.json"
img_path_test = "road-lane-instance-segmentation/versions/1/road_lane_instance_segmentation/val2017/"
json_path_test = "road-lane-instance-segmentation/versions/1/road_lane_instance_segmentation/annotations/instances_val2017.json"
num_classes = 8
num_epochs = 1
lr = 0.0001
best_score = 0.0
patience = 20
epochs_without_improvement = 0


class RoadLineDatasetTrain(torch.utils.data.Dataset):
    def __init__(self, image_path, json_path, is_train=False):
        super().__init__()
        self.image_path = image_path
        with open(json_path) as f:
            self.data = json.load(f)
        self.is_train = is_train


    def polygon_to_mask_cv2(self, segmentation, height, width):
        mask = np.zeros((height, width), dtype=np.uint8)

        pts = np.array(segmentation, dtype=np.int32).reshape(-1, 2)

        cv2.fillPoly(mask, [pts], 1)

        return mask
    
    def custom_transforms(self, image, mask, boxes):
            _, H, W = image.shape 

            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

                x_min_flipped = W - boxes[:, 2]  
                x_max_flipped = W - boxes[:, 0]  
                boxes[:, 0] = x_min_flipped
                boxes[:, 2] = x_max_flipped
                    
            return image, mask, boxes

    def __getitem__(self, idx):
        img_path = self.image_path + self.data["images"][idx]["file_name"]
        torch_image = read_image(img_path).float() / 255.0
        labels = []
        boxes = []
        masks = []

        for i in self.data["annotations"]:
            if i["image_id"] != idx:
                continue

            bbox = i['bbox']
            segmentation_points = i['segmentation'][0]
            segmentation_points_int = [int(p) for p in segmentation_points]

            temp_mask = self.polygon_to_mask_cv2(segmentation_points_int, self.data["images"][idx]["height"], self.data["images"][idx]["width"])
            masks.append(torch.from_numpy(temp_mask).to(torch.uint8))
            labels.append(i["category_id"])

            x_min, y_min, width, height = bbox
            x_max = x_min + width
            y_max = y_min + height

            boxes.append([x_min, y_min, x_max, y_max])

        masks_tensor = torch.stack(masks)
        areas = (masks_tensor > 0).sum(dim=(1, 2))
        boxes = torch.tensor(boxes, dtype=torch.float32)
        if self.is_train:
            torch_image, masks_tensor, boxes = self.custom_transforms(torch_image, masks_tensor, boxes)

        target = {
            "boxes": tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(torch_image)),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "masks": masks_tensor,
            "image_id": idx,
            "area": areas,
            "iscrowd": torch.zeros((len(labels),), dtype=torch.int64)
        }
        return torch_image, target
    

    def __len__(self):
        return len(self.data["images"])


def get_model_instance_segmentation(num_classes):

    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256

    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )

    return model


def filter_preds_by_confidence(pred, threshold=0.7):

    keep = pred['scores'] > threshold
    
    filtered_pred = {}
    for k, v in pred.items():

        if torch.is_tensor(v):
            filtered_pred[k] = v[keep]
        else:

            filtered_pred[k] = [item for item, keep_item in zip(v, keep) if keep_item]
    
    return filtered_pred



device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


dataset = RoadLineDatasetTrain(img_path_train, json_path_train, is_train=True)
dataset_test = RoadLineDatasetTrain(img_path_test, json_path_test) 


data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=utils.collate_fn
)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=1,
    shuffle=False,
    collate_fn=utils.collate_fn
)


model = get_model_instance_segmentation(num_classes)

model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(
    params,
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0005
)

optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.001, betas=[0.9, 0.999])

lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=3,
    gamma=0.1
)

image = read_image("road-lane-instance-segmentation/versions/1/road_lane_instance_segmentation/val2017/2023-06-15_14-06-50-front_mp4_1020_jpg.rf.9f6036a9bd175533631488de57800dec.jpg")


for epoch in range(num_epochs):
    model.train()

    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)

    lr_scheduler.step()

    coco_evaluator = evaluate(model, data_loader_test, device=device)
    coco_score = coco_evaluator.coco_eval['segm'].stats[0]

    model.eval()
    with torch.no_grad():
        x = image.float() / 255.0

        x = x[:3, ...].to(device)
        predictions = model([x, ])
        pred = predictions[0]


    val_loss_total = 0.0
    model.train()

    with torch.no_grad():
        for images, targets in data_loader_test:
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())

            val_loss_total += losses.item()

    average_val_loss = val_loss_total / len(data_loader_test)
    print("Validation loss:", average_val_loss)


    pred = filter_preds_by_confidence(pred, threshold=0.7)

    image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
    image = image[:3, ...]
    pred_labels = [f"{dataset_test.data['categories'][label]['name']}: {score:.3f}" for label, score in zip(pred["labels"], pred["scores"])]
    pred_boxes = pred["boxes"].long()
    output_image = draw_bounding_boxes(image, pred_boxes, pred_labels, colors="red")

    masks = (pred["masks"] > 0.7).squeeze(1)
    output_image = draw_segmentation_masks(output_image, masks, alpha=0.5, colors="blue")


    plt.figure(figsize=(12, 12))
    plt.title(f"Results at {epoch}")
    plt.imshow(output_image.permute(1, 2, 0))
    plt.savefig(f"res_during_runs/adam/{epoch}.png")
    if coco_score > best_score:
        best_score = coco_score
        epochs_without_improvement = 0

        torch.save(model.state_dict(), "best_model.pth")
    else:
        epochs_without_improvement += 1
        print(f"No improvement. {epochs_without_improvement} / {patience} epochs without improvement.")

    if epochs_without_improvement >= patience:
        print("Early stopping triggered.")
        break

model.load_state_dict(torch.load("best_model.pth", weights_only=True))
model.eval()


images = [
    "road-lane-instance-segmentation/versions/1/road_lane_instance_segmentation/val2017/2023-06-15_14-06-50-front_mp4_1020_jpg.rf.9f6036a9bd175533631488de57800dec.jpg",
    "road-lane-instance-segmentation/versions/1/road_lane_instance_segmentation/val2017/20230619164657_247484_TS_1920_jpg.rf.8e966018177846c51eed5baa1f9f794b.jpg",
    "road-lane-instance-segmentation/versions/1/road_lane_instance_segmentation/val2017/Dash_Cam_Owners_Indonesia_498_June_2023_mp4_17400_jpg.rf.e107d1ab56fbc4bb3589b6169bfc1c88.jpg",
    "road-lane-instance-segmentation/versions/1/road_lane_instance_segmentation/val2017/Malaysia_Dash_Cam_Video_Compilation_14_Malaysian_Dash_Cam_Owners_mp4_1740_jpg.rf.9fe610f2fa9c63005992b81ee4a8b63a.jpg",
    "road-lane-instance-segmentation/versions/1/road_lane_instance_segmentation/val2017/MOVA5101_avi_1920_jpg.rf.1888b4d9e504f87748824cbbd1c0e495.jpg"
]

for i, img_path in enumerate(images):
    image = read_image(img_path)
    with torch.no_grad():
        x = image.float() / 255.0
        x = x[:3, ...].to(device)
        predictions = model([x, ])
        pred = predictions[0]


    pred = filter_preds_by_confidence(pred, threshold=0.7)

    image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
    image = image[:3, ...]
    pred_labels = [f"{dataset_test.data['categories'][label]['name']}: {score:.3f}" for label, score in zip(pred["labels"], pred["scores"])]
    pred_boxes = pred["boxes"].long()
    output_image = draw_bounding_boxes(image, pred_boxes, pred_labels, colors="red")

    masks = (pred["masks"] > 0.7).squeeze(1)
    output_image = draw_segmentation_masks(output_image, masks, alpha=0.5, colors="blue")


    plt.figure(figsize=(12, 12))
    plt.title(f"Results at {i}")
    plt.imshow(output_image.permute(1, 2, 0))
    plt.savefig(f"res_adam_{i}.png")

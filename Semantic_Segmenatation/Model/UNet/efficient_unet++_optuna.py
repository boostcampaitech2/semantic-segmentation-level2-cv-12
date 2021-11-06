import optuna
import segmentation_models_pytorch as smp
from typing import Optional
from albumentations.pytorch import ToTensorV2
import albumentations as A
import albumentations.augmentations.transforms as AF
from pycocotools.coco import COCO
import pandas as pd
import numpy as np
import cv2
from utils import label_accuracy_score, add_hist
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch
import os
import random
import time
import json
import warnings
warnings.filterwarnings('ignore')


# 전처리를 위한 라이브러리


plt.rcParams['axes.grid'] = False

print('pytorch version: {}'.format(torch.__version__))
print('GPU 사용 가능 여부: {}'.format(torch.cuda.is_available()))

print(torch.cuda.get_device_name(0))
print(torch.cuda.device_count())

# GPU 사용 가능 여부에 따라 device 정보 저장
device = "cuda" if torch.cuda.is_available() else "cpu"

batch_size = 4   # Mini-batch size
num_epochs = 200
learning_rate = 0.0001


# seed 고정
random_seed = 21
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

dataset_path = '../input/data'
anns_file_path = dataset_path + '/' + 'train_all.json'

# Read annotations
with open(anns_file_path, 'r') as f:
    dataset = json.loads(f.read())

categories = dataset['categories']
anns = dataset['annotations']
imgs = dataset['images']
nr_cats = len(categories)
nr_annotations = len(anns)
nr_images = len(imgs)

# Load categories and super categories
cat_names = []
super_cat_names = []
super_cat_ids = {}
super_cat_last_name = ''
nr_super_cats = 0
for cat_it in categories:
    cat_names.append(cat_it['name'])
    super_cat_name = cat_it['supercategory']
    # Adding new supercat
    if super_cat_name != super_cat_last_name:
        super_cat_names.append(super_cat_name)
        super_cat_ids[super_cat_name] = nr_super_cats
        super_cat_last_name = super_cat_name
        nr_super_cats += 1

print('Number of super categories:', nr_super_cats)
print('Number of categories:', nr_cats)
print('Number of annotations:', nr_annotations)
print('Number of images:', nr_images)

# Count annotations
cat_histogram = np.zeros(nr_cats, dtype=int)
for ann in anns:
    cat_histogram[ann['category_id']-1] += 1

# Convert to DataFrame
df = pd.DataFrame(
    {'Categories': cat_names, 'Number of annotations': cat_histogram})
df = df.sort_values('Number of annotations', 0, False)


# category labeling
sorted_temp_df = df.sort_index()

# background = 0 에 해당되는 label 추가 후 기존들을 모두 label + 1 로 설정
sorted_df = pd.DataFrame(["Backgroud"], columns=["Categories"])
sorted_df = sorted_df.append(sorted_temp_df, ignore_index=True)


# ## 데이터 전처리 함수 정의 (Dataset)

category_names = list(sorted_df.Categories)


def get_classname(classID, cats):
    for i in range(len(cats)):
        if cats[i]['id'] == classID:
            return cats[i]['name']
    return "None"


class CustomDataLoader(Dataset):
    """COCO format"""

    def __init__(self, data_dir, mode='train', transform=None):
        super().__init__()
        self.mode = mode
        self.transform = transform
        self.coco = COCO(data_dir)

    def __getitem__(self, index: int):
        # dataset이 index되어 list처럼 동작
        image_id = self.coco.getImgIds(imgIds=index)
        image_infos = self.coco.loadImgs(image_id)[0]

        # cv2 를 활용하여 image 불러오기
        images = cv2.imread(os.path.join(
            dataset_path, image_infos['file_name']))
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB).astype(np.float32)
        images /= 255.0

        if (self.mode in ('train', 'val')):
            ann_ids = self.coco.getAnnIds(imgIds=image_infos['id'])
            anns = self.coco.loadAnns(ann_ids)

            # Load the categories in a variable
            cat_ids = self.coco.getCatIds()
            cats = self.coco.loadCats(cat_ids)

            # masks : size가 (height x width)인 2D
            # 각각의 pixel 값에는 "category id" 할당
            # Background = 0
            masks = np.zeros((image_infos["height"], image_infos["width"]))
            # General trash = 1, ... , Cigarette = 10
            anns = sorted(anns, key=lambda idx: idx['area'], reverse=True)

            for i in range(len(anns)):
                className = get_classname(anns[i]['category_id'], cats)
                pixel_value = category_names.index(className)
                masks[self.coco.annToMask(anns[i]) == 1] = pixel_value
            masks = masks.astype(np.int8)

            # transform -> albumentations 라이브러리 활용
            if self.transform is not None:
                transformed = self.transform(image=images, mask=masks)
                images = transformed["image"]
                masks = transformed["mask"]
            return images, masks, image_infos

        if self.mode == 'test':
            # transform -> albumentations 라이브러리 활용
            if self.transform is not None:
                transformed = self.transform(image=images)
                images = transformed["image"]
            return images, image_infos

    def __len__(self) -> int:
        # 전체 dataset의 size를 return
        return len(self.coco.getImgIds())


# ## Dataset 정의 및 DataLoader 할당


# train.json / validation.json / test.json 디렉토리 설정
train_path = dataset_path + '/train_modified.json'
val_path = dataset_path + '/val_modified.json'

# collate_fn needs for batch


def collate_fn(batch):
    return tuple(zip(*batch))


def train(num_epochs, model, data_loader, val_loader, criterion, optimizer, scheduler, val_every, device):
    print(f'Start training..')
    n_class = 11
    best_mIoU = 0.0
    # wandb.init(project='semantic_seg',entity='bcaitech_cv2',name='efficient_unetplusx2_SmoothFocalLoss')

    for epoch in range(num_epochs):
        model.train()
        # lr = scheduler.get_last_lr()[0]
        # wandb.log({"learning_rate":lr})

        hist = np.zeros((n_class, n_class))
        for step, (images, masks, _) in enumerate(data_loader):
            images = torch.stack(images)
            masks = torch.stack(masks).long()

            # gpu 연산을 위해 device 할당
            images, masks = images.to(device), masks.to(device)

            # device 할당
            model = model.to(device)

            # inference
            outputs = model(images)
            # loss 계산 (cross entropy loss)
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            masks = masks.detach().cpu().numpy()

            hist = add_hist(hist, masks, outputs, n_class=n_class)
            acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)

            # step 주기에 따른 loss 출력
            if (step + 1) % 50 == 0:
                print(
                    f'Epoch [{epoch+1}/{num_epochs}], Step [{step+1}/{len(data_loader)}],  Loss: {round(loss.item(),4)}, mIoU: {round(mIoU,4)}')
                # wandb.log({"train/loss":loss})

        # validation 주기에 따른 loss 출력 및 best model 저장
        if (epoch + 1) % val_every == 0:
            avrg_loss, best_mIoU = validation(
                epoch + 1, model, val_loader, criterion, device, best_mIoU)

    return best_mIoU


def validation(epoch, model, data_loader, criterion, device, best_mIoU):
    print(f'Start validation #{epoch}')
    model.eval()

    with torch.no_grad():
        n_class = 11
        total_loss = 0
        cnt = 0

        hist = np.zeros((n_class, n_class))
        for step, (images, masks, _) in enumerate(data_loader):

            images = torch.stack(images)
            masks = torch.stack(masks).long()

            images, masks = images.to(device), masks.to(device)

            # device 할당
            model = model.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss
            cnt += 1

            outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            masks = masks.detach().cpu().numpy()

            hist = add_hist(hist, masks, outputs, n_class=n_class)

        acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)
        IoU_by_class = [{classes: round(IoU, 4)} for IoU, classes in zip(
            IoU, sorted_df['Categories'])]
        avrg_loss = total_loss / cnt
        print(f'Validation #{epoch}  Average Loss: {round(avrg_loss.item(), 4)}, Accuracy : {round(acc, 4)}, \
                mIoU: {round(mIoU, 4)}')
        # wandb.log({"val/aAcc": acc,"val/mIoU":mIoU})
        print(f'IoU by class : {IoU_by_class}')
        if mIoU > best_mIoU:
            print(f"Best performance at epoch: {epoch}")
            best_mIoU = mIoU
    return avrg_loss, best_mIoU


def label_to_smooth_label(
    labels: torch.Tensor,
    num_classes: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    smoothing=0.0,
) -> torch.Tensor:
    r"""Convert an integer label x-D tensor to a one-hot (x+1)-D tensor.

    Args:
        labels: tensor with labels of shape :math:`(N, *)`, where N is batch size.
          Each value is an integer representing correct classification.
        num_classes: number of classes in labels.
        device: the desired device of returned tensor.
        dtype: the desired data type of returned tensor.

    Returns:
        the labels in one hot tensor of shape :math:`(N, C, *)`,

    Examples:
        >>> labels = torch.LongTensor([
                [[0, 1], 
                [2, 0]]
            ])
        >>> one_hot(labels, num_classes=3)
        tensor([[[[1.0000e+00, 1.0000e-06],
                  [1.0000e-06, 1.0000e+00]],

                 [[1.0000e-06, 1.0000e+00],
                  [1.0000e-06, 1.0000e-06]],

                 [[1.0000e-06, 1.0000e-06],
                  [1.0000e+00, 1.0000e-06]]]])

    """
    shape = labels.shape
    confidence = 1.0 - smoothing

    one_hot = torch.zeros((shape[0], num_classes) + shape[1:],
                          device=device, dtype=dtype) + (smoothing/(num_classes-1))
    one_hot = one_hot.scatter_(1, labels.unsqueeze(1), confidence)

    return one_hot


# https://github.com/zhezh/focalloss/blob/master/focalloss.py
def focal_loss(input, target, alpha, gamma, reduction, smoothing):
    r"""Criterion that computes Focal loss.

    According to :cite:`lin2018focal`, the Focal loss is computed as follows:

    .. math::

        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)

    Where:
       - :math:`p_t` is the model's estimated probability for each class.

    Args:
        input: logits tensor with shape :math:`(N, C, *)` where C = number of classes.
        target: labels tensor with shape :math:`(N, *)` where each value is :math:`0 ≤ targets[i] ≤ C−1`.
        alpha: Weighting factor :math:`\alpha \in [0, 1]`.
        gamma: Focusing parameter :math:`\gamma >= 0`.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed.

    Return:
        the computed loss.

    Example:
        >>> N = 5  # num_classes
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = focal_loss(input, target, alpha=0.5, gamma=2.0, reduction='mean')
        >>> output.backward()
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not len(input.shape) >= 2:
        raise ValueError(
            f"Invalid input shape, we expect BxCx*. Got: {input.shape}")

    if input.size(0) != target.size(0):
        raise ValueError(
            f'Expected input batch_size ({input.size(0)}) to match target batch_size ({target.size(0)}).')

    # input : (B, C, H, W)
    n = input.size(0)  # B

    # out_sie : (B, H, W)
    out_size = (n,) + input.size()[2:]

    # input : (B, C, H, W)
    # target : (B, H, W)
    if target.size()[1:] != input.size()[2:]:
        raise ValueError(
            f'Expected target size {out_size}, got {target.size()}')

    if not input.device == target.device:
        raise ValueError(
            f"input and target must be in the same device. Got: {input.device} and {target.device}")

    if isinstance(alpha, float):
        pass
    elif isinstance(alpha, np.ndarray):
        alpha = torch.from_numpy(alpha)
        # alpha : (B, C, H, W)
        alpha = alpha.view(-1, len(alpha), 1, 1).expand_as(input)
    elif isinstance(alpha, torch.Tensor):
        # alpha : (B, C, H, W)
        alpha = alpha.view(-1, len(alpha), 1, 1).expand_as(input)

    # compute softmax over the classes axis
    # input_soft : (B, C, H, W)
    input_soft = F.softmax(input, dim=1) + 1e-6

    # create the labels one hot tensor
    # target_one_hot : (B, C, H, W)
    target_one_hot = label_to_smooth_label(target.long(
    ), num_classes=input.shape[1], device=input.device, dtype=input.dtype, smoothing=smoothing)
    # compute the actual focal loss
    weight = torch.pow(1.0 - input_soft, gamma)

    # alpha, weight, input_soft : (B, C, H, W)
    # focal : (B, C, H, W)
    focal = -alpha * weight * torch.log(input_soft)

    # loss_tmp : (B, H, W)
    loss_tmp = torch.sum(target_one_hot * focal, dim=1)

    if reduction == 'none':
        # loss : (B, H, W)
        loss = loss_tmp
    elif reduction == 'mean':
        # loss : scalar
        loss = torch.mean(loss_tmp)
    elif reduction == 'sum':
        # loss : scalar
        loss = torch.sum(loss_tmp)
    else:
        raise NotImplementedError(f"Invalid reduction mode: {reduction}")
    return loss


class SmoothFocalLoss(nn.Module):
    r"""Criterion that computes Focal loss.

    According to :cite:`lin2018focal`, the Focal loss is computed as follows:

    .. math:

        FL(p_t) = -alpha_t(1 - p_t)^{gamma}, log(p_t)

    Where:
       - :math:`p_t` is the model's estimated probability for each class.

    Args:
        alpha: Weighting factor :math:`\alpha \in [0, 1]`.
        gamma: Focusing parameter :math:`\gamma >= 0`.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed.

    Shape:
        - Input: :math:`(N, C, *)` where C = number of classes.
        - Target: :math:`(N, *)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.

    Example:
        >>> N = 5  # num_classes
        >>> kwargs = {"alpha": 0.5, "gamma": 2.0, "reduction": 'mean'}
        >>> criterion = FocalLoss(**kwargs)
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = criterion(input, target)
        >>> output.backward()
    """

    def __init__(self, alpha, gamma=2.0, reduction='mean', smoothing=0.2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.smoothing = smoothing

    def forward(self, input, target):
        return focal_loss(input, target, self.alpha, self.gamma, self.reduction, self.smoothing)


def objective(trial):
    num_epochs = 100

    model = smp.UnetPlusPlus(
        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_name="efficientnet-b0",
        # use `imagenet` pre-trained weights for encoder initialization
        encoder_weights="imagenet",
        # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        in_channels=3,
        # model output channels (number of classes in your dataset)
        classes=11,
    )

    p_flip = float(trial.suggest_categorical('p_flip', ["0", "0.3"]))
    p_blur = float(trial.suggest_categorical('p_blur', ["0", "0.3"]))
    p_crop = float(trial.suggest_categorical('p_crop', ["0", "0.3"]))
    p_rgb = float(trial.suggest_categorical('p_rgb', ["0", "0.3"]))

    focal_factor = trial.suggest_float('focal_factor', 0, 0.5)
    smooting_factor = trial.suggest_float('smooting_factor', 0, 0.5)

    train_transform = A.Compose([
        AF.Flip(always_apply=False, p=p_flip),
        AF.RGBShift(r_shift_limit=20, g_shift_limit=20,
                    b_shift_limit=20, always_apply=False, p=p_rgb),
        AF.Blur(always_apply=False, p=p_blur, blur_limit=(3, 7)),
        A.augmentations.crops.transforms.RandomSizedCrop(always_apply=False, p=p_crop, min_max_height=(
            128, 512), height=512, width=512, w2h_ratio=1.0, interpolation=0),
        ToTensorV2()
    ])
    val_transform = A.Compose([
        ToTensorV2()
    ])

    train_dataset = CustomDataLoader(
        data_dir=train_path, mode='train', transform=train_transform)
    val_dataset = CustomDataLoader(
        data_dir=val_path, mode='val', transform=val_transform)

    # DataLoader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=4,
                                               collate_fn=collate_fn)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=4,
                                             collate_fn=collate_fn)

    criterion = SmoothFocalLoss(alpha=focal_factor, smoothing=smooting_factor)
    # Optimizer 정의
    optimizer = torch.optim.AdamW(
        params=model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50)

    saved_dir = './saved'
    val_every = 20
    device = "cuda" if torch.cuda.is_available() else "cpu"

    return train(num_epochs, model, train_loader, val_loader, criterion, optimizer, scheduler, saved_dir, val_every, device)


if __name__ == '__main__':
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)

    trial = study.best_trial

    print('best_mIoU: {}'.format(trial.value))
    print("Best hyperparameters: {}".format(trial.params))

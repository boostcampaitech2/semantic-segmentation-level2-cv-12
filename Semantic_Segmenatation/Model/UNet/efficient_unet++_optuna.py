from evaluation import SmoothFocalLoss
import optuna
import segmentation_models_pytorch as smp

from albumentations.pytorch import ToTensorV2
import albumentations as A
import albumentations.augmentations.transforms as AF
import pandas as pd
import numpy as np
from utils import label_accuracy_score, add_hist
import torch
import random
import warnings

from dataset import CustomDataLoader
from utils import add_hist, label_accuracy_score

warnings.filterwarnings('ignore')


# GPU 사용 가능 여부에 따라 device 정보 저장
device = "cuda" if torch.cuda.is_available() else "cpu"


# seed 고정
random_seed = 21
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

dataset_path = '/opt/ml/segmentation/input/data'
train_path = dataset_path + '/train_modified.json'
val_path = dataset_path + '/val_modified.json'

Categories = [
    "background", "general trash", "paper", "paper pack", "metal", "glass",
    "plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"
]


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
            IoU, Categories)]
        avrg_loss = total_loss / cnt
        print(f'Validation #{epoch}  Average Loss: {round(avrg_loss.item(), 4)}, Accuracy : {round(acc, 4)}, \
                mIoU: {round(mIoU, 4)}')
        # wandb.log({"val/aAcc": acc,"val/mIoU":mIoU})
        print(f'IoU by class : {IoU_by_class}')
        if mIoU > best_mIoU:
            print(f"Best performance at epoch: {epoch}")
            best_mIoU = mIoU
    return avrg_loss, best_mIoU


def objective(trial):
    num_epochs = 100
    batch_size = 4   # Mini-batch size
    learning_rate = 0.0001

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

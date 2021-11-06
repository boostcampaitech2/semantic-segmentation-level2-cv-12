import torch
import torch.nn as nn
import numpy as np
import random
import os

import albumentations as A
from albumentations.pytorch import ToTensorV2

from dataset import CustomDataLoader
import segmentation_models_pytorch as smp

# seed 고정
random_seed = 21
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

device = "cuda" if torch.cuda.is_available() else "cpu"

# train.json / validation.json / test.json 디렉토리 설정
dataset_path = '/opt/ml/segmentation/input/data'
train_path = dataset_path + '/train.json'
val_path = dataset_path + '/val.json'


# collate_fn needs for batch
def collate_fn(batch):
    return tuple(zip(*batch))


train_transform = A.Compose([
                            ToTensorV2()
                            ])

val_transform = A.Compose([
                          ToTensorV2()
                          ])

# train dataset
train_dataset = CustomDataLoader(
    data_dir=train_path, mode='train', transform=train_transform)

# validation dataset
val_dataset = CustomDataLoader(
    data_dir=val_path, mode='val', transform=val_transform)


def save_model(model, saved_dir, file_name='efficient_unet_best_model.pt'):
    check_point = {'net': model.state_dict()}
    output_path = os.path.join(saved_dir, file_name)
    torch.save(model, output_path)


def train(num_epochs, model, data_loader, val_loader, criterion, optimizer, saved_dir, val_every, device):
    print(f'Start training..')
    n_class = 11
    best_loss = 9999999

    for epoch in range(num_epochs):
        model.train()

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

            outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            masks = masks.detach().cpu().numpy()

            hist = add_hist(hist, masks, outputs, n_class=n_class)
            acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)

            # step 주기에 따른 loss 출력
            if (step + 1) % 25 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{step+1}/{len(train_loader)}], \
                        Loss: {round(loss.item(),4)}, mIoU: {round(mIoU,4)}')

        # validation 주기에 따른 loss 출력 및 best model 저장
        if (epoch + 1) % val_every == 0:
            avrg_loss = validation(
                epoch + 1, model, val_loader, criterion, device)
            if avrg_loss < best_loss:
                print(f"Best performance at epoch: {epoch + 1}")
                print(f"Save model in {saved_dir}")
                best_loss = avrg_loss
                save_model(model, saved_dir)


def validation(epoch, model, data_loader, criterion, device):
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
        print(f'IoU by class : {IoU_by_class}')

    return avrg_loss


if __name__ == '__main__':
    num_epochs = 20
    batch_size = 8
    learning_rate = 0.0001

    model = smp.Unet(
        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_name="efficientnet-b0",
        # use `imagenet` pre-trained weights for encoder initialization
        encoder_weights="imagenet",
        # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        in_channels=3,
        # model output channels (number of classes in your dataset)
        classes=11,
    )

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

    # Loss function 정의
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=learning_rate, weight_decay=1e-6)
    val_every = 1
    saved_dir = './saved'
    if not os.path.isdir(saved_dir):
        os.mkdir(saved_dir)

    train(num_epochs, model, train_loader, val_loader,
          criterion, optimizer, saved_dir, val_every, device)

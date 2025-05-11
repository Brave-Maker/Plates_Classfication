import os
import sys
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import resnet34


def main():
    # 1. 设备配置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using {} device.".format(device))

    # 2. 数据预处理
    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.RandomGrayscale(p=0.1),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    }

    # 3. 数据集与 DataLoader
    image_path = "./dataset/plates"
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)

    train_dataset = datasets.ImageFolder(
        root=os.path.join(image_path, "train"),
        transform=data_transform["train"]
    )
    val_dataset = datasets.ImageFolder(
        root=os.path.join(image_path, "val"),
        transform=data_transform["val"]
    )

    train_num = len(train_dataset)
    val_num = len(val_dataset)
    print(f"Using {train_num} images for training, {val_num} images for validation.")

    batch_size = 32
    num_workers = min(os.cpu_count(), batch_size, 8)
    print(f"Using {num_workers} dataloader workers per process.")

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    # 4. 保存类别索引
    cla_dict = {v: k for k, v in train_dataset.class_to_idx.items()}
    with open('class_indices.json', 'w') as f:
        json.dump(cla_dict, f, indent=2)

    # 5. 模型、损失函数与优化器
    net = resnet34()
    pretrained_path = "./resnet34-pre.pth"
    assert os.path.exists(pretrained_path), f"{pretrained_path} does not exist."
    net.load_state_dict(torch.load(pretrained_path, map_location='cpu'))

    # 修改最后的全连接层
    in_features = net.fc.in_features
    net.fc = nn.Linear(in_features, len(cla_dict))
    net.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-4)

    # 6. 训练设置
    epochs = 60
    best_acc = 0.0
    save_path = './resNet34.pth'

    # 用于记录指标
    train_losses = []
    val_accuracies = []

    # 7. 训练 & 验证循环
    for epoch in range(1, epochs + 1):
        # --- 训练 ---
        net.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, file=sys.stdout, desc=f"Train Epoch {epoch}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # --- 验证 ---
        net.eval()
        correct = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, file=sys.stdout, desc=f"Val   Epoch {epoch}/{epochs}"):
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()

        val_acc = correct / val_num
        val_accuracies.append(val_acc)

        print(f"[Epoch {epoch:02d}] Train Loss: {avg_train_loss:.4f}   Val Acc: {val_acc:.4f}")

        # 保存最优模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(net.state_dict(), save_path)

    print("Finished Training.")

    # 8. 绘图并保存
    epochs_range = list(range(1, epochs + 1))

    plt.figure()
    plt.plot(epochs_range, train_losses, marker='o')
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('train_loss.png')
    plt.close()

    plt.figure()
    plt.plot(epochs_range, val_accuracies, marker='o')
    plt.title('Validation Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.savefig('val_accuracy.png')
    plt.close()


if __name__ == '__main__':
    main()

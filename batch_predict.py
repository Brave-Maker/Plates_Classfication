import os
import json
import torch
from PIL import Image
from torchvision import transforms
from model import resnet34

def main():
    # 1. 设备配置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. 图像预处理
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # 3. 读取类别索引
    json_path = './class_indices.json'
    assert os.path.exists(json_path), f"file: '{json_path}' does not exist."
    with open(json_path, "r") as f:
        class_indict = json.load(f)  # e.g. {"0": "cleaned", "1": "dirty"}

    # 4. 定义字母映射
    # 假设索引 0 对应 cleaned -> 'C'，1 对应 dirty -> 'D'
    letter_map = {
        0: 'C',
        1: 'D'
    }

    # 5. 创建并加载模型
    model = resnet34(num_classes=len(class_indict)).to(device)
    weights_path = "./resNet34.pth"
    assert os.path.exists(weights_path), f"file: '{weights_path}' does not exist."
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # 6. 准备测试集
    test_path = "./dataset/plates/test"
    assert os.path.exists(test_path), f"directory: '{test_path}' does not exist."
    # 收集所有 .jpg 文件
    img_path_list = [
        os.path.join(test_path, fname)
        for fname in os.listdir(test_path)
        if fname.lower().endswith(".jpg")
    ]
    assert img_path_list, f"No .jpg images found in '{test_path}'."

    # 7. 推理
    model.eval()
    batch_size = 8
    total = 0
    predictions = []

    with torch.no_grad():
        # 按 batch 处理
        for start in range(0, len(img_path_list), batch_size):
            batch_paths = img_path_list[start:start+batch_size]
            imgs = []
            for p in batch_paths:
                img = Image.open(p).convert("RGB")
                img = data_transform(img)
                imgs.append(img)
            batch_tensor = torch.stack(imgs, dim=0).to(device)

            outputs = model(batch_tensor).cpu()
            probs = torch.softmax(outputs, dim=1)
            _, classes = torch.max(probs, dim=1)

            for idx, cls in enumerate(classes):
                cls_idx = cls.item()
                letter = letter_map.get(cls_idx, str(cls_idx))
                predictions.append(letter)

                print(f"image: {batch_paths[idx]:30s} "
                      f"class: {class_indict[str(cls_idx)]:7s} "
                      f"-> '{letter}'  prob: {probs[idx, cls_idx]:.3f}")

            total += len(batch_paths)

    print(f"Total images processed: {total}")

    # 8. 保存预测结果
    with open('test_predictions.json', 'w') as json_file:
        json.dump(predictions, json_file, indent=2, ensure_ascii=False)

    print("Test predictions saved to 'test_predictions.json'.")

if __name__ == '__main__':
    main()

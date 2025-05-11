import os
import json
import csv
from collections import defaultdict

def load_ground_truth(csv_path):
    """
    读取 CSV，返回 dict: filename -> int(label)
    """
    gt = {}
    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 2:
                continue
            filename = row[0].strip()
            label = int(row[1].strip())
            gt[filename] = label
    return gt

def load_predictions(pred_json_path, filenames):
    """
    读取预测 JSON（["D","C",...]），并映射回数字列表
    filenames: 按排序的文件名列表，用于对齐
    返回：list of int(pred)
    """
    with open(pred_json_path, 'r') as f:
        preds_letters = json.load(f)

    if len(preds_letters) != len(filenames):
        raise ValueError(f"预测数量 ({len(preds_letters)}) 与测试图片数 ({len(filenames)}) 不匹配")

    letter_map = {'C': 0, 'D': 1}
    preds = []
    for letter in preds_letters:
        if letter not in letter_map:
            raise ValueError(f"未知预测字母 '{letter}'")
        preds.append(letter_map[letter])
    return preds

def compute_metrics(true_list, pred_list):
    """
    计算总体准确率，以及每类别的准确率
    true_list, pred_list: 对齐的标签列表（数字0/1）
    """
    total = len(true_list)
    correct = sum(t == p for t, p in zip(true_list, pred_list))

    # per-class 统计正确与总数
    stats = {
        0: {'total': 0, 'correct': 0},
        1: {'total': 0, 'correct': 0}
    }
    for t, p in zip(true_list, pred_list):
        stats[t]['total'] += 1
        if t == p:
            stats[t]['correct'] += 1

    overall_acc = correct / total if total else 0.0
    class_acc = {
        cls: (stats[cls]['correct'] / stats[cls]['total'] if stats[cls]['total'] else 0.0)
        for cls in stats
    }

    return overall_acc, class_acc

if __name__ == '__main__':
    # ---------- 配置，根据实际情况修改路径 ----------
    CSV_PATH = './test_labels.csv'            # 标注文件
    PRED_JSON = './test_predictions.json'     # 预测结果
    TEST_DIR  = './dataset/plates/test'       # 测试图片目录

    # 1. 按文件名排序，构造文件名列表
    filenames = sorted(
        fn for fn in os.listdir(TEST_DIR)
        if fn.lower().endswith('.jpg')
    )
    if not filenames:
        raise ValueError(f"在目录 '{TEST_DIR}' 下未找到任何 .jpg 文件")

    # 2. 加载 ground truth，按 filenames 顺序构造 true_list
    ground_truth_dict = load_ground_truth(CSV_PATH)
    true_list = []
    for fn in filenames:
        if fn not in ground_truth_dict:
            raise KeyError(f"标注文件中缺少图片 '{fn}' 的标签")
        true_list.append(ground_truth_dict[fn])

    # 3. 加载 predictions，得到 pred_list
    pred_list = load_predictions(PRED_JSON, filenames)

    # 4. 输出文件名及标签列表
    print("===== 文件名列表 =====")
    print(filenames)
    print("\n===== 真实标签列表 (0=cleaned,1=dirty) =====")
    print(true_list)
    print("\n===== 预测标签列表 (0=cleaned,1=dirty) =====")
    print(pred_list)

    # 5. 统计各类别数量
    true_counts = defaultdict(int)
    pred_counts = defaultdict(int)
    for t in true_list:
        true_counts[t] += 1
    for p in pred_list:
        pred_counts[p] += 1

    print("\n===== 各类别样本数 =====")
    print(f"真实标签中 Cleaned(0): {true_counts[0]}，Dirty(1): {true_counts[1]}")
    print(f"预测标签中 Cleaned(0): {pred_counts[0]}，Dirty(1): {pred_counts[1]}")

    # 6. 计算准确率
    overall_acc, class_acc = compute_metrics(true_list, pred_list)

    # 7. 打印准确率结果
    print("\n===== 准确率结果 =====")
    print(f"总样本数: {len(filenames)}")
    print(f"总体准确率: {overall_acc:.4f}")
    print(f"Cleaned (0) 准确率: {class_acc.get(0, 0.0):.4f}")
    print(f"Dirty   (1) 准确率: {class_acc.get(1, 0.0):.4f}")

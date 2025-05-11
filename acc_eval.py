import json

# 读取预测结果
with open("test_predictions.json", "r") as f:
    predictions = json.load(f)

# 检查长度是否正确
assert len(predictions) == 744, f"预测结果数量错误，应为744，实际为 {len(predictions)}"

# 提取预测为 "C" 的图片编号（去除前导零）
cleaned_ids = [
    str(i) for i, pred in enumerate(predictions)
    if pred == "C"
]

# 拼接参数字符串
cleaned_param = ",".join(cleaned_ids)

# 构造完整URL
base_url = "http://202.207.12.156:20000/calculate_accuracy"
full_url = f"{base_url}?cleaned_ids={cleaned_param}"

# 输出结果
print("提交URL如下：")
print(full_url)


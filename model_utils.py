# model_utils.py
import torch
from sentence_transformers import SentenceTransformer

# 检测可用设备（优先使用GPU）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"模型将加载到设备: {device}")
model_path = "./all-MiniLM-L6-v2"
model = SentenceTransformer(model_path)
# 全局唯一模型实例

def get_model():
    global model
    if model is None:
        # 首次调用时加载模型到指定设备
        model = SentenceTransformer('all-MiniLM-L12-v2', device=device)
        # 如果使用GPU，启用半精度以节省内存
        if device.type == 'cuda':
            model = model.half()  # 转为FP16
            print("已启用GPU半精度计算以优化内存")
    return model
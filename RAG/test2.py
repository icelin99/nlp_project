import torch
print(torch.cuda.is_available())  # 应返回 True
print(torch.__version__)          # 检查 PyTorch 版本
print(torch.version.cuda)         # 显示 PyTorch 使用的 CUDA 版本

import torch
print(torch.cuda.is_available())  # True が表示されれば GPU が使用可能
print(torch.cuda.get_device_name(0))  # 使用する GPU の名前が表示されます


import torch

print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())
print(torch.cuda.device(0))
print(torch.cuda.get_device_name(0))

print(torch.load('models\\2022-12-10_19-38-03\loss_fn_weak.pt'))
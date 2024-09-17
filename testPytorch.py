import torch
device = torch.device("cpu")
print(f"Using device: {device}")

devNumber=torch.cuda.current_device()

print(f"Current device number is : {devNumber}")
devName=torch.cuda.get_device_name(devNumber)

print(f" name is: {devName}")
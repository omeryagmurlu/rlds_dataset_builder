import os
import torch

path = "/home/mnikolaus/code/data/collected_data/2025_03_31-16_34_01"
subfolders = ["Gello leader", "Panda 102 follower", "../"]

for subfolder in subfolders:
    folder_path = os.path.join(path, subfolder)
    
    # Skip if subfolder doesn't exist
    if not os.path.exists(folder_path):
        print(f"⚠️ Subfolder {subfolder} not found, skipping...")
        continue
        
    print(f"\n🔍 Checking subfolder: {subfolder}")
    
    # Get all .pt files in subfolder
    pt_files = [f for f in os.listdir(folder_path) if f.endswith('.pt')]
    
    for file_name in pt_files:
        file_path = os.path.join(folder_path, file_name)
        try:
            # Load PyTorch data
            data = torch.load(file_path)
            print(f"\n📁 File: {file_name}")
            
            # Handle different data structures
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, torch.Tensor):
                        print(f"│   Key: {key} | Shape: {value.shape} | Dtype: {value.dtype}")
                    else:
                        print(f"│   Key: {key} | Type: {type(value).__name__}")
            
            elif isinstance(data, (list, tuple)):
                for idx, item in enumerate(data):
                    if isinstance(item, torch.Tensor):
                        print(f"│   Index: {idx} | Shape: {item.shape} | Dtype: {item.dtype}")
                    else:
                        print(f"│   Index: {idx} | Type: {type(item).__name__}")
            
            elif isinstance(data, torch.Tensor):
                print(f"│   Shape: {data.shape} | Dtype: {data.dtype}")
            
            else:
                print(f"│   Unknown data type: {type(data).__name__}")

        except Exception as e:
            print(f"\n❌ Error loading {file_name}: {str(e)}")



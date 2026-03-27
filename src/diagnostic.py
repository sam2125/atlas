import torch
import os

# Check if files exist
paths = {
    "ViT-B-16": "/data8/rajiv/mariamma/sidharth/atlas/atlas/checkpoints/ViT-B-16/ImageNetVal/finetuned.pt",
    "ViT-B-32": "/data8/rajiv/mariamma/sidharth/atlas/atlas/checkpoints/ViT-B-32/ImageNetVal/finetuned.pt",
    "ViT-L-14": "/data8/rajiv/mariamma/sidharth/atlas/atlas/checkpoints/ViT-L-14/ImageNetVal/finetuned.pt",
}

for name, path in paths.items():
    print(f"\n{name}:")
    print(f"  Path: {path}")
    print(f"  Exists: {os.path.exists(path)}")
    
    if os.path.exists(path):
        print(f"  Size: {os.path.getsize(path) / 1e6:.1f} MB")
        
        # Try loading
        try:
            ckpt = torch.load(path, map_location='cpu')
            print(f"  Type: {type(ckpt)}")
            
            if isinstance(ckpt, dict):
                print(f"  Keys: {list(ckpt.keys())[:10]}...")
                print(f"  Num keys: {len(ckpt)}")
                
                # Check first key structure
                first_key = list(ckpt.keys())[0]
                print(f"  First key: {first_key}")
                print(f"  First value shape: {ckpt[first_key].shape}")
        except Exception as e:
            print(f"  ERROR loading: {e}")
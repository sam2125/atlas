# import os
# import json
# import torch

# from src.linearize import LinearizedImageEncoder
# from src.task_vectors import LinearizedTaskVector
# from src.args import parse_arguments
# from src.eval import eval_single_dataset

# DATASETS = [
#     "Cars",
#     "DTD",
#     "EuroSAT",
#     "GTSRB",
#     "MNIST",
#     "RESISC45",
#     "SUN397",
#     "SVHN",
# ]

# def main():
#     args = parse_arguments()
    
#     if args.seed is not None:
#         args.save = f"checkpoints_{args.seed}/{args.model}"
#     else:
#         args.save = f"checkpoints/{args.model}"
    
#     results = {}
    
#     for dataset in DATASETS:
#         print(f"\n{'='*50}")
#         print(f"Evaluating on {dataset}")
#         print('='*50)
        
#         # Paths to checkpoints
#         zs_path = os.path.join(args.save, f"{dataset}Val", "linear_zeroshot.pt")
#         ft_path = os.path.join(args.save, f"{dataset}Val", "linear_finetuned.pt")
        
#         if not os.path.exists(zs_path):
#             print(f"  Skipping {dataset}: zeroshot checkpoint not found at {zs_path}")
#             continue
#         if not os.path.exists(ft_path):
#             print(f"  Skipping {dataset}: finetuned checkpoint not found at {ft_path}")
#             continue
        
#         # Load linearized image encoder
#         image_encoder = LinearizedImageEncoder(args, keep_lang=False)
        
#         # Create task vector and apply it
#         task_vector = LinearizedTaskVector(zs_path, ft_path)
        
#         # Apply task vector with coefficient 1.0 (full finetuned model)
#         image_encoder = task_vector.apply_to(image_encoder, scaling_coef=1.0)
        
#         # Evaluate
#         eval_result = eval_single_dataset(image_encoder, dataset, args)
        
#         results[dataset] = {
#             "top1": eval_result["top1"],
#             "top5": eval_result.get("top5", None)
#         }
        
#         print(f"  {dataset} Top-1 Accuracy: {100 * eval_result['top1']:.2f}%")
#         # if "top5" in eval_result:
#             # print(f"  {dataset} Top-5 Accuracy: {100 * eval_result['top5']:.2f}%")
    
#     # Save results
#     save_path = os.path.join(args.save, "linear_finetuned_accuracies.json")
#     with open(save_path, 'w') as f:
#         json.dump(results, f, indent=4)
#     print(f"\nResults saved to {save_path}")
    
#     # Print summary
#     print("\n" + "="*50)
#     print("Summary")
#     print("="*50)
#     for dataset, acc in results.items():
#         print(f"{dataset:15s}: {100 * acc['top1']:.2f}%")

# if __name__ == "__main__":
#     main()

import os
import json
import torch

os.environ["OPEN_CLIP_USE_NATIVE_ATT"] = "0"

# Disable scaled dot product attention
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

from src.linearize import LinearizedImageEncoder
from src.task_vectors import LinearizedTaskVector
from src.args import parse_arguments
from src.eval import eval_single_dataset

DATASETS = [
    "Cars",
    "DTD",
    "EuroSAT",
    "GTSRB",
    "MNIST",
    "RESISC45",
    "SUN397",
    "SVHN",
]

def main():
    args = parse_arguments()
    
    if args.seed is not None:
        args.save = f"checkpoints_{args.seed}/{args.model}"
    else:
        args.save = f"checkpoints/{args.model}"
    
    results = {}
    
    for dataset in DATASETS:
        print(f"\n{'='*50}")
        print(f"Evaluating on {dataset}")
        print('='*50)
        
        # Paths to checkpoints
        zs_path = os.path.join(args.save, f"{dataset}Val", "linear_zeroshot.pt")
        ft_path = os.path.join(args.save, f"{dataset}Val", "linear_finetuned.pt")
        
        if not os.path.exists(zs_path):
            print(f"  Skipping {dataset}: zeroshot checkpoint not found at {zs_path}")
            continue
        if not os.path.exists(ft_path):
            print(f"  Skipping {dataset}: finetuned checkpoint not found at {ft_path}")
            continue
        
        # Create task vector
        task_vector = LinearizedTaskVector(zs_path, ft_path)
        
        # Apply task vector to zeroshot checkpoint (pass path, not encoder)
        image_encoder = task_vector.apply_to(zs_path, scaling_coef=1.0)
        
        # Evaluate
        eval_result = eval_single_dataset(image_encoder, dataset, args)
        
        results[dataset] = {
            "top1": eval_result["top1"],
            "top5": eval_result.get("top5", None)
        }
        
        print(f"  {dataset} Top-1 Accuracy: {100 * eval_result['top1']:.2f}%")
        if "top5" in eval_result:
            print(f"  {dataset} Top-5 Accuracy: {100 * eval_result['top5']:.2f}%")
    
    # Save results
    save_path = os.path.join(args.save, "linear_finetuned_accuracies.json")
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {save_path}")
    
    # Print summary
    print("\n" + "="*50)
    print("Summary")
    print("="*50)
    for dataset, acc in results.items():
        print(f"{dataset:15s}: {100 * acc['top1']:.2f}%")

if __name__ == "__main__":
    main()
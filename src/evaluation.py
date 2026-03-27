"""
Model Evaluation for ViT-B-16
=============================
Evaluates merged model on all 8 tasks.
"""

# import sys
# sys.path.append("/data/mariammaa/sidharth/anisotropic_scaling/atlas")

import sys
sys.path.insert(0, "/data/mariammaa/sidharth/anisotropic_scaling/atlas")


import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
from tqdm import tqdm
import numpy as np
import open_clip

# Use existing data loader
from data_loader import get_dataloader, TASKS, TASK_LIST


# =============================================================================
# Configuration for ViT-B-16
# =============================================================================

MODEL_NAME = "ViT-B-16"
INPUT_DIM = 768

BASE_DIR = Path("/data/mariammaa/sidharth/model_fusion_hess")
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"/ "tall_masks_vit_b16"
MERGED_DIR = BASE_DIR / "outputs_b16" / "merged_models"
RESULTS_DIR = BASE_DIR / "outputs_b16" / "eval_results"


# =============================================================================
# Classification Head
# =============================================================================

class ClassificationHead(nn.Module):
    """Classification head for CLIP visual encoder."""
    
    def __init__(
        self, 
        input_dim: int = 768, 
        num_classes: int = 10,
        normalize: bool = True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.normalize = normalize
        
        self.weight = nn.Parameter(torch.randn(num_classes, input_dim))
        self.bias = nn.Parameter(torch.zeros(num_classes))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            x = F.normalize(x, dim=-1)
        return F.linear(x, self.weight, self.bias)


def _setup_src_module_mock():
    # """Setup mock modules for loading checkpoints."""
    # import types
    
    # if 'src' not in sys.modules:
    #     src = types.ModuleType('src')
    #     src_models = types.ModuleType('src.models')
    #     src_models_modeling = types.ModuleType('src.models.modeling')
        
    #     src.models = src_models
    #     src_models.modeling = src_models_modeling
    #     src_models_modeling.ClassificationHead = ClassificationHead
        
    #     sys.modules['src'] = src
    #     sys.modules['src.models'] = src_models
    #     sys.modules['src.models.modeling'] = src_models_modeling
    """Setup mock modules for loading checkpoints with src.modeling.ClassificationHead
    """
    import types
    
    # Create ClassificationHead in all possible module paths
    module_paths = [
        'src',
        'src.modeling', 
        'src.models',
        'src.models.modeling',
    ]
    
    for path in module_paths:
        if path not in sys.modules:
            sys.modules[path] = types.ModuleType(path)
    
    # Add ClassificationHead to both possible locations
    sys.modules['src.modeling'].ClassificationHead = ClassificationHead
    sys.modules['src.models.modeling'].ClassificationHead = ClassificationHead
    
    # Setup parent references
    sys.modules['src'].modeling = sys.modules['src.modeling']
    sys.modules['src'].models = sys.modules['src.models']
    sys.modules['src.models'].modeling = sys.modules['src.models.modeling']


def load_classification_head(path: Path, device: str = "cpu") -> ClassificationHead:
    """Load a classification head from checkpoint."""
    _setup_src_module_mock()
    
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    
    if isinstance(checkpoint, nn.Module):
        try:
            state_dict = checkpoint.state_dict()
            normalize = getattr(checkpoint, 'normalize', True)
        except Exception:
            state_dict = {}
            normalize = True
            if hasattr(checkpoint, 'weight'):
                w = checkpoint.weight
                state_dict['weight'] = w.data if hasattr(w, 'data') else w
            if hasattr(checkpoint, 'bias') and checkpoint.bias is not None:
                b = checkpoint.bias
                state_dict['bias'] = b.data if hasattr(b, 'data') else b
    elif isinstance(checkpoint, dict):
        normalize = True
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "weight" in checkpoint:
            state_dict = checkpoint
        else:
            state_dict = checkpoint
    else:
        normalize = getattr(checkpoint, 'normalize', True)
        if hasattr(checkpoint, 'weight'):
            w = checkpoint.weight
            state_dict = {'weight': w.data if hasattr(w, 'data') else w}
            if hasattr(checkpoint, 'bias') and checkpoint.bias is not None:
                b = checkpoint.bias
                state_dict['bias'] = b.data if hasattr(b, 'data') else b
        else:
            raise ValueError(f"Unknown checkpoint format: {type(checkpoint)}")
    
    if "weight" in state_dict:
        num_classes, input_dim = state_dict["weight"].shape
    else:
        input_dim = INPUT_DIM
        num_classes = 10
    
    head = ClassificationHead(
        input_dim=input_dim,
        num_classes=num_classes,
        normalize=normalize
    )
    
    try:
        head.load_state_dict(state_dict, strict=False)
    except Exception as e:
        print(f"  Warning: Loading manually ({e})")
        if 'weight' in state_dict:
            head.weight.data = state_dict['weight'].float()
        if 'bias' in state_dict and state_dict['bias'] is not None:
            head.bias.data = state_dict['bias'].float()
    
    return head.to(device)


# =============================================================================
# Model Loading
# =============================================================================

def load_pretrained_clip(
    model_name: str = MODEL_NAME,
    pretrained: str = "openai",
    device: str = "cpu",
    visual_only: bool = True
) -> Tuple[nn.Module, Dict[str, torch.Tensor]]:
    """Load pretrained CLIP model."""
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, 
        pretrained=pretrained
    )
    model = model.to(device)
    model.eval()
    
    if visual_only:
        visual_encoder = model.visual
        state_dict = {k: v.float() for k, v in visual_encoder.state_dict().items()}
        return visual_encoder, state_dict
    else:
        state_dict = {k: v.float() for k, v in model.state_dict().items()}
        return model, state_dict


def load_merged_model(merged_path: Path, device: str = "cpu") -> Tuple[nn.Module, Dict]:
    """Load a merged model."""
    visual_encoder, _ = load_pretrained_clip(device=device, visual_only=True)
    
    checkpoint = torch.load(merged_path, map_location=device, weights_only=False)
    
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            metadata = checkpoint.get("metadata", {})
        else:
            state_dict = checkpoint
            metadata = {}
    else:
        raise ValueError(f"Unknown checkpoint format: {type(checkpoint)}")
    
    visual_encoder.load_state_dict(state_dict)
    visual_encoder.eval()
    
    return visual_encoder, metadata


def load_finetuned_model(task: str, device: str = "cpu") -> nn.Module:
    """Load a fine-tuned model for a specific task."""
    visual_encoder, pretrained_sd = load_pretrained_clip(device=device, visual_only=True)
    
    ckpt_path = CHECKPOINTS_DIR / task / "model.pt"
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    if hasattr(checkpoint, 'state_dict'):
        state_dict = checkpoint.state_dict()
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    else:
        raise ValueError(f"Unknown checkpoint format: {type(checkpoint)}")
    
    # Extract visual encoder weights
    visual_prefix = "model.visual."
    finetuned_sd = {}
    for key, value in state_dict.items():
        if key.startswith(visual_prefix):
            new_key = key[len(visual_prefix):]
            finetuned_sd[new_key] = value.float()
    
    if len(finetuned_sd) == 0:
        finetuned_sd = {k: v.float() for k, v in state_dict.items()}
    
    pretrained_sd.update(finetuned_sd)
    visual_encoder.load_state_dict(pretrained_sd)
    visual_encoder.eval()
    
    return visual_encoder


# =============================================================================
# Evaluation
# =============================================================================

@torch.no_grad()
def evaluate_task(
    model: nn.Module,
    head: nn.Module,
    dataloader,
    device: str = "cuda"
) -> Dict[str, float]:
    """Evaluate a model on a single task."""
    model = model.to(device)
    head = head.to(device)
    model.eval()
    head.eval()
    
    correct = 0
    total = 0
    
    for images, labels in tqdm(dataloader, desc="    Evaluating", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        
        features = model(images)
        logits = head(features)
        preds = logits.argmax(dim=1)
        
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    accuracy = correct / total if total > 0 else 0.0
    
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
    }


def evaluate_all_tasks(
    model: nn.Module,
    tasks: Optional[List[str]] = None,
    device: str = "cuda",
    batch_size: int = 64,
    num_workers: int = 4
) -> Dict[str, Dict[str, float]]:
    """Evaluate a model on all tasks."""
    if tasks is None:
        tasks = TASK_LIST
    
    results = {}
    
    for task in tasks:
        print(f"\nEvaluating {task}...")
        
        # Load classification head
        head_path = CHECKPOINTS_DIR / task / "head.pt"
        if not head_path.exists():
            print(f"  Warning: Head not found at {head_path}, skipping")
            continue
        
        head = load_classification_head(head_path, device=device)
        print(f"  Loaded head: {head.input_dim} -> {head.num_classes}")
        
        # Load test dataloader using existing data_loader.py
        dataloader = get_dataloader(
            task=task,
            batch_size=batch_size,
            num_workers=num_workers,
            split="test"
        )
        
        # Evaluate
        metrics = evaluate_task(model, head, dataloader, device)
        
        results[task] = {
            "accuracy": metrics["accuracy"],
            "correct": metrics["correct"],
            "total": metrics["total"]
        }
        
        print(f"  Accuracy: {metrics['accuracy']*100:.2f}% ({metrics['correct']}/{metrics['total']})")
    
    # Compute average
    if results:
        avg_accuracy = sum(r["accuracy"] for r in results.values()) / len(results)
        results["average"] = {"accuracy": avg_accuracy}
        print(f"\nAverage accuracy: {avg_accuracy*100:.2f}%")
    
    return results


def compare_models(
    merged_path: Path,
    tasks: Optional[List[str]] = None,
    device: str = "cuda",
    batch_size: int = 64,
    include_finetuned: bool = True
) -> Dict[str, Dict[str, float]]:
    """Compare merged model against pretrained and fine-tuned models."""
    if tasks is None:
        tasks = TASK_LIST
    
    results = {
        "pretrained": {},
        "merged": {},
        "finetuned": {}
    }
    
    # Pretrained
    print("\n" + "=" * 60)
    print("EVALUATING PRETRAINED MODEL")
    print("=" * 60)
    pretrained_encoder, _ = load_pretrained_clip(device=device, visual_only=True)
    results["pretrained"] = evaluate_all_tasks(
        pretrained_encoder, tasks, device, batch_size
    )
    
    # Merged
    print("\n" + "=" * 60)
    print("EVALUATING MERGED MODEL")
    print("=" * 60)
    merged_encoder, metadata = load_merged_model(merged_path, device)
    print(f"Merged model metadata: {metadata}")
    results["merged"] = evaluate_all_tasks(
        merged_encoder, tasks, device, batch_size
    )
    
    # Fine-tuned
    if include_finetuned:
        print("\n" + "=" * 60)
        print("EVALUATING FINE-TUNED MODELS")
        print("=" * 60)
        
        for task in tasks:
            print(f"\nEvaluating fine-tuned model for {task}...")
            try:
                finetuned_encoder = load_finetuned_model(task, device)
                
                head_path = CHECKPOINTS_DIR / task / "head.pt"
                if not head_path.exists():
                    print(f"  Warning: Head not found, skipping")
                    continue
                
                head = load_classification_head(head_path, device)
                dataloader = get_dataloader(
                    task=task,
                    batch_size=batch_size,
                    num_workers=4,
                    split="test"
                )
                
                metrics = evaluate_task(finetuned_encoder, head, dataloader, device)
                results["finetuned"][task] = {
                    "accuracy": metrics["accuracy"],
                    "correct": metrics["correct"],
                    "total": metrics["total"]
                }
                print(f"  Accuracy: {metrics['accuracy']*100:.2f}%")
                
            except Exception as e:
                print(f"  Error: {e}")
                import traceback
                traceback.print_exc()
        
        if results["finetuned"]:
            avg = sum(r["accuracy"] for r in results["finetuned"].values()) / len(results["finetuned"])
            results["finetuned"]["average"] = {"accuracy": avg}
    
    # Print summary
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Task':<15} {'Pretrained':>12} {'Merged':>12} {'Fine-tuned':>12}")
    print("-" * 55)
    
    for task in tasks:
        pre_acc = results["pretrained"].get(task, {}).get("accuracy", 0) * 100
        mer_acc = results["merged"].get(task, {}).get("accuracy", 0) * 100
        ft_acc = results["finetuned"].get(task, {}).get("accuracy", 0) * 100 if include_finetuned else 0
        
        print(f"{task:<15} {pre_acc:>11.2f}% {mer_acc:>11.2f}% {ft_acc:>11.2f}%")
    
    print("-" * 55)
    pre_avg = results["pretrained"].get("average", {}).get("accuracy", 0) * 100
    mer_avg = results["merged"].get("average", {}).get("accuracy", 0) * 100
    ft_avg = results["finetuned"].get("average", {}).get("accuracy", 0) * 100 if include_finetuned else 0
    print(f"{'Average':<15} {pre_avg:>11.2f}% {mer_avg:>11.2f}% {ft_avg:>11.2f}%")
    
    return results


# =============================================================================
# Main
# =============================================================================

# def main()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate merged ViT-B-16 models")
    parser.add_argument("--merged-path", type=str, default=None)
    parser.add_argument("--tasks", type=str, nargs="+", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    
    args = parser.parse_args()
    
    print(f"Using device: {args.device}")
    print(f"Model: {MODEL_NAME}")
    
    # Find merged model
    if args.merged_path is None:
        merged_files = list(MERGED_DIR.glob("merged_*.pt"))
        if merged_files:
            args.merged_path = str(max(merged_files, key=lambda p: p.stat().st_mtime))
            print(f"Using most recent merged model: {args.merged_path}")
        else:
            print(f"No merged model found in {MERGED_DIR}. Please specify --merged-path")
            exit(1)
    
    merged_path = Path(args.merged_path)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    if args.compare:
        results = compare_models(
            merged_path=merged_path,
            tasks=args.tasks,
            device=args.device,
            batch_size=args.batch_size,
            include_finetuned=True
        )
    else:
        print("\n" + "=" * 60)
        print("EVALUATING MERGED MODEL")
        print("=" * 60)
        
        merged_encoder, metadata = load_merged_model(merged_path, args.device)
        print(f"Metadata: {metadata}")
        
        results = evaluate_all_tasks(
            merged_encoder,
            tasks=args.tasks,
            device=args.device,
            batch_size=args.batch_size
        )
    
    # Save results
    output_path = args.output or (RESULTS_DIR / f"eval_{merged_path.stem}.json")
    
    def make_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        return obj
    
    with open(output_path, "w") as f:
        json.dump(make_serializable(results), f, indent=2)
    print(f"\nResults saved to {output_path}")
    
    print("\nDone!")
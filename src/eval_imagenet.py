# # # # import torch
# # # # import open_clip
# # # # from datasets import Dataset
# # # # from tqdm import tqdm
# # # # from PIL import Image
# # # # import os

# # # # device = "cuda" if torch.cuda.is_available() else "cpu"

# # # # # ImageNet class templates for zero-shot
# # # # TEMPLATES = [
# # # #     "a photo of a {}.",
# # # #     "a blurry photo of a {}.",
# # # #     "a photo of many {}.",
# # # #     "a photo of the large {}.",
# # # #     "a photo of the small {}.",
# # # #     "itap of a {}.",
# # # #     "a bad photo of the {}.",
# # # #     "a origami {}.",
# # # #     "a photo of the {}.",
# # # #     "a sketch of a {}.",
# # # # ]

# # # # def load_imagenet_val(parquet_dir):
# # # #     """Load ImageNet validation set from downloaded parquet files."""
# # # #     parquet_files = sorted([
# # # #         os.path.join(parquet_dir, f)
# # # #         for f in os.listdir(parquet_dir)
# # # #         if f.startswith("validation") and f.endswith(".parquet")
# # # #     ])
# # # #     print(f"Loading {len(parquet_files)} parquet files...")
# # # #     ds = Dataset.from_parquet(parquet_files)
# # # #     print(f"Loaded {len(ds)} images")
# # # #     return ds

# # # # def get_classnames():
# # # #     """Get ImageNet class names."""
# # # #     # Standard ImageNet class names
# # # #     import json
# # # #     import urllib.request
# # # #     url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
# # # #     classnames = json.loads(urllib.request.urlopen(url).read().decode())
# # # #     return classnames

# # # # def get_text_features(model, tokenizer, classnames, templates):
# # # #     """Compute text features for zero-shot classification."""
# # # #     with torch.no_grad():
# # # #         text_features = []
# # # #         for classname in tqdm(classnames, desc="Encoding text"):
# # # #             texts = [t.format(classname) for t in templates]
# # # #             texts = tokenizer(texts).to(device)
# # # #             class_embeddings = model.encode_text(texts)
# # # #             class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
# # # #             class_embedding = class_embeddings.mean(dim=0)
# # # #             class_embedding /= class_embedding.norm()
# # # #             text_features.append(class_embedding)
# # # #         text_features = torch.stack(text_features, dim=0)
# # # #     return text_features

# # # # def evaluate_model(model, preprocess, tokenizer, dataset, classnames, batch_size=256):
# # # #     """Evaluate model on ImageNet validation set."""
# # # #     model = model.to(device).eval()
    
# # # #     # Get text features
# # # #     print("Computing text features...")
# # # #     text_features = get_text_features(model, tokenizer, classnames, TEMPLATES)
    
# # # #     # Evaluate in batches
# # # #     correct = 0
# # # #     total = 0
    
# # # #     print("Evaluating images...")
# # # #     for i in tqdm(range(0, len(dataset), batch_size)):
# # # #         batch = dataset[i:i+batch_size]
        
# # # #         images = []
# # # #         for img in batch['image']:
# # # #             if img.mode != 'RGB':
# # # #                 img = img.convert('RGB')
# # # #             images.append(preprocess(img))
        
# # # #         images = torch.stack(images).to(device)
# # # #         labels = torch.tensor(batch['label']).to(device)
        
# # # #         with torch.no_grad():
# # # #             image_features = model.encode_image(images)
# # # #             image_features /= image_features.norm(dim=-1, keepdim=True)
            
# # # #             logits = image_features @ text_features.T
# # # #             preds = logits.argmax(dim=-1)
            
# # # #             correct += (preds == labels).sum().item()
# # # #             total += len(labels)
    
# # # #     accuracy = correct / total * 100
# # # #     return accuracy

# # # # def load_finetuned_model(model_name, checkpoint_path):
# # # #     """Load a fine-tuned CLIP model from checkpoint."""
# # # #     # Create base model
# # # #     model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained='openai')
# # # #     tokenizer = open_clip.get_tokenizer(model_name)
    
# # # #     # Load fine-tuned weights
# # # #     print(f"Loading checkpoint from {checkpoint_path}")
# # # #     checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
# # # #     # Handle different checkpoint formats
# # # #     if 'state_dict' in checkpoint:
# # # #         state_dict = checkpoint['state_dict']
# # # #     elif 'model' in checkpoint:
# # # #         state_dict = checkpoint['model']
# # # #     else:
# # # #         state_dict = checkpoint
    
# # # #     # Remove 'module.' prefix if present (from DataParallel)
# # # #     state_dict = {k.replace('module.', '').replace('model.', ''): v for k, v in state_dict.items()}
    
# # # #     model.load_state_dict(state_dict, strict=False)
# # # #     return model, preprocess, tokenizer

# # # # def main():
# # # #     # Paths
# # # #     parquet_dir = "/data8/rajiv/mariamma/sidharth/atlas/atlas/data/imagenet_val/data"
    
# # # #     # Your fine-tuned model checkpoints - UPDATE THESE PATHS
# # # #     FINETUNED_MODELS = {
# # # #         "ViT-B-16": "/data8/rajiv/mariamma/sidharth/atlas/atlas/checkpoints/ViT-B-16/ImageNetVal/finetuned.pt",
# # # #         "ViT-B-32": "/data8/rajiv/mariamma/sidharth/atlas/atlas/checkpoints/ViT-B-32/ImageNetVal/finetuned.pt",
# # # #         "ViT-L-14": "/data8/rajiv/mariamma/sidharth/atlas/atlas/checkpoints/ViT-L-14/ImageNetVal/finetuned.pt",
# # # #     }
    
# # # #     # Or evaluate pretrained models (for comparison)
# # # #     PRETRAINED_MODELS = [
# # # #         ("ViT-B-16", "openai"),
# # # #         ("ViT-B-32", "openai"),
# # # #         ("ViT-L-14", "openai"),
# # # #     ]
    
# # # #     # Load dataset
# # # #     ds = load_imagenet_val(parquet_dir)
# # # #     classnames = get_classnames()
# # # #     print(f"Number of classes: {len(classnames)}")
    
# # # #     results = {}
    
# # # #     # Evaluate pretrained models
# # # #     print("\n" + "="*60)
# # # #     print("EVALUATING PRETRAINED MODELS")
# # # #     print("="*60)
    
# # # #     # for model_name, pretrained in PRETRAINED_MODELS:
# # # #     #     print(f"\n>>> {model_name} ({pretrained})")
# # # #     #     model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
# # # #     #     tokenizer = open_clip.get_tokenizer(model_name)
        
# # # #     #     acc = evaluate_model(model, preprocess, tokenizer, ds, classnames)
# # # #     #     results[f"{model_name}_pretrained"] = acc
# # # #     #     print(f"Accuracy: {acc:.2f}%")
        
# # # #     #     del model
# # # #     #     torch.cuda.empty_cache()
    
# # # #     # Evaluate fine-tuned models
# # # #     print("\n" + "="*60)
# # # #     print("EVALUATING FINETUNED MODELS")
# # # #     print("="*60)
    
# # # #     for model_name, ckpt_path in FINETUNED_MODELS.items():
# # # #         if not os.path.exists(ckpt_path):
# # # #             print(f"\nSkipping {model_name} - checkpoint not found: {ckpt_path}")
# # # #             continue
            
# # # #         print(f"\n>>> {model_name} (finetuned)")
# # # #         model, preprocess, tokenizer = load_finetuned_model(model_name, ckpt_path)
        
# # # #         acc = evaluate_model(model, preprocess, tokenizer, ds, classnames)
# # # #         results[f"{model_name}_finetuned"] = acc
# # # #         print(f"Accuracy: {acc:.2f}%")
        
# # # #         del model
# # # #         torch.cuda.empty_cache()
    
# # # #     # Summary
# # # #     print("\n" + "="*60)
# # # #     print("FINAL RESULTS")
# # # #     print("="*60)
# # # #     for name, acc in results.items():
# # # #         print(f"{name}: {acc:.2f}%")

# # # # if __name__ == "__main__":
# # # #     main()

# # # import torch
# # # import open_clip
# # # from datasets import Dataset
# # # from tqdm import tqdm
# # # import os

# # # device = "cuda" if torch.cuda.is_available() else "cpu"

# # # # =============================================================================
# # # # OpenAI's 80 ImageNet templates (used in CLIP paper)
# # # # =============================================================================
# # # OPENAI_IMAGENET_TEMPLATES = [
# # #     "a bad photo of a {}.",
# # #     "a photo of many {}.",
# # #     "a sculpture of a {}.",
# # #     "a photo of the hard to see {}.",
# # #     "a low resolution photo of the {}.",
# # #     "a rendering of a {}.",
# # #     "graffiti of a {}.",
# # #     "a bad photo of the {}.",
# # #     "a cropped photo of the {}.",
# # #     "a tattoo of a {}.",
# # #     "the embroidered {}.",
# # #     "a photo of a hard to see {}.",
# # #     "a bright photo of a {}.",
# # #     "a photo of a clean {}.",
# # #     "a photo of a dirty {}.",
# # #     "a dark photo of the {}.",
# # #     "a drawing of a {}.",
# # #     "a photo of my {}.",
# # #     "the plastic {}.",
# # #     "a photo of the cool {}.",
# # #     "a close-up photo of a {}.",
# # #     "a black and white photo of the {}.",
# # #     "a painting of the {}.",
# # #     "a painting of a {}.",
# # #     "a pixelated photo of the {}.",
# # #     "a sculpture of the {}.",
# # #     "a bright photo of the {}.",
# # #     "a cropped photo of a {}.",
# # #     "a plastic {}.",
# # #     "a photo of the dirty {}.",
# # #     "a jpeg corrupted photo of a {}.",
# # #     "a blurry photo of the {}.",
# # #     "a photo of the {}.",
# # #     "a good photo of the {}.",
# # #     "a rendering of the {}.",
# # #     "a {} in a video game.",
# # #     "a photo of one {}.",
# # #     "a doodle of a {}.",
# # #     "a close-up photo of the {}.",
# # #     "a photo of a {}.",
# # #     "the origami {}.",
# # #     "the {} in a video game.",
# # #     "a sketch of a {}.",
# # #     "a doodle of the {}.",
# # #     "a origami {}.",
# # #     "a low resolution photo of a {}.",
# # #     "the toy {}.",
# # #     "a rendition of the {}.",
# # #     "a photo of the clean {}.",
# # #     "a photo of a large {}.",
# # #     "a rendition of a {}.",
# # #     "a photo of a nice {}.",
# # #     "a photo of a weird {}.",
# # #     "a blurry photo of a {}.",
# # #     "a cartoon {}.",
# # #     "art of a {}.",
# # #     "a sketch of the {}.",
# # #     "a embroidered {}.",
# # #     "a pixelated photo of a {}.",
# # #     "itap of the {}.",
# # #     "a jpeg corrupted photo of the {}.",
# # #     "a good photo of a {}.",
# # #     "a plushie {}.",
# # #     "a photo of the nice {}.",
# # #     "a photo of the small {}.",
# # #     "a photo of the weird {}.",
# # #     "the cartoon {}.",
# # #     "art of the {}.",
# # #     "a drawing of the {}.",
# # #     "a photo of the large {}.",
# # #     "a black and white photo of a {}.",
# # #     "the plushie {}.",
# # #     "a dark photo of a {}.",
# # #     "itap of a {}.",
# # #     "graffiti of the {}.",
# # #     "a toy {}.",
# # #     "itap of my {}.",
# # #     "a photo of a cool {}.",
# # #     "a photo of a small {}.",
# # #     "a tattoo of the {}.",
# # # ]

# # # # =============================================================================
# # # # Official ImageNet classnames (WordNet synset ordering)
# # # # =============================================================================
# # # def get_imagenet_classnames():
# # #     """
# # #     Get official ImageNet classnames in correct order (synset index 0-999).
# # #     """
# # #     import urllib.request
# # #     import json
    
# # #     # Option 1: CLIP classnames from a mirror/working source
# # #     urls_to_try = [
# # #         # OpenCLIP's classnames
# # #         "https://raw.githubusercontent.com/mlfoundations/open_clip/main/src/open_clip/imagenet_classnames.txt",
# # #         # Backup: keras imagenet class index
# # #         "https://raw.githubusercontent.com/raghakot/keras-vis/master/resources/imagenet_class_index.json",
# # #     ]
    
# # #     # Try OpenCLIP first (same format as OpenAI's)
# # #     try:
# # #         response = urllib.request.urlopen(urls_to_try[0])
# # #         classnames = [line.decode('utf-8').strip() for line in response.readlines()]
# # #         print(f"Loaded {len(classnames)} classnames from OpenCLIP repo")
# # #         return classnames
# # #     except Exception as e:
# # #         print(f"Failed to load from OpenCLIP: {e}")
    
# # #     # Fallback to imagenet_class_index.json
# # #     try:
# # #         response = urllib.request.urlopen(urls_to_try[1])
# # #         class_idx = json.loads(response.read().decode())
# # #         classnames = [class_idx[str(i)][1].replace("_", " ") for i in range(1000)]
# # #         print(f"Loaded {len(classnames)} classnames from keras-vis")
# # #         return classnames
# # #     except Exception as e:
# # #         print(f"Failed to load classnames: {e}")
# # #         raise


# # # def load_imagenet_val(parquet_dir):
# # #     parquet_files = sorted([
# # #         os.path.join(parquet_dir, f)
# # #         for f in os.listdir(parquet_dir)
# # #         if f.startswith("validation") and f.endswith(".parquet")
# # #     ])
# # #     print(f"Loading {len(parquet_files)} parquet files...")
# # #     ds = Dataset.from_parquet(parquet_files)
# # #     print(f"Loaded {len(ds)} images")
# # #     return ds

# # # def verify_label_ordering(ds):
# # #     """
# # #     Verify dataset labels are in correct range and check ordering.
# # #     """
# # #     labels = ds['label']
# # #     min_label = min(labels)
# # #     max_label = max(labels)
# # #     unique_labels = len(set(labels))
    
# # #     print(f"Label statistics:")
# # #     print(f"  Min: {min_label}, Max: {max_label}")
# # #     print(f"  Unique labels: {unique_labels}")
# # #     print(f"  Expected: 0-999, 1000 unique")
    
# # #     if min_label != 0 or max_label != 999 or unique_labels != 1000:
# # #         print("  ⚠️ WARNING: Label range mismatch!")
# # #         return False
    
# # #     print("  ✓ Labels look correct")
# # #     return True

# # # def get_text_features(model, tokenizer, classnames, templates):
# # #     """Compute text features with proper averaging over templates."""
# # #     with torch.no_grad():
# # #         text_features = []
# # #         for classname in tqdm(classnames, desc="Encoding text"):
# # #             texts = [t.format(classname) for t in templates]
# # #             texts = tokenizer(texts).to(device)
# # #             class_embeddings = model.encode_text(texts)
# # #             class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
# # #             class_embedding = class_embeddings.mean(dim=0)
# # #             class_embedding /= class_embedding.norm()
# # #             text_features.append(class_embedding)
# # #         text_features = torch.stack(text_features, dim=0)
# # #     return text_features

# # # def load_finetuned_clip(model_name, checkpoint_path):
# # #     """Load CLIP model with fine-tuned visual encoder."""
# # #     model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained='openai')
# # #     tokenizer = open_clip.get_tokenizer(model_name)
    
# # #     print(f"Loading checkpoint from {checkpoint_path}")
# # #     checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
# # #     if isinstance(checkpoint, dict):
# # #         visual_sd = {k: v.float() for k, v in checkpoint.items()}
# # #     else:
# # #         visual_sd = {k: v.float() for k, v in checkpoint.state_dict().items()}
    
# # #     model.visual.load_state_dict(visual_sd, strict=True)
# # #     print(f"  Loaded {len(visual_sd)} visual encoder parameters")
    
# # #     return model, preprocess, tokenizer

# # # def evaluate_model(model, preprocess, tokenizer, dataset, classnames, templates, batch_size=256):
# # #     """Evaluate model on ImageNet validation set."""
# # #     model = model.to(device).eval()
    
# # #     print("Computing text features...")
# # #     text_features = get_text_features(model, tokenizer, classnames, templates)
    
# # #     correct = 0
# # #     total = 0
    
# # #     print("Evaluating images...")
# # #     for i in tqdm(range(0, len(dataset), batch_size)):
# # #         batch = dataset[i:i+batch_size]
        
# # #         images = []
# # #         for img in batch['image']:
# # #             if img.mode != 'RGB':
# # #                 img = img.convert('RGB')
# # #             images.append(preprocess(img))
        
# # #         images = torch.stack(images).to(device)
# # #         labels = torch.tensor(batch['label']).to(device)
        
# # #         with torch.no_grad():
# # #             image_features = model.encode_image(images)
# # #             image_features /= image_features.norm(dim=-1, keepdim=True)
            
# # #             # Note: logit_scale omitted since it doesn't affect argmax
# # #             logits = image_features @ text_features.T
# # #             preds = logits.argmax(dim=-1)
            
# # #             correct += (preds == labels).sum().item()
# # #             total += len(labels)
    
# # #     accuracy = correct / total * 100
# # #     return accuracy

# # # def main():
# # #     parquet_dir = "/data8/rajiv/mariamma/sidharth/atlas/atlas/data/imagenet_val/data"
    
# # #     FINETUNED_MODELS = {
# # #         "ViT-B-16": "/data8/rajiv/mariamma/sidharth/atlas/atlas/checkpoints/ViT-B-16/ImageNetVal/finetuned.pt",
# # #         "ViT-B-32": "/data8/rajiv/mariamma/sidharth/atlas/atlas/checkpoints/ViT-B-32/ImageNetVal/finetuned.pt",
# # #         "ViT-L-14": "/data8/rajiv/mariamma/sidharth/atlas/atlas/checkpoints/ViT-L-14/ImageNetVal/finetuned.pt",
# # #     }
    
# # #     # Load dataset
# # #     ds = load_imagenet_val(parquet_dir)
    
# # #     # Verify label ordering
# # #     verify_label_ordering(ds)
    
# # #     # Get official classnames
# # #     classnames = get_imagenet_classnames()
    
# # #     # Use OpenAI's 80 templates
# # #     templates = OPENAI_IMAGENET_TEMPLATES
# # #     print(f"Using {len(templates)} templates")
    
# # #     results = {}
    
# # #     # Expected published results for reference
# # #     EXPECTED = {
# # #         "ViT-B-32": 63.2,
# # #         "ViT-B-16": 68.3,
# # #         "ViT-L-14": 75.5,
# # #     }
    
# # #     # Evaluate pretrained
# # #     print("\n" + "="*60)
# # #     print("PRETRAINED (ZERO-SHOT) BASELINES")
# # #     print("="*60)
    
# # #     # for model_name in ["ViT-B-32", "ViT-B-16", "ViT-L-14"]:
# # #     #     print(f"\n>>> {model_name} (pretrained)")
# # #     #     print(f"    Expected (CLIP paper): {EXPECTED[model_name]:.1f}%")
        
# # #     #     model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained='openai')
# # #     #     tokenizer = open_clip.get_tokenizer(model_name)
        
# # #     #     acc = evaluate_model(model, preprocess, tokenizer, ds, classnames, templates)
# # #     #     results[f"{model_name}_pretrained"] = acc
# # #     #     print(f"    Measured: {acc:.2f}%")
# # #     #     print(f"    Delta: {acc - EXPECTED[model_name]:+.2f}%")
        
# # #     #     del model
# # #     #     torch.cuda.empty_cache()
    
# # #     # Evaluate fine-tuned
# # #     print("\n" + "="*60)
# # #     print("FINETUNED MODELS")
# # #     print("="*60)
    
# # #     for model_name, ckpt_path in FINETUNED_MODELS.items():
# # #         if not os.path.exists(ckpt_path):
# # #             print(f"\nSkipping {model_name} - not found: {ckpt_path}")
# # #             continue
        
# # #         print(f"\n>>> {model_name} (finetuned)")
# # #         model, preprocess, tokenizer = load_finetuned_clip(model_name, ckpt_path)
        
# # #         acc = evaluate_model(model, preprocess, tokenizer, ds, classnames, templates)
# # #         results[f"{model_name}_finetuned"] = acc
# # #         print(f"    Accuracy: {acc:.2f}%")
        
# # #         # Show improvement over pretrained
# # #         pretrained_key = f"{model_name}_pretrained"
# # #         if pretrained_key in results:
# # #             delta = acc - results[pretrained_key]
# # #             print(f"    Improvement over pretrained: {delta:+.2f}%")
        
# # #         del model
# # #         torch.cuda.empty_cache()
    
# # #     # Summary
# # #     print("\n" + "="*60)
# # #     print("FINAL RESULTS")
# # #     print("="*60)
# # #     print(f"{'Model':<25} {'Accuracy':>10}")
# # #     print("-"*37)
# # #     for name, acc in sorted(results.items()):
# # #         print(f"{name:<25} {acc:>9.2f}%")

# # # if __name__ == "__main__":
# # #     main()


# # import torch
# # import torch.nn as nn
# # import open_clip
# # from datasets import Dataset
# # from tqdm import tqdm
# # import os
# # import sys
# # import types

# # device = "cuda" if torch.cuda.is_available() else "cpu"

# # # =============================================================================
# # # Mock src.modeling for checkpoint loading
# # # =============================================================================
# # def setup_src_mock():
# #     if 'src' in sys.modules:
# #         return
    
# #     class ImageEncoder(nn.Module):
# #         def __init__(self, *args, **kwargs):
# #             super().__init__()
# #             self.model = None
# #         def forward(self, x):
# #             return self.model(x) if self.model else x
    
# #     class ClassificationHead(nn.Module):
# #         def __init__(self, input_dim=512, num_classes=10, normalize=True):
# #             super().__init__()
# #             self.input_dim = input_dim
# #             self.num_classes = num_classes
# #             self.normalize = normalize
# #             self.weight = nn.Parameter(torch.randn(num_classes, input_dim))
# #             self.bias = nn.Parameter(torch.zeros(num_classes))
# #         def forward(self, x):
# #             if self.normalize:
# #                 x = nn.functional.normalize(x, dim=-1)
# #             return nn.functional.linear(x, self.weight, self.bias)
    
# #     src = types.ModuleType('src')
# #     src_modeling = types.ModuleType('src.modeling')
# #     src_models = types.ModuleType('src.models')
# #     src_models_modeling = types.ModuleType('src.models.modeling')
    
# #     src_modeling.ImageEncoder = ImageEncoder
# #     src_modeling.ClassificationHead = ClassificationHead
# #     src_models_modeling.ClassificationHead = ClassificationHead
    
# #     src.modeling = src_modeling
# #     src.models = src_models
# #     src_models.modeling = src_models_modeling
    
# #     sys.modules['src'] = src
# #     sys.modules['src.modeling'] = src_modeling
# #     sys.modules['src.models'] = src_models
# #     sys.modules['src.models.modeling'] = src_models_modeling
# #     print("✓ Mocked src.modeling")

# # setup_src_mock()

# # # =============================================================================
# # # Templates
# # # =============================================================================
# # OPENAI_IMAGENET_TEMPLATES = [
# #     "a bad photo of a {}.",
# #     "a photo of many {}.",
# #     "a sculpture of a {}.",
# #     "a photo of the hard to see {}.",
# #     "a low resolution photo of the {}.",
# #     "a rendering of a {}.",
# #     "graffiti of a {}.",
# #     "a bad photo of the {}.",
# #     "a cropped photo of the {}.",
# #     "a tattoo of a {}.",
# #     "the embroidered {}.",
# #     "a photo of a hard to see {}.",
# #     "a bright photo of a {}.",
# #     "a photo of a clean {}.",
# #     "a photo of a dirty {}.",
# #     "a dark photo of the {}.",
# #     "a drawing of a {}.",
# #     "a photo of my {}.",
# #     "the plastic {}.",
# #     "a photo of the cool {}.",
# #     "a close-up photo of a {}.",
# #     "a black and white photo of the {}.",
# #     "a painting of the {}.",
# #     "a painting of a {}.",
# #     "a pixelated photo of the {}.",
# #     "a sculpture of the {}.",
# #     "a bright photo of the {}.",
# #     "a cropped photo of a {}.",
# #     "a plastic {}.",
# #     "a photo of the dirty {}.",
# #     "a jpeg corrupted photo of a {}.",
# #     "a blurry photo of the {}.",
# #     "a photo of the {}.",
# #     "a good photo of the {}.",
# #     "a rendering of the {}.",
# #     "a {} in a video game.",
# #     "a photo of one {}.",
# #     "a doodle of a {}.",
# #     "a close-up photo of the {}.",
# #     "a photo of a {}.",
# #     "the origami {}.",
# #     "the {} in a video game.",
# #     "a sketch of a {}.",
# #     "a doodle of the {}.",
# #     "a origami {}.",
# #     "a low resolution photo of a {}.",
# #     "the toy {}.",
# #     "a rendition of the {}.",
# #     "a photo of the clean {}.",
# #     "a photo of a large {}.",
# #     "a rendition of a {}.",
# #     "a photo of a nice {}.",
# #     "a photo of a weird {}.",
# #     "a blurry photo of a {}.",
# #     "a cartoon {}.",
# #     "art of a {}.",
# #     "a sketch of the {}.",
# #     "a embroidered {}.",
# #     "a pixelated photo of a {}.",
# #     "itap of the {}.",
# #     "a jpeg corrupted photo of the {}.",
# #     "a good photo of a {}.",
# #     "a plushie {}.",
# #     "a photo of the nice {}.",
# #     "a photo of the small {}.",
# #     "a photo of the weird {}.",
# #     "the cartoon {}.",
# #     "art of the {}.",
# #     "a drawing of the {}.",
# #     "a photo of the large {}.",
# #     "a black and white photo of a {}.",
# #     "the plushie {}.",
# #     "a dark photo of a {}.",
# #     "itap of a {}.",
# #     "graffiti of the {}.",
# #     "a toy {}.",
# #     "itap of my {}.",
# #     "a photo of a cool {}.",
# #     "a photo of a small {}.",
# #     "a tattoo of the {}.",
# # ]

# # # =============================================================================
# # # Helper functions
# # # =============================================================================
# # def get_imagenet_classnames():
# #     import urllib.request
# #     import json
    
# #     try:
# #         url = "https://raw.githubusercontent.com/raghakot/keras-vis/master/resources/imagenet_class_index.json"
# #         response = urllib.request.urlopen(url)
# #         class_idx = json.loads(response.read().decode())
# #         classnames = [class_idx[str(i)][1].replace("_", " ") for i in range(1000)]
# #         print(f"Loaded {len(classnames)} classnames")
# #         return classnames
# #     except Exception as e:
# #         print(f"Failed to load classnames: {e}")
# #         raise

# # def load_imagenet_val(parquet_dir):
# #     parquet_files = sorted([
# #         os.path.join(parquet_dir, f)
# #         for f in os.listdir(parquet_dir)
# #         if f.startswith("validation") and f.endswith(".parquet")
# #     ])
# #     print(f"Loading {len(parquet_files)} parquet files...")
# #     ds = Dataset.from_parquet(parquet_files)
# #     print(f"Loaded {len(ds)} images")
# #     return ds

# # def get_text_features(model, tokenizer, classnames, templates):
# #     with torch.no_grad():
# #         text_features = []
# #         for classname in tqdm(classnames, desc="Encoding text"):
# #             texts = [t.format(classname) for t in templates]
# #             texts = tokenizer(texts).to(device)
# #             class_embeddings = model.encode_text(texts)
# #             class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
# #             class_embedding = class_embeddings.mean(dim=0)
# #             class_embedding /= class_embedding.norm()
# #             text_features.append(class_embedding)
# #         text_features = torch.stack(text_features, dim=0)
# #     return text_features

# # def load_finetuned_clip(model_name, checkpoint_path):
# #     model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained='openai')
# #     tokenizer = open_clip.get_tokenizer(model_name)
    
# #     print(f"Loading checkpoint from {checkpoint_path}")
# #     checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
# #     # Extract state dict
# #     if isinstance(checkpoint, nn.Module):
# #         if hasattr(checkpoint, 'model') and checkpoint.model is not None:
# #             visual_sd = {k: v.float() for k, v in checkpoint.model.state_dict().items()}
# #         else:
# #             visual_sd = {k: v.float() for k, v in checkpoint.state_dict().items()}
# #     elif isinstance(checkpoint, dict):
# #         if 'state_dict' in checkpoint:
# #             visual_sd = checkpoint['state_dict']
# #         elif 'model' in checkpoint:
# #             visual_sd = checkpoint['model']
# #         else:
# #             visual_sd = checkpoint
# #         visual_sd = {k: v.float() for k, v in visual_sd.items()}
# #     else:
# #         raise ValueError(f"Unknown checkpoint format: {type(checkpoint)}")
    
# #     # Remove prefixes
# #     cleaned_sd = {}
# #     for k, v in visual_sd.items():
# #         new_k = k.replace('model.visual.', '').replace('visual.', '').replace('model.', '')
# #         cleaned_sd[new_k] = v
    
# #     model.visual.load_state_dict(cleaned_sd, strict=True)
# #     print(f"  ✓ Loaded {len(cleaned_sd)} parameters")
    
# #     return model, preprocess, tokenizer

# # def evaluate_model(model, preprocess, tokenizer, dataset, classnames, templates, batch_size=256):
# #     model = model.to(device).eval()
    
# #     print("Computing text features...")
# #     text_features = get_text_features(model, tokenizer, classnames, templates)
    
# #     correct = 0
# #     total = 0
    
# #     print("Evaluating images...")
# #     for i in tqdm(range(0, len(dataset), batch_size)):
# #         batch = dataset[i:i+batch_size]
        
# #         images = []
# #         for img in batch['image']:
# #             if img.mode != 'RGB':
# #                 img = img.convert('RGB')
# #             images.append(preprocess(img))
        
# #         images = torch.stack(images).to(device)
# #         labels = torch.tensor(batch['label']).to(device)
        
# #         with torch.no_grad():
# #             image_features = model.encode_image(images)
# #             image_features /= image_features.norm(dim=-1, keepdim=True)
# #             logits = image_features @ text_features.T
# #             preds = logits.argmax(dim=-1)
# #             correct += (preds == labels).sum().item()
# #             total += len(labels)
    
# #     return correct / total * 100

# # def main():
# #     parquet_dir = "/data8/rajiv/mariamma/sidharth/atlas/atlas/data/imagenet_val/data"
    
# #     FINETUNED_MODELS = {
# #         "ViT-B-16": "/data8/rajiv/mariamma/sidharth/atlas/atlas/checkpoints/ViT-B-16/ImageNetVal/finetuned.pt",
# #         "ViT-B-32": "/data8/rajiv/mariamma/sidharth/atlas/atlas/checkpoints/ViT-B-32/ImageNetVal/finetuned.pt",
# #         "ViT-L-14": "/data8/rajiv/mariamma/sidharth/atlas/atlas/checkpoints/ViT-L-14/ImageNetVal/finetuned.pt",
# #     }
    
# #     ds = load_imagenet_val(parquet_dir)
# #     classnames = get_imagenet_classnames()
# #     templates = OPENAI_IMAGENET_TEMPLATES
    
# #     results = {}
    
# #     # Finetuned models
# #     print("\n" + "="*60)
# #     print("FINETUNED MODELS")
# #     print("="*60)
    
# #     for model_name, ckpt_path in FINETUNED_MODELS.items():
# #         if not os.path.exists(ckpt_path):
# #             print(f"\nSkipping {model_name} - not found")
# #             continue
        
# #         print(f"\n>>> {model_name} (finetuned)")
# #         model, preprocess, tokenizer = load_finetuned_clip(model_name, ckpt_path)
        
# #         acc = evaluate_model(model, preprocess, tokenizer, ds, classnames, templates)
# #         results[f"{model_name}_finetuned"] = acc
# #         print(f"    Accuracy: {acc:.2f}%")
        
# #         del model
# #         torch.cuda.empty_cache()
    
# #     # Summary
# #     print("\n" + "="*60)
# #     print("FINAL RESULTS")
# #     print("="*60)
# #     for name, acc in sorted(results.items()):
# #         print(f"{name}: {acc:.2f}%")

# # if __name__ == "__main__":
# #     main()

# import torch
# import torch.nn as nn
# import os
# import sys
# import types

# # Setup mock first
# def setup_src_mock():
#     if 'src' in sys.modules:
#         return
    
#     class ImageEncoder(nn.Module):
#         def __init__(self, *args, **kwargs):
#             super().__init__()
#             self.model = None
#         def forward(self, x):
#             return self.model(x) if self.model else x
    
#     class ClassificationHead(nn.Module):
#         def __init__(self, input_dim=512, num_classes=10, normalize=True):
#             super().__init__()
#             self.input_dim = input_dim
#             self.num_classes = num_classes
#             self.normalize = normalize
#             self.weight = nn.Parameter(torch.randn(num_classes, input_dim))
#             self.bias = nn.Parameter(torch.zeros(num_classes))
#         def forward(self, x):
#             if self.normalize:
#                 x = nn.functional.normalize(x, dim=-1)
#             return nn.functional.linear(x, self.weight, self.bias)
    
#     src = types.ModuleType('src')
#     src_modeling = types.ModuleType('src.modeling')
#     src_models = types.ModuleType('src.models')
#     src_models_modeling = types.ModuleType('src.models.modeling')
    
#     src_modeling.ImageEncoder = ImageEncoder
#     src_modeling.ClassificationHead = ClassificationHead
#     src_models_modeling.ClassificationHead = ClassificationHead
    
#     src.modeling = src_modeling
#     src.models = src_models
#     src_models.modeling = src_models_modeling
    
#     sys.modules['src'] = src
#     sys.modules['src.modeling'] = src_modeling
#     sys.modules['src.models'] = src_models
#     sys.modules['src.models.modeling'] = src_models_modeling
#     print("✓ Mocked src.modeling")

# setup_src_mock()

# # Now test loading just ONE checkpoint
# path = "/data8/rajiv/mariamma/sidharth/atlas/atlas/checkpoints/ViT-B-16/ImageNetVal/finetuned.pt"

# print(f"Loading {path}...")
# print("  Step 1: Calling torch.load...", flush=True)

# checkpoint = torch.load(path, map_location='cpu', weights_only=False)

# print(f"  Step 2: Loaded! Type = {type(checkpoint)}", flush=True)

# if isinstance(checkpoint, nn.Module):
#     print(f"  It's an nn.Module: {checkpoint.__class__.__name__}")
#     print(f"  Has 'model' attr: {hasattr(checkpoint, 'model')}")
#     if hasattr(checkpoint, 'model') and checkpoint.model is not None:
#         print(f"  model type: {type(checkpoint.model)}")
#         sd = checkpoint.model.state_dict()
#     else:
#         sd = checkpoint.state_dict()
#     print(f"  State dict keys (first 5): {list(sd.keys())[:5]}")
#     print(f"  Total keys: {len(sd)}")
# elif isinstance(checkpoint, dict):
#     print(f"  It's a dict with keys: {list(checkpoint.keys())[:10]}")
# else:
#     print(f"  Unknown type: {type(checkpoint)}")

# print("\nDone!")


import torch
import torch.nn as nn
import open_clip
from datasets import Dataset
from tqdm import tqdm
import os
import sys
import types

device = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================
# Mock src.modeling
# =============================================================================
def setup_src_mock():
    if 'src' in sys.modules:
        return
    
    class ImageEncoder(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.model = None
        def forward(self, x):
            return self.model(x) if self.model else x
    
    class ClassificationHead(nn.Module):
        def __init__(self, input_dim=512, num_classes=10, normalize=True):
            super().__init__()
            self.input_dim = input_dim
            self.num_classes = num_classes
            self.normalize = normalize
            self.weight = nn.Parameter(torch.randn(num_classes, input_dim))
            self.bias = nn.Parameter(torch.zeros(num_classes))
        def forward(self, x):
            if self.normalize:
                x = nn.functional.normalize(x, dim=-1)
            return nn.functional.linear(x, self.weight, self.bias)
    
    src = types.ModuleType('src')
    src_modeling = types.ModuleType('src.modeling')
    src_models = types.ModuleType('src.models')
    src_models_modeling = types.ModuleType('src.models.modeling')
    
    src_modeling.ImageEncoder = ImageEncoder
    src_modeling.ClassificationHead = ClassificationHead
    src_models_modeling.ClassificationHead = ClassificationHead
    
    src.modeling = src_modeling
    src.models = src_models
    src_models.modeling = src_models_modeling
    
    sys.modules['src'] = src
    sys.modules['src.modeling'] = src_modeling
    sys.modules['src.models'] = src_models
    sys.modules['src.models.modeling'] = src_models_modeling
    print("✓ Mocked src.modeling", flush=True)

setup_src_mock()

# =============================================================================
# Templates
# =============================================================================
OPENAI_IMAGENET_TEMPLATES = [
    "a bad photo of a {}.", "a photo of many {}.", "a sculpture of a {}.",
    "a photo of the hard to see {}.", "a low resolution photo of the {}.",
    "a rendering of a {}.", "graffiti of a {}.", "a bad photo of the {}.",
    "a cropped photo of the {}.", "a tattoo of a {}.", "the embroidered {}.",
    "a photo of a hard to see {}.", "a bright photo of a {}.",
    "a photo of a clean {}.", "a photo of a dirty {}.", "a dark photo of the {}.",
    "a drawing of a {}.", "a photo of my {}.", "the plastic {}.",
    "a photo of the cool {}.", "a close-up photo of a {}.",
    "a black and white photo of the {}.", "a painting of the {}.",
    "a painting of a {}.", "a pixelated photo of the {}.", "a sculpture of the {}.",
    "a bright photo of the {}.", "a cropped photo of a {}.", "a plastic {}.",
    "a photo of the dirty {}.", "a jpeg corrupted photo of a {}.",
    "a blurry photo of the {}.", "a photo of the {}.", "a good photo of the {}.",
    "a rendering of the {}.", "a {} in a video game.", "a photo of one {}.",
    "a doodle of a {}.", "a close-up photo of the {}.", "a photo of a {}.",
    "the origami {}.", "the {} in a video game.", "a sketch of a {}.",
    "a doodle of the {}.", "a origami {}.", "a low resolution photo of a {}.",
    "the toy {}.", "a rendition of the {}.", "a photo of the clean {}.",
    "a photo of a large {}.", "a rendition of a {}.", "a photo of a nice {}.",
    "a photo of a weird {}.", "a blurry photo of a {}.", "a cartoon {}.",
    "art of a {}.", "a sketch of the {}.", "a embroidered {}.",
    "a pixelated photo of a {}.", "itap of the {}.",
    "a jpeg corrupted photo of the {}.", "a good photo of a {}.", "a plushie {}.",
    "a photo of the nice {}.", "a photo of the small {}.",
    "a photo of the weird {}.", "the cartoon {}.", "art of the {}.",
    "a drawing of the {}.", "a photo of the large {}.",
    "a black and white photo of a {}.", "the plushie {}.", "a dark photo of a {}.",
    "itap of a {}.", "graffiti of the {}.", "a toy {}.", "itap of my {}.",
    "a photo of a cool {}.", "a photo of a small {}.", "a tattoo of the {}.",
]

# =============================================================================
# Helper functions
# =============================================================================
def get_imagenet_classnames():
    import urllib.request
    import json
    url = "https://raw.githubusercontent.com/raghakot/keras-vis/master/resources/imagenet_class_index.json"
    response = urllib.request.urlopen(url)
    class_idx = json.loads(response.read().decode())
    classnames = [class_idx[str(i)][1].replace("_", " ") for i in range(1000)]
    print(f"Loaded {len(classnames)} classnames", flush=True)
    return classnames

def load_imagenet_val(parquet_dir):
    parquet_files = sorted([
        os.path.join(parquet_dir, f)
        for f in os.listdir(parquet_dir)
        if f.startswith("validation") and f.endswith(".parquet")
    ])
    print(f"Loading {len(parquet_files)} parquet files...", flush=True)
    ds = Dataset.from_parquet(parquet_files)
    print(f"Loaded {len(ds)} images", flush=True)
    return ds

def get_text_features(model, tokenizer, classnames, templates):
    with torch.no_grad():
        text_features = []
        for classname in tqdm(classnames, desc="Encoding text"):
            texts = [t.format(classname) for t in templates]
            texts = tokenizer(texts).to(device)
            class_embeddings = model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            text_features.append(class_embedding)
        text_features = torch.stack(text_features, dim=0)
    return text_features

def load_finetuned_clip(model_name, checkpoint_path):
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained='openai')
    tokenizer = open_clip.get_tokenizer(model_name)
    
    print(f"  Loading checkpoint...", flush=True)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Extract state dict from ImageEncoder wrapper
    if isinstance(checkpoint, nn.Module):
        if hasattr(checkpoint, 'model') and checkpoint.model is not None:
            full_sd = checkpoint.model.state_dict()
        else:
            full_sd = checkpoint.state_dict()
    else:
        full_sd = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
    
    # Extract ONLY visual encoder keys (they start with 'visual.')
    visual_sd = {}
    for k, v in full_sd.items():
        if k.startswith('visual.'):
            new_k = k[7:]  # Remove 'visual.' prefix
            visual_sd[new_k] = v.float()
    
    print(f"  Extracted {len(visual_sd)} visual params", flush=True)
    model.visual.load_state_dict(visual_sd, strict=True)
    print(f"  ✓ Loaded successfully", flush=True)
    
    return model, preprocess, tokenizer

def evaluate_model(model, preprocess, tokenizer, dataset, classnames, templates, batch_size=256):
    model = model.to(device).eval()
    
    print("  Computing text features...", flush=True)
    text_features = get_text_features(model, tokenizer, classnames, templates)
    
    correct = 0
    total = 0
    
    print("  Evaluating images...", flush=True)
    for i in tqdm(range(0, len(dataset), batch_size)):
        batch = dataset[i:i+batch_size]
        
        images = []
        for img in batch['image']:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            images.append(preprocess(img))
        
        images = torch.stack(images).to(device)
        labels = torch.tensor(batch['label']).to(device)
        
        with torch.no_grad():
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = image_features @ text_features.T
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += len(labels)
    
    return correct / total * 100

def main():
    parquet_dir = "/data8/rajiv/mariamma/sidharth/atlas/atlas/data/imagenet_val/data"
    
    FINETUNED_MODELS = {
        "ViT-B-16": "/data8/rajiv/mariamma/sidharth/atlas/atlas/checkpoints/ViT-B-16/ImageNetVal/finetuned.pt",
        "ViT-B-32": "/data8/rajiv/mariamma/sidharth/atlas/atlas/checkpoints/ViT-B-32/ImageNetVal/finetuned.pt",
        "ViT-L-14": "/data8/rajiv/mariamma/sidharth/atlas/atlas/checkpoints/ViT-L-14/ImageNetVal/finetuned.pt",
    }
    
    ds = load_imagenet_val(parquet_dir)
    classnames = get_imagenet_classnames()
    templates = OPENAI_IMAGENET_TEMPLATES
    
    results = {}
    
    print("\n" + "="*60, flush=True)
    print("FINETUNED MODELS", flush=True)
    print("="*60, flush=True)
    
    for model_name, ckpt_path in FINETUNED_MODELS.items():
        if not os.path.exists(ckpt_path):
            print(f"\nSkipping {model_name} - not found", flush=True)
            continue
        
        print(f"\n>>> {model_name}", flush=True)
        model, preprocess, tokenizer = load_finetuned_clip(model_name, ckpt_path)
        
        acc = evaluate_model(model, preprocess, tokenizer, ds, classnames, templates)
        results[f"{model_name}_finetuned"] = acc
        print(f"  Accuracy: {acc:.2f}%", flush=True)
        
        del model
        torch.cuda.empty_cache()
    
    print("\n" + "="*60, flush=True)
    print("FINAL RESULTS", flush=True)
    print("="*60, flush=True)
    for name, acc in sorted(results.items()):
        print(f"{name}: {acc:.2f}%", flush=True)

if __name__ == "__main__":
    main()
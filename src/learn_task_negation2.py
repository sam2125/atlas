# """Learn coefficients on a task vector for task negation,
# using a supervised objective with gradient ascent on the
# target dataset and gradient descent on the control datasets.

# Modified to use remaining 7 tasks as control instead of ImageNet.
# """
# import os
# import time
# import json
# import torch

# from torch.cuda.amp import GradScaler
# from src.linearize import LinearizedImageEncoder
# from src.modeling import ImageEncoder, MultiHeadImageClassifier
# from src.task_vectors import LinearizedTaskVector, NonLinearTaskVector
# from src.composition import WeightedImageEncoder, WeightedLinearizedModel

# from src.args import parse_arguments
# from src.eval import eval_single_dataset
# from src.datasets.registry import get_dataset
# from src.heads import get_classification_head
# from src.datasets.common import get_dataloader, maybe_dictionarize
# from src.distributed import cleanup_ddp, distribute_loader, is_main_process, setup_ddp

# def avg(x):
#     return sum(x) / len(x)

# def lp_reg(x, p=None, gamma=0.5) -> torch.Tensor:
#     return 0 if p is None else gamma * torch.norm(x, p=p, dim=0).mean()

# def main(rank, args):

#     setup_ddp(rank, args.world_size, port=args.port)

#     tgt_dataset = args.tgt_dataset
#     ctr_datasets = args.ctr_datasets  # Now a list of 7 datasets

#     ckpdir = os.path.join(args.save, tgt_dataset)
#     os.makedirs(ckpdir, exist_ok=True)

#     assert args.finetuning_mode in [
#         "linear", "standard",
#     ], "Only linear and standard fine-tuning are supported."

#     linearized_finetuning = args.finetuning_mode == "linear"
#     if linearized_finetuning:
#         print("Using linearized fine-tuning.")

#     ft_path = (
#         os.path.join(args.save, tgt_dataset, "linear_finetuned.pt")
#         if linearized_finetuning
#         else os.path.join(args.save, tgt_dataset, "finetuned.pt")
#     )
#     zs_path = (
#         os.path.join(args.save, tgt_dataset, "linear_zeroshot.pt")
#         if linearized_finetuning
#         else os.path.join(args.save, tgt_dataset, "zeroshot.pt")
#     )
#     if not os.path.exists(zs_path):
#         raise ValueError(f"The checkpoint for the zero-shot model does not exist at {zs_path}.")
#     if not os.path.exists(ft_path):
#         raise ValueError(f"The checkpoint for the fine-tuned model does not exist at {ft_path}.")

#     if args.finetuning_mode == "linear":
#         task_vectors = [LinearizedTaskVector(zs_path, ft_path),]
#         image_encoder = LinearizedImageEncoder(args, keep_lang=False)
#         image_encoder.model = WeightedLinearizedModel(
#             image_encoder.model, task_vectors, blockwise=args.blockwise_coef
#         )
#     else:
#         task_vectors = [NonLinearTaskVector(zs_path, ft_path),]
#         image_encoder = ImageEncoder(args)
#         image_encoder = WeightedImageEncoder(
#             image_encoder, task_vectors, blockwise=args.blockwise_coef
#         )

#     # Get classification heads for target and all control datasets
#     tgt_classification_head = get_classification_head(args, tgt_dataset)
#     ctr_classification_heads = [get_classification_head(args, ctr_dataset) for ctr_dataset in ctr_datasets]
    
#     # All heads: target first, then all controls
#     all_heads = [tgt_classification_head] + ctr_classification_heads
#     model = MultiHeadImageClassifier(image_encoder, all_heads)

#     model.freeze_head()
#     model = model.cuda()

#     preprocess_fn = model.train_preprocess
    
#     # Calculate batch size per dataset
#     n_ctr_datasets = len(ctr_datasets)
#     batch_size_tgt = int(args.batch_size / 2)
#     batch_size_ctr = int(args.batch_size / 2 / n_ctr_datasets)
    
#     tgt_dataloader = get_dataloader(
#         get_dataset(
#             tgt_dataset, preprocess_fn,
#             location=args.data_location,
#             batch_size=batch_size_tgt,
#             num_workers=2),
#         is_train=False, args=args, image_encoder=None
#     )
    
#     # Create dataloaders for all control datasets
#     ctr_dataloaders = [get_dataloader(
#         get_dataset(
#             ctr_dataset, preprocess_fn,
#             location=args.data_location,
#             batch_size=batch_size_ctr,
#             num_workers=2),
#         is_train=False, args=args, image_encoder=None
#     ) for ctr_dataset in ctr_datasets]
    
#     num_batches = len(tgt_dataloader)
    
#     # Printing loss between four and ten times an epoch
#     if args.print_every * 10 < num_batches:
#         print_every = int(num_batches / 10)
#     elif args.print_every * 4 > num_batches:
#         print_every = max(int(num_batches / 4), 1)
#     else:
#         print_every = args.print_every

#     # Distribute the data and model across the GPUs.
#     ddp_tgt_loader = distribute_loader(tgt_dataloader)
#     ddp_ctr_loaders = [distribute_loader(ctr_dataloader) for ctr_dataloader in ctr_dataloaders]
#     ddp_model = torch.nn.parallel.DistributedDataParallel(
#         model,
#         device_ids=[rank],
#         find_unused_parameters=False,
#         output_device=rank,
#     )

#     loss_fn = torch.nn.CrossEntropyLoss()

#     params = [p for p in ddp_model.parameters() if p.requires_grad]
#     optimizer = torch.optim.AdamW(params, lr=args.lr * args.lr_multiplier, weight_decay=args.wd)

#     if linearized_finetuning:
#         head_path = os.path.join(ckpdir, "learned_linear_negations_7task.pt")
#         log_path = os.path.join(args.save, "learned_linear_negations_7task.json")
#         coef = ddp_model.module.image_encoder.model.coef
#     else:
#         head_path = os.path.join(ckpdir, "learned_negations_7task.pt")
#         log_path = os.path.join(args.save, "learned_negations_7task.json")
#         coef = ddp_model.module.image_encoder.coef
#     if isinstance(args.subsample, int):
#         raise NotImplementedError(f"Option for {args.subsample}-shot is not implemented.")
#     elif args.subsample < 1.0:
#         head_path = head_path[:-3] + f"_{args.subsample*100:.0f}perc.pt"
#         log_path = log_path[:-5] + f"_{args.subsample*100:.0f}perc.json"

#     scaler = GradScaler()
#     tgt_zs_acc = args.zs_acc[tgt_dataset]
#     best_acc = tgt_zs_acc
#     ctr_zs_accs = {ctr_dataset: args.zs_acc[ctr_dataset] for ctr_dataset in ctr_datasets}
    
#     # if is_main_process():
#     #     print(f"=> Zero-shot accuracy on {tgt_dataset} (target): {100*tgt_zs_acc:.2f}%.")
#     #     for ctr_dataset in ctr_datasets:
#     #         print(f"=> Zero-shot accuracy on {ctr_dataset} (control): {100*ctr_zs_accs[ctr_dataset]:.2f}%.")
#     #     if os.path.exists(log_path):
#     #         with open(log_path) as f:
#     #             negation_acc = json.load(f)
#     #     else:
#     #         negation_acc = {}

#     # best_coef = None
#     # negation_acc[tgt_dataset] = {}
#     # val_acc = []

#     if is_main_process():
#         print(f"=> Zero-shot accuracy on {tgt_dataset} (target): {100*tgt_zs_acc:.2f}%.")
#         ...
#         if os.path.exists(log_path):
#             with open(log_path) as f:
#                 negation_acc = json.load(f)
#         else:
#             negation_acc = {}
#         negation_acc[tgt_dataset] = {}   # <-- Move inside the if block

#     best_coef = None
#     val_acc = []
    
#     for epoch in range(args.epoch):
    
#         ddp_tgt_loader.sampler.set_epoch(epoch)
#         for ddp_ctr_loader in ddp_ctr_loaders:
#             ddp_ctr_loader.sampler.set_epoch(epoch)
#         ctr_iters = [iter(ddp_ctr_loader) for ddp_ctr_loader in ddp_ctr_loaders]
        
#         for i, batch in enumerate(ddp_tgt_loader):
#             # Get batches from all control datasets
#             ctr_batches = [next(ctr_iter) for ctr_iter in ctr_iters]
#             start_time = time.time()

#             step = (
#                 i // args.num_grad_accumulation
#                 + epoch * num_batches // args.num_grad_accumulation
#             )

#             batch = maybe_dictionarize(batch)
#             ctr_batches = [maybe_dictionarize(ctr_batch) for ctr_batch in ctr_batches]
            
#             # Concatenate all inputs: target + all controls
#             inputs = torch.cat(
#                 [batch["images"].cuda()] + 
#                 [ctr_batch["images"].cuda() for ctr_batch in ctr_batches]
#             )
#             data_time = time.time() - start_time
            
#             # Split sizes for each head
#             split = [len(batch["images"])] + [len(ctr_batch["images"]) for ctr_batch in ctr_batches]
            
#             with torch.autocast(device_type='cuda', dtype=torch.float16):
#                 logits = ddp_model(inputs, split)
#                 labels = [batch["labels"].cuda()] + [ctr_batch["labels"].cuda() for ctr_batch in ctr_batches]
                
#                 # Compute losses
#                 all_losses = [loss_fn(x, y) for x, y in zip(logits, labels)]
#                 loss_tgt = all_losses[0]
#                 losses_ctr = all_losses[1:]
                
#                 """Gradient ascent on the target dataset,
#                 gradient descent on the control datasets (average of 7 tasks)."""
#                 loss = -loss_tgt + avg(losses_ctr)
                
#                 # Apply regularisation if needed.
#                 reg = lp_reg(coef, args.lp_reg)
#                 loss = loss + reg
#                 # Scale the loss
#                 loss = loss / args.num_grad_accumulation

#             scaler.scale(loss).backward()

#             if i % args.num_grad_accumulation == 0:

#                 torch.nn.utils.clip_grad_norm_(params, 1.0)
#                 scaler.step(optimizer)
#                 scaler.update()
#                 optimizer.zero_grad()

#             batch_time = time.time() - start_time

#             if (
#                 step % print_every == 0
#                 and (i % args.num_grad_accumulation == 0)
#                 and is_main_process()
#             ):
#                 percent_complete = 100 * i / len(ddp_tgt_loader)
#                 ctr_losses_str = [f"{l.item():.4f}" for l in losses_ctr]
#                 print(
#                     f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{len(ddp_tgt_loader)}]\t"
#                     f"Loss (tgt.): {loss_tgt.item():.6f}\tLoss (ctr. avg): {avg([l.item() for l in losses_ctr]):.6f}\t"
#                     f"Data (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}",
#                     flush=True,
#                 )

#         if is_main_process():
#             # Evaluate on all datasets
#             image_encoder = ddp_model.module.image_encoder
#             tgt_acc = eval_single_dataset(image_encoder, tgt_dataset, args)["top1"]
#             ctr_accs = {
#                 ctr_dataset: eval_single_dataset(image_encoder, ctr_dataset, args)["top1"]
#                 for ctr_dataset in ctr_datasets
#             }

#             # Check if all control datasets maintain threshold
#             all_ctr_above_threshold = all(
#                 ctr_accs[ctr_dataset] >= ctr_zs_accs[ctr_dataset] * args.control_threshold
#                 for ctr_dataset in ctr_datasets
#             )

#             # Save the best coefficients.
#             if tgt_acc < best_acc and all_ctr_above_threshold:
#                 best_acc = tgt_acc
#                 best_coef = coef.data.clone()
#                 torch.save(best_coef, head_path)
            
#             # Log validation accuracies
#             epoch_val_acc = {
#                 f"{tgt_dataset}:top1": tgt_acc,
#                 f"{tgt_dataset}:normalised_top1": tgt_acc / args.zs_acc[tgt_dataset],
#             }
#             for ctr_dataset in ctr_datasets:
#                 epoch_val_acc[f"{ctr_dataset}:top1"] = ctr_accs[ctr_dataset]
#                 epoch_val_acc[f"{ctr_dataset}:normalised_top1"] = ctr_accs[ctr_dataset] / args.zs_acc[ctr_dataset]
#             epoch_val_acc["avg_ctr:top1"] = avg(ctr_accs.values())
#             epoch_val_acc["avg_ctr:normalised_top1"] = avg([
#                 ctr_accs[ctr_dataset] / args.zs_acc[ctr_dataset] 
#                 for ctr_dataset in ctr_datasets
#             ])
#             val_acc.append(epoch_val_acc)

#             # Print epoch summary
#             print(f"Epoch {epoch} - Target {tgt_dataset}: {100*tgt_acc:.2f}%, Avg Control: {100*avg(ctr_accs.values()):.2f}%")

#         # Early stopping if any control drops below threshold
#         if not all_ctr_above_threshold:
#             if is_main_process():
#                 print(f"Early stopping: control dataset(s) dropped below threshold.")
#             break

#     # Log stats and test the model with the optimal coefficients.
#     if is_main_process():
#         print("-" * 100)
#         print("=> Start evaluation on test set.")
#         negation_acc[tgt_dataset]["val"] = val_acc
        
#         if best_coef is None:
#             print("Warning: No valid coefficients found, using final coefficients.")
#             best_coef = coef.data.clone()
        
#         image_encoder = ddp_model.module.image_encoder
#         if linearized_finetuning:
#             image_encoder.model.coef = torch.nn.Parameter(best_coef)
#         else:
#             image_encoder.coef = torch.nn.Parameter(best_coef)
        
#         # Test on target (without Val suffix)
#         tgt_test_name = tgt_dataset.replace("Val", "")
#         negation_acc[tgt_dataset]["test"] = eval_single_dataset(
#             image_encoder, tgt_test_name, args
#         )["top1"]
        
#         # Test on all control datasets
#         negation_acc[tgt_dataset]["test_controls"] = {}
#         for ctr_dataset in ctr_datasets:
#             ctr_test_name = ctr_dataset.replace("Val", "")
#             negation_acc[tgt_dataset]["test_controls"][ctr_test_name] = eval_single_dataset(
#                 image_encoder, ctr_test_name, args
#             )["top1"]
#         negation_acc[tgt_dataset]["test_controls_avg"] = avg(
#             negation_acc[tgt_dataset]["test_controls"].values()
#         )
        
#         # Remove the "Val" suffix in the dict keys
#         negation_acc = {k.replace("Val", ""): v for k, v in negation_acc.items()}
#         with open(log_path, 'w') as f:
#             json.dump(negation_acc, f, indent=4)

#     cleanup_ddp()

# if __name__ == "__main__":

#     # {num_epochs, lr_multiplier}
#     # NOTE: These hyper-parameters are tuned on the validation sets with ViT-B-32
#     # and may need more tuning for other backbones.
#     all_datasets = ["Cars", "DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SUN397", "SVHN"]
    
#     hyperparams = {
#         "Cars": [20, 5],
#         "DTD": [20, 10],
#         "EuroSAT": [3, 5],
#         "GTSRB": [5, 5],
#         "MNIST": [7, 3],
#         "RESISC45": [10, 2],
#         "SUN397": [10, 3],
#         "SVHN": [2, 5],
#     }

#     args = parse_arguments()
#     # We use gradient accumulation to simulate larger batch sizes if the model does not fit in memory.
#     args.batch_size = 64 if args.model == "ViT-L-14" else 128
#     args.num_grad_accumulation = 2 if args.model == "ViT-L-14" else 1
#     args.print_every = 10
#     if args.seed is not None:
#         args.save = f"checkpoints_{args.seed}/{args.model}"
#     else:
#         args.save = f"checkpoints/{args.model}"
#     with open(os.path.join(args.save, "zeroshot_accuracies.json"), 'r') as f:
#         args.zs_acc = json.load(f)
#     with open(os.path.join(args.save, "ft_accuracies.json"), 'r') as f:
#         args.ft_acc = json.load(f)

#     for dataset in all_datasets:
#         args.tgt_dataset = dataset + "Val"
#         # Control datasets are the remaining 7 tasks
#         args.ctr_datasets = [d + "Val" for d in all_datasets if d != dataset]
#         args.epoch, args.lr_multiplier = hyperparams[dataset]
#         print("=" * 100)
#         print(f"Learn task vector coefficients of {args.model} for negating {dataset}")
#         print(f"Control datasets: {', '.join([d.replace('Val', '') for d in args.ctr_datasets])}")
#         print("=" * 100)
#         torch.multiprocessing.spawn(main, args=(args,), nprocs=args.world_size)


"""Learn coefficients on a task vector for task negation,
using a supervised objective with gradient ascent on the
target dataset and gradient descent on the control datasets.

Modified to use remaining 7 tasks as control instead of ImageNet.
Fixed for multi-GPU training.
"""
import os
import time
import json
import torch

from torch.cuda.amp import GradScaler
from src.linearize import LinearizedImageEncoder
from src.modeling import ImageEncoder, MultiHeadImageClassifier
from src.task_vectors import LinearizedTaskVector, NonLinearTaskVector
from src.composition import WeightedImageEncoder, WeightedLinearizedModel

from src.args import parse_arguments
from src.eval import eval_single_dataset
from src.datasets.registry import get_dataset
from src.heads import get_classification_head
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.distributed import cleanup_ddp, distribute_loader, is_main_process, setup_ddp

def avg(x):
    return sum(x) / len(x)

def lp_reg(x, p=None, gamma=0.5) -> torch.Tensor:
    return 0 if p is None else gamma * torch.norm(x, p=p, dim=0).mean()

def main(rank, args):

    setup_ddp(rank, args.world_size, port=args.port)

    tgt_dataset = args.tgt_dataset
    ctr_datasets = args.ctr_datasets  # Now a list of 7 datasets

    ckpdir = os.path.join(args.save, tgt_dataset)
    os.makedirs(ckpdir, exist_ok=True)

    assert args.finetuning_mode in [
        "linear", "standard",
    ], "Only linear and standard fine-tuning are supported."

    linearized_finetuning = args.finetuning_mode == "linear"
    if linearized_finetuning:
        print("Using linearized fine-tuning.")

    ft_path = (
        os.path.join(args.save, tgt_dataset, "linear_finetuned.pt")
        if linearized_finetuning
        else os.path.join(args.save, tgt_dataset, "finetuned.pt")
    )
    zs_path = (
        os.path.join(args.save, tgt_dataset, "linear_zeroshot.pt")
        if linearized_finetuning
        else os.path.join(args.save, tgt_dataset, "zeroshot.pt")
    )
    if not os.path.exists(zs_path):
        raise ValueError(f"The checkpoint for the zero-shot model does not exist at {zs_path}.")
    if not os.path.exists(ft_path):
        raise ValueError(f"The checkpoint for the fine-tuned model does not exist at {ft_path}.")

    if args.finetuning_mode == "linear":
        task_vectors = [LinearizedTaskVector(zs_path, ft_path),]
        image_encoder = LinearizedImageEncoder(args, keep_lang=False)
        image_encoder.model = WeightedLinearizedModel(
            image_encoder.model, task_vectors, blockwise=args.blockwise_coef
        )
    else:
        task_vectors = [NonLinearTaskVector(zs_path, ft_path),]
        image_encoder = ImageEncoder(args)
        image_encoder = WeightedImageEncoder(
            image_encoder, task_vectors, blockwise=args.blockwise_coef
        )

    # Get classification heads for target and all control datasets
    tgt_classification_head = get_classification_head(args, tgt_dataset)
    ctr_classification_heads = [get_classification_head(args, ctr_dataset) for ctr_dataset in ctr_datasets]
    
    # All heads: target first, then all controls
    all_heads = [tgt_classification_head] + ctr_classification_heads
    model = MultiHeadImageClassifier(image_encoder, all_heads)

    model.freeze_head()
    model = model.cuda()

    preprocess_fn = model.train_preprocess
    
    # Calculate batch size per dataset
    n_ctr_datasets = len(ctr_datasets)
    batch_size_tgt = int(args.batch_size / 2)
    batch_size_ctr = int(args.batch_size / 2 / n_ctr_datasets)
    
    tgt_dataloader = get_dataloader(
        get_dataset(
            tgt_dataset, preprocess_fn,
            location=args.data_location,
            batch_size=batch_size_tgt,
            num_workers=2),
        is_train=False, args=args, image_encoder=None
    )
    
    # Create dataloaders for all control datasets
    ctr_dataloaders = [get_dataloader(
        get_dataset(
            ctr_dataset, preprocess_fn,
            location=args.data_location,
            batch_size=batch_size_ctr,
            num_workers=2),
        is_train=False, args=args, image_encoder=None
    ) for ctr_dataset in ctr_datasets]
    
    num_batches = len(tgt_dataloader)
    
    # Printing loss between four and ten times an epoch
    if args.print_every * 10 < num_batches:
        print_every = int(num_batches / 10)
    elif args.print_every * 4 > num_batches:
        print_every = max(int(num_batches / 4), 1)
    else:
        print_every = args.print_every

    # Distribute the data and model across the GPUs.
    ddp_tgt_loader = distribute_loader(tgt_dataloader)
    ddp_ctr_loaders = [distribute_loader(ctr_dataloader) for ctr_dataloader in ctr_dataloaders]
    ddp_model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[rank],
        find_unused_parameters=False,
        output_device=rank,
    )

    loss_fn = torch.nn.CrossEntropyLoss()

    params = [p for p in ddp_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr * args.lr_multiplier, weight_decay=args.wd)

    if linearized_finetuning:
        head_path = os.path.join(ckpdir, "learned_linear_negations_7task.pt")
        log_path = os.path.join(args.save, "learned_linear_negations_7task.json")
        coef = ddp_model.module.image_encoder.model.coef
    else:
        head_path = os.path.join(ckpdir, "learned_negations_7task_with_early_stop.pt")
        log_path = os.path.join(args.save, "learned_negations_7task_with_early_stop.json")
        coef = ddp_model.module.image_encoder.coef
    if isinstance(args.subsample, int):
        raise NotImplementedError(f"Option for {args.subsample}-shot is not implemented.")
    elif args.subsample < 1.0:
        head_path = head_path[:-3] + f"_{args.subsample*100:.0f}perc.pt"
        log_path = log_path[:-5] + f"_{args.subsample*100:.0f}perc.json"

    scaler = GradScaler()
    tgt_zs_acc = args.zs_acc[tgt_dataset]
    best_acc = tgt_zs_acc
    ctr_zs_accs = {ctr_dataset: args.zs_acc[ctr_dataset] for ctr_dataset in ctr_datasets}
    
    # Initialize on main process only
    if is_main_process():
        print(f"=> Zero-shot accuracy on {tgt_dataset} (target): {100*tgt_zs_acc:.2f}%.")
        for ctr_dataset in ctr_datasets:
            print(f"=> Zero-shot accuracy on {ctr_dataset} (control): {100*ctr_zs_accs[ctr_dataset]:.2f}%.")
        if os.path.exists(log_path):
            with open(log_path) as f:
                negation_acc = json.load(f)
        else:
            negation_acc = {}
        negation_acc[tgt_dataset] = {}

    best_coef = None
    val_acc = []
    all_ctr_above_threshold = True  # Initialize for ALL processes
    # best_score = float('inf')
    
    for epoch in range(args.epoch):
    
        ddp_tgt_loader.sampler.set_epoch(epoch)
        for ddp_ctr_loader in ddp_ctr_loaders:
            ddp_ctr_loader.sampler.set_epoch(epoch)
        ctr_iters = [iter(ddp_ctr_loader) for ddp_ctr_loader in ddp_ctr_loaders]
        
        for i, batch in enumerate(ddp_tgt_loader):
            # Get batches from all control datasets
            ctr_batches =[]
            for j, ctr_iter in enumerate(ctr_iters):
                try:
                    ctr_batch = next(ctr_iter)
                except StopIteration:
                    ctr_iters[j] = iter(ddp_ctr_loaders[j])
                    ctr_batch = next(ctr_iters[j])
                ctr_batches.append(ctr_batch)
            # ctr_batches = [next(ctr_iter) for ctr_iter in ctr_iters]
            start_time = time.time()

            step = (
                i // args.num_grad_accumulation
                + epoch * num_batches // args.num_grad_accumulation
            )

            batch = maybe_dictionarize(batch)
            ctr_batches = [maybe_dictionarize(ctr_batch) for ctr_batch in ctr_batches]
            
            # Concatenate all inputs: target + all controls
            inputs = torch.cat(
                [batch["images"].cuda()] + 
                [ctr_batch["images"].cuda() for ctr_batch in ctr_batches]
            )
            data_time = time.time() - start_time
            
            # Split sizes for each head
            split = [len(batch["images"])] + [len(ctr_batch["images"]) for ctr_batch in ctr_batches]
            
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits = ddp_model(inputs, split)
                labels = [batch["labels"].cuda()] + [ctr_batch["labels"].cuda() for ctr_batch in ctr_batches]
                
                # Compute losses
                all_losses = [loss_fn(x, y) for x, y in zip(logits, labels)]
                loss_tgt = all_losses[0]
                losses_ctr = all_losses[1:]
                
                """Gradient ascent on the target dataset,
                gradient descent on the control datasets (average of 7 tasks)."""
                loss = -loss_tgt + avg(losses_ctr)
                
                # Apply regularisation if needed.
                reg = lp_reg(coef, args.lp_reg)
                loss = loss + reg
                # Scale the loss
                loss = loss / args.num_grad_accumulation

            scaler.scale(loss).backward()

            if i % args.num_grad_accumulation == 0:

                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            batch_time = time.time() - start_time

            if (
                step % print_every == 0
                and (i % args.num_grad_accumulation == 0)
                and is_main_process()
            ):
                percent_complete = 100 * i / len(ddp_tgt_loader)
                ctr_losses_str = [f"{l.item():.4f}" for l in losses_ctr]
                print(
                    f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{len(ddp_tgt_loader)}]\t"
                    f"Loss (tgt.): {loss_tgt.item():.6f}\tLoss (ctr. avg): {avg([l.item() for l in losses_ctr]):.6f}\t"
                    f"Data (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}",
                    flush=True,
                )

        # Evaluation on main process
        if is_main_process():
            image_encoder = ddp_model.module.image_encoder
            tgt_acc = eval_single_dataset(image_encoder, tgt_dataset, args)["top1"]
            ctr_accs = {
                ctr_dataset: eval_single_dataset(image_encoder, ctr_dataset, args)["top1"]
                for ctr_dataset in ctr_datasets
            }

            # Check if all control datasets maintain threshold
            all_ctr_above_threshold = all(
                ctr_accs[ctr_dataset] >= ctr_zs_accs[ctr_dataset] * args.control_threshold
                for ctr_dataset in ctr_datasets
            )

            # Save the best coefficients.
            if tgt_acc < best_acc and all_ctr_above_threshold:
                best_acc = tgt_acc
                best_coef = coef.data.clone()
                torch.save(best_coef, head_path)
            
            # all_ctr_above_threshold = all(
            #     ctr_accs[ctr_dataset] >= ctr_zs_accs[ctr_dataset] * args.control_thresholds[ctr_dataset]
            #     for ctr_dataset in ctr_datasets
            # )

            # # Calculate combined score (lower is better)
            # # Normalized: target should go down, controls should stay high
            # tgt_normalized = tgt_acc / tgt_zs_acc  # <1 means negation working
            # ctr_normalized = avg([
            #     ctr_accs[ctr_dataset] / ctr_zs_accs[ctr_dataset]
            #     for ctr_dataset in ctr_datasets
            # ])  # ~1 means controls preserved
            
            # # Combined score: want low target, high control
            # # alpha controls tradeoff (higher = more control preservation)
            # alpha = args.alpha if hasattr(args, 'alpha') else 0.5
            # combined_score = tgt_normalized - alpha * ctr_normalized
            
            # # Save the best coefficients based on combined score
            # if combined_score < best_score:
            #     best_score = combined_score
            #     best_acc = tgt_acc
            #     best_coef = coef.data.clone()
            #     torch.save(best_coef, head_path)
            #     if is_main_process():
            #         print(f"  → New best! Score: {combined_score:.4f} (tgt_norm: {tgt_normalized:.3f}, ctr_norm: {ctr_normalized:.3f})")

            # Log validation accuracies
            epoch_val_acc = {
                f"{tgt_dataset}:top1": tgt_acc,
                f"{tgt_dataset}:normalised_top1": tgt_acc / args.zs_acc[tgt_dataset],
            }
            for ctr_dataset in ctr_datasets:
                epoch_val_acc[f"{ctr_dataset}:top1"] = ctr_accs[ctr_dataset]
                epoch_val_acc[f"{ctr_dataset}:normalised_top1"] = ctr_accs[ctr_dataset] / args.zs_acc[ctr_dataset]
            epoch_val_acc["avg_ctr:top1"] = avg(ctr_accs.values())
            epoch_val_acc["avg_ctr:normalised_top1"] = avg([
                ctr_accs[ctr_dataset] / args.zs_acc[ctr_dataset] 
                for ctr_dataset in ctr_datasets
            ])
            val_acc.append(epoch_val_acc)

            # Print epoch summary
            print(f"Epoch {epoch} - Target {tgt_dataset}: {100*tgt_acc:.2f}%, Avg Control: {100*avg(ctr_accs.values()):.2f}%")

        # Sync all processes after evaluation
        torch.distributed.barrier()
        
        # Broadcast early stopping decision to all ranks
        stop_flag = torch.zeros(1, device='cuda')
        if is_main_process():
            if not all_ctr_above_threshold:
                print(f"Early stopping: control dataset(s) dropped below threshold.")
                stop_flag[0] = 1
        torch.distributed.broadcast(stop_flag, src=0)
        if stop_flag.item() == 1:
            break

    # Log stats and test the model with the optimal coefficients.
    if is_main_process():
        print("-" * 100)
        print("=> Start evaluation on test set.")
        negation_acc[tgt_dataset]["val"] = val_acc
        
        if best_coef is None:
            print("Warning: No valid coefficients found, using final coefficients.")
            best_coef = coef.data.clone()
        
        image_encoder = ddp_model.module.image_encoder
        if linearized_finetuning:
            image_encoder.model.coef = torch.nn.Parameter(best_coef)
        else:
            image_encoder.coef = torch.nn.Parameter(best_coef)
        
        # Test on target (without Val suffix)
        tgt_test_name = tgt_dataset.replace("Val", "")
        negation_acc[tgt_dataset]["test"] = eval_single_dataset(
            image_encoder, tgt_test_name, args
        )["top1"]
        
        # Test on all control datasets
        negation_acc[tgt_dataset]["test_controls"] = {}
        for ctr_dataset in ctr_datasets:
            ctr_test_name = ctr_dataset.replace("Val", "")
            negation_acc[tgt_dataset]["test_controls"][ctr_test_name] = eval_single_dataset(
                image_encoder, ctr_test_name, args
            )["top1"]
        negation_acc[tgt_dataset]["test_controls_avg"] = avg(
            negation_acc[tgt_dataset]["test_controls"].values()
        )
        
        # Remove the "Val" suffix in the dict keys
        negation_acc = {k.replace("Val", ""): v for k, v in negation_acc.items()}
        with open(log_path, 'w') as f:
            json.dump(negation_acc, f, indent=4)

    cleanup_ddp()

if __name__ == "__main__":

    all_datasets = ["Cars", "DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SUN397", "SVHN"]
    
    hyperparams = {
        "Cars": [20, 5,0],
        "DTD": [20, 10, 0.7],
        "EuroSAT": [3, 5, 0.5],
        "GTSRB": [5, 5,0.8],
        "MNIST": [7, 3, 0.8],
        "RESISC45": [10, 2, 0.6],
        "SUN397": [10, 3, 0],
        "SVHN": [2, 5, 0.8],
    }

    args = parse_arguments()
    args.batch_size = 64 if args.model == "ViT-L-14" else 128
    args.num_grad_accumulation = 2 if args.model == "ViT-L-14" else 1
    args.print_every = 10
    if args.seed is not None:
        args.save = f"checkpoints_{args.seed}/{args.model}"
    else:
        args.save = f"checkpoints/{args.model}"
    with open(os.path.join(args.save, "zeroshot_accuracies.json"), 'r') as f:
        args.zs_acc = json.load(f)
    with open(os.path.join(args.save, "ft_accuracies.json"), 'r') as f:
        args.ft_acc = json.load(f)

    for dataset in all_datasets:
        args.tgt_dataset = dataset + "Val"
        args.ctr_datasets = [d + "Val" for d in all_datasets if d != dataset]
        args.epoch, args.lr_multiplier, _ = hyperparams[dataset]
        args.control_thresholds = {
            d + "Val": hyperparams[d][2]
            for d in all_datasets if d != dataset
        }

        print("=" * 100)
        print(f"Learn task vector coefficients of {args.model} for negating {dataset}")
        print(f"Control datasets: {', '.join([d.replace('Val', '') for d in args.ctr_datasets])}")
        print(f"Control thresholds: {args.control_thresholds}")
        print("=" * 100)
        torch.multiprocessing.spawn(main, args=(args,), nprocs=args.world_size)
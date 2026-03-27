# # # # # import pandas as pd

# # # # # # Read both parquet files
# # # # # test_df = pd.read_parquet('/data8/rajiv/mariamma/sidharth/atlas/atlas/data/imagenet/test/test-00000-of-00001.parquet')
# # # # # val_df = pd.read_parquet('/data8/rajiv/mariamma/sidharth/atlas/atlas/data/imagenet/val/val-00000-of-00001.parquet')

# # # # # # Concatenate them
# # # # # merged_df = pd.concat([test_df, val_df], ignore_index=True)

# # # # # # Save to a single parquet file
# # # # # merged_df.to_parquet('/data8/rajiv/mariamma/sidharth/atlas/atlas/data/imagenet/val/val_merged.parquet', index=False)

# # # # # print(f"Test samples: {len(test_df)}")
# # # # # print(f"Val samples: {len(val_df)}")
# # # # # print(f"Merged samples: {len(merged_df)}")
# # # # from datasets import Dataset

# # # # ds = Dataset.from_parquet("/data8/rajiv/mariamma/sidharth/atlas/atlas/data/imagenet/val/val_merged.parquet")
# # # # print(ds)
# # # # print(ds.features)
# # # # print(ds[0])

# # # from datasets import load_dataset
# # # import os

# # # # Set cache directory (optional - to control where it downloads)
# # # cache_dir = "/data8/rajiv/mariamma/sidharth/atlas/atlas/data/imagenet_hf"

# # # # Download only validation split
# # # ds = load_dataset(
# # #     "ILSVRC/imagenet-1k",
# # #     split="train",
# # #     streaming=False,  # Set True if you want to iterate without full download
# # #     trust_remote_code=True
# # # )
# # # print(f"Dataset size: {len(ds)}")
# # # print(f"Features: {ds.features}")
# # # print(f"First sample: {ds[0]}")

# # from datasets import load_dataset

# # ds = load_dataset(
# #     "ILSVRC/imagenet-1k",
# #     split="validation",
# #     trust_remote_code=True,
# #     verification_mode="no_checks"  # Skip integrity checks that might trigger train download
# # )

# # # Verify
# # print(f"Number of samples: {len(ds)}")  # Should be exactly 50,000
# # print(f"Features: {ds.features}")
# # print(f"First sample label: {ds[0]['label']}")

# # # Save locally as parquet (optional)
# # ds.save_to_disk("/data8/rajiv/mariamma/sidharth/atlas/atlas/data/imagenet_val_50k")

# from huggingface_hub import list_repo_files, hf_hub_download
# import os

# # Step 1: Find validation file names
# print("Finding validation files...")
# all_files = list_repo_files("ILSVRC/imagenet-1k", repo_type="dataset")
# val_files = [f for f in all_files if "val" in f.lower() and f.endswith(".parquet")]

# print(f"Found {len(val_files)} validation files:")
# for f in val_files:
#     print(f"  {f}")


from huggingface_hub import hf_hub_download
import os

save_dir = "/data8/rajiv/mariamma/sidharth/atlas/atlas/data/imagenet_val"
os.makedirs(save_dir, exist_ok=True)

# Download all 14 validation parquet files
for i in range(14):
    filename = f"data/validation-{i:05d}-of-00014.parquet"
    print(f"Downloading {filename}...")
    hf_hub_download(
        repo_id="ILSVRC/imagenet-1k",
        filename=filename,
        repo_type="dataset",
        local_dir=save_dir
    )

print(f"\nDone! Files saved to {save_dir}/data/")
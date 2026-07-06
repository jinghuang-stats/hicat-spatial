import os
import torch
from torchvision import transforms
import numpy as np
from PIL import Image, ImageOps
Image.MAX_IMAGE_PIXELS = None
from tqdm import tqdm
import argparse
from torch.utils.data import Dataset, DataLoader
from typing import Tuple
import scanpy as sc 
from time import time
import psutil
import platform
from datetime import datetime
from multiprocessing import cpu_count
from pathlib import Path

#python ExtractFeatures/UNI_16_16_h5ad.py --read_path ./data/ --save_dir ./result/ --sample H1_low --device cuda:1

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--read_path', type=str, required=True)
    parser.add_argument('--sample',type=str,required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda:2')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--stride', type=int, default=112)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--checkpoint_path', type=str, default=None)
    return parser.parse_args()

def log_system_info(tag="START"):
    print(f"\n[INFO-{tag}] Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[INFO-{tag}] CPU usage: {psutil.cpu_percent()}%")
    print(f"[INFO-{tag}] Memory usage: {psutil.virtual_memory().percent}%")
    print(f"[INFO-{tag}] Platform: {platform.platform()}")
    print(f"[INFO-{tag}] Process ID: {os.getpid()}")

def monitor_cuda_memory():
    def decorator(func):
        def wrapper(*args, **kwargs):
            # get args.device automatically
            if args:
                args_obj = args[0]
                device = getattr(args_obj, "device", "cuda:0")
            else:
                device = kwargs.get("device", "cuda:0")
            
            print(f"[Debug] Using device: {device}")
            if torch.cuda.is_available():
                torch.cuda.set_device(device)
                _ = torch.tensor(0., device=device)  # activate CUDA allocator
                torch.cuda.reset_peak_memory_stats()
            else:
                print("⚠️ CUDA is not available，all steps will be done on CPU")

            start = time()
            result = func(*args, **kwargs)
            end = time()

            if torch.cuda.is_available():
                peak = torch.cuda.max_memory_allocated(device=device) / 1024 ** 3
                print(f"\n[CUDA:{device}] Maximum CUDA Memory Usage: {peak:.4f} GB")
            print(f"[TIME] main() cost: {end - start:.2f} s")

            return result
        return wrapper
    return decorator

def create_model(ckpt_path: str) -> torch.nn.Module:
    """Load pretrained weights."""
    try:
        import timm
    except ImportError as exc:
        raise ImportError(
            "UNI feature extraction requires timm. Install HiCAT with the "
            "image optional dependencies."
        ) from exc

    ckpt_path = Path(ckpt_path).expanduser().resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"UNI checkpoint file not found: {ckpt_path}. "
            "Download UNI weights and pass checkpoint_path to uni_extract_features()."
        )

    """Create and load the ViT model."""
    model = timm.create_model(
        "vit_large_patch16_224",
        img_size=224,
        patch_size=16,
        init_values=1e-5,
        num_classes=0,
        global_pool='',
    )
    state_dict = torch.load(str(ckpt_path), map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    return model


def _resolve_uni_checkpoint_path(checkpoint_path=None) -> str:
    """Resolve the UNI checkpoint file path."""
    if checkpoint_path is None:
        checkpoint_path = (
            Path(__file__).resolve().parent
            / "checkpoints"
            / "vit_large_patch16_224.dinov2.uni_mass100k"
            / "pytorch_model.bin"
        )

    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if checkpoint_path.is_dir():
        checkpoint_path = checkpoint_path / "pytorch_model.bin"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"UNI checkpoint file not found: {checkpoint_path}. Expected a file like "
            "pytorch_model.bin. Pass checkpoint_path='/path/to/pytorch_model.bin'."
        )
    return str(checkpoint_path)


# --- Feature Extraction ---
@torch.inference_mode()
def extract_features(model: torch.nn.Module, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    #final_output, _ = model.forward_intermediates(batch, return_prefix_tokens=False)
    #final_output = model(batch)
    with torch.cuda.amp.autocast():
        final_output = model(batch)
    local_emb = final_output[:, 1:]  # shape [B, 196, D]
    global_emb = final_output[:, 0]  # shape [B, D]
    return global_emb, local_emb


class SlidingWindowDataset(Dataset):
    def __init__(self, image: Image.Image, patch_size: int = 224, stride: int = 112):
        self.image = ImageOps.expand(image, border=patch_size // 2, fill=0)
        self.patch_size = patch_size
        self.stride = stride
        self.transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        self.coords = []

        W, H = self.image.size
        for top in range(0, H - patch_size + 1, stride):
            for left in range(0, W - patch_size + 1, stride):
                self.coords.append((top, left))

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        top, left = self.coords[idx]
        patch = self.image.crop((left, top, left + self.patch_size, top + self.patch_size))
        patch_tensor = self.transform(patch)
        return patch_tensor, top, left

def get_center_weights(size=14, sigma=0.5):
    """Create a 2D Gaussian-like center weight matrix of shape [size, size]."""
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    xx, yy = np.meshgrid(x, y)
    weights = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    weights /= weights.max()  # Normalize to [0,1]
    return weights

def extract_dense_feature_map(
    model: torch.nn.Module,
    image: Image.Image,
    device: torch.device,
    batch_size: int = 64,
    patch_size: int = 224,
    stride: int = 112,
    token_size: int = 16,
    num_workers: int = 4
) -> np.ndarray:
    model.eval()
    dataset = SlidingWindowDataset(image, patch_size=patch_size, stride=stride)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    print("Total number of 224*224*3 image tiles:", len(dataset))

    W_pad, H_pad = dataset.image.size
    print(f"Padded image shape:({H_pad},{W_pad})")
    H_tokens = H_pad // token_size
    W_tokens = W_pad // token_size
    print(f"Padded token size:({H_tokens},{W_tokens})")
    
    D = 1024
    D_fused = D * 2

    feature_map = torch.zeros((H_tokens, W_tokens, D_fused), dtype=torch.float16, device="cpu")
    count_map = torch.zeros((H_tokens, W_tokens), dtype=torch.float16, device="cpu")

    # Precompute center weights (shared by all patches)
    center_weight = torch.tensor(get_center_weights(size=14), dtype=torch.float32, device="cpu")  # [14,14]

    for batch, tops, lefts in tqdm(loader, desc="Sliding Window Inference"):
        batch = batch.to(device, non_blocking=True)
        cls_tokens, local_tokens = extract_features(model, batch)  # [B, D], [B, 196, D]
        fused = torch.cat([
            cls_tokens[:, None, None, :].expand(-1, 14, 14, -1),
            local_tokens.view(-1, 14, 14, D)
        ], dim=-1)  # shape: [B, 14, 14, D_fused]

        for i in range(batch.shape[0]):
            top_idx = tops[i].item() // token_size
            left_idx = lefts[i].item() // token_size

            # Center-weighted fusion
            weighted = fused[i].cpu() * center_weight[..., None]  # [14, 14, D_fused]
            feature_map[top_idx:top_idx + 14, left_idx:left_idx + 14] += weighted
            count_map[top_idx:top_idx + 14, left_idx:left_idx + 14] += center_weight

    # Normalize final feature map
    feature_map /= count_map.unsqueeze(-1).clamp(min=1e-6)

    # Crop to original image region (remove padding)
    H_ori, W_ori = image.size[1], image.size[0]
    H_out = H_ori // token_size
    W_out = W_ori // token_size
    start_y = (patch_size // 2) // token_size
    start_x = (patch_size // 2) // token_size

    final_map = feature_map[start_y:start_y + H_out, start_x:start_x + W_out]
    return final_map.detach().cpu().numpy()  # [H_out, W_out, D_fused]


@torch.inference_mode()
@monitor_cuda_memory()
def main(args):
    ############## print remote server configuration ################
    ############## print remote server configuration ################
    ############## print remote server configuration ################
    print("✅ CUDA is available：", torch.cuda.is_available())
    print("✅ Count of GPUs：",torch.cuda.device_count())
    # print("✅ Current device：", torch.cuda.current_device())
    # print("✅ Name of device：", torch.cuda.get_device_name(torch.cuda.current_device()))
    print("✅ Count of cpus:",cpu_count())
    
    ############# for A100: run fast 
    ############# for A100: run fast 
    ############# for A100: run fast 
    torch.set_float32_matmul_precision('high')
    
    ############# Create and setup model##########################
    ############# Create and setup model##########################
    ############# Create and setup model##########################
    t0 = time()
    local_dir = (
    "/UNI_ExtractFeatures_V5/UNI-model/"
    "vit_large_patch16_224.dinov2.uni_mass100k/"
    "pytorch_model.bin"
    )
    device = torch.device(args.device)
    model = create_model(local_dir)
    model = model.to(device)
    #model = torch.compile(model)
    model.eval()
    print("name of model：", next(model.parameters()).device)
    t1=time()
    print(f"Create and setup model cost {int(t1-t0)}s!!!")
    
    ############# read image file, mask file(generated from HistoSweep) and create save dir #######################
    ############# read image file, mask file(generated from HistoSweep) and create save dir #######################
    ############# read image file, mask file(generated from HistoSweep) and create save dir #######################
    
    t0 = time()
    #args.image_path = args.read_path + f"{args.sample}/Image/he_raw_high.jpg"
    #args.image_path = args.read_path + f"{args.sample}/Image/he_processed.jpg"
    #args.image_path = args.read_path + f"{args.sample}/Image/he_processed.tif"
    args.image_path = args.read_path + f"{args.sample}/Image/he_processed.jpg"
    if(not os.path.exists(f"{args.image_path}")):
        print("image file don't exist")
        exit(1)
        
    args.mask_path = args.read_path + f"{args.sample}/Image/mask-small.png"
    if(not os.path.exists(f"{args.mask_path}")):
        print("mask file don't exist")
        exit(1)
    
    args.save_dir =  args.save_dir + f"{args.sample}/"
    if(not os.path.exists(f"{args.save_dir}")):
        os.makedirs(f"{args.save_dir}")
          
    img = Image.open(args.image_path).convert("RGB")
    print(f"raw image.shape = {np.array(img).shape}")
    mask = np.array(Image.open(args.mask_path)) > 0  # Convert to binary mask
    print(f"grid mask.shape ={mask.shape}")
    t1=time()
    print(f"read image file, mask file(generated from HistoSweep) and create save dir cost {int(t1-t0)}s!!!")
    
    
    ################ extract 16*16 super pixel embeddings ############################
    ################ extract 16*16 super pixel embeddings ############################
    ################ extract 16*16 super pixel embeddings ############################
    t0 = time()
    final_map = extract_dense_feature_map(
        model=model,
        image=img,
        device=args.device,
        batch_size=args.batch_size,
        patch_size=224,
        stride=args.stride,
        token_size=16,
        num_workers=args.num_workers
    )
    print(f"final_map.shape={final_map.shape}")  
    t1 =time()
    print(f'extract 16_16_features cost {int(t1-t0)}s!!!')
     
    ################# filter 16*16 super pixel embedding base on mask file ###############
    ################# filter 16*16 super pixel embedding base on mask file ###############
    ################# filter 16*16 super pixel embedding base on mask file ###############
    t0 = time()
    token_grid_coords = []
    token_features = []
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j]:
                token_grid_coords.append((i, j))
                token_features.append(final_map[i, j])
    t1 = time()
    print(f'filter 16*16 super pixel embedding costs {int(t1-t0)}s!!!')
    
    ################ save 16*16 super pixel embeddings to h5ad file ######################
    ################ save 16*16 super pixel embeddings to h5ad file ######################
    ################ save 16*16 super pixel embeddings to h5ad file ######################
    t0 = time()
    print(t0)
    adata = sc.AnnData(X=np.array(token_features))
    token_grid_coords = np.array(token_grid_coords)
    adata.obsm["spatial"] = token_grid_coords[:, [1, 0]]
    adata.obs["sample"] = args.sample # need to set for Viulization
    embeddings = adata.X
    
    # # perform PCA dimension reduction 
    # start_pca = time()
    # ncluster_list = [5,10,15,20,25,30]
    # pca = PCA(n_components=50)
    # PCA_emb = pca.fit_transform(embeddings)
    # total_variance = pca.explained_variance_ratio_.sum()
    # print(f"total_variance_ratio: {total_variance:.4f}")
    # end_pca = time()
    # print(f"perform PCA dimension reduction costs {int(end_pca - start_pca)} s")
    # adata.obsm["X_pca"] = PCA_emb
    
    # # perform kmeans clusering
    # start_kmeans = time()
    # for cluster in ncluster_list:
    #     print(f"clustering with Kmeans(K={cluster})")
    #     kmeans = KMeans(n_clusters=cluster, random_state=42)
    #     cluster_labels = kmeans.fit_predict(PCA_emb)
    #     adata.obs["kmeans_{}".format(cluster)] = cluster_labels.astype(str)
    # end_kmeans = time()
    # print(f"perform Kmeans costs {int(end_kmeans - start_kmeans)} s")
    print(adata)
    adata.write(f'{args.save_dir}uni_super_emb.h5ad',compression ="gzip")
    
    print("done ...........................")


# added
def uni_extract_features(
    sample,
    device="cuda",
    checkpoint_path=None,
    batch_size: int = 128,
    stride: int = 112,
    num_workers: int = 8,
):
    """
    Extract UNI superpixel embeddings from a preprocessed histology image.

    Expected input
    --------------
    sample/
    ├── he_processed.jpg
    └── mask-small.png

    Output
    ------
    sample/
    └── uni_super_emb.h5ad

    Parameters
    ----------
    sample : str
        Sample folder containing the preprocessed image and tissue mask.


    device : str, default="cuda"
        Device used for inference. Use "cuda" or "cpu".

    checkpoint_path : str or pathlib.Path, optional
        UNI checkpoint file. If omitted, use the packaged default checkpoint
        location resolved by :func:`_resolve_uni_checkpoint_path`.

    batch_size : int, default=128
        Batch size for sliding-window inference.

    stride : int, default=112
        Sliding-window stride in pixels.

    num_workers : int, default=8
        Number of workers used by the DataLoader.
    """

    image_file = f"{sample}/he_processed.jpg"
    mask_file = f"{sample}/mask-small.png"

    if not os.path.exists(image_file):
        raise FileNotFoundError(image_file)

    if not os.path.exists(mask_file):
        raise FileNotFoundError(mask_file)

    model_path = _resolve_uni_checkpoint_path(checkpoint_path)

    device = torch.device(device)

    print("Loading UNI model...")
    model = create_model(model_path)
    model = model.to(device)
    model.eval()

    img = Image.open(image_file).convert("RGB")
    mask = np.array(Image.open(mask_file)) > 0

    final_map = extract_dense_feature_map(
        model=model,
        image=img,
        device=device,
        batch_size=batch_size,
        patch_size=224,
        stride=stride,
        token_size=16,
        num_workers=num_workers,
    )

    token_grid_coords = []
    token_features = []

    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j]:
                token_grid_coords.append((i, j))
                token_features.append(final_map[i, j])

    adata = sc.AnnData(np.asarray(token_features))

    token_grid_coords = np.asarray(token_grid_coords)
    adata.obsm["spatial"] = token_grid_coords[:, [1, 0]]
    adata.obs["sample"] = sample

    out_file = f"{sample}/uni_super_emb.h5ad"
    adata.write_h5ad(out_file, compression="gzip")

    print("----------Finished extracting UNI embeddings----------")


if __name__ == '__main__':
    args = get_args()
    ############# argparse parameters################################
    ############# argparse parameters################################
    ############# argparse parameters################################
    print(args)    
    
    start_time = time()
    log_system_info("START")
    t0 = time()
    main(args)
    t1 = time()
    print(f"All steps cost {t1 - t0}s!!!!")
    log_system_info("END")
    print(f"[INFO-END] Total runtime: {time() - start_time:.2f} seconds")

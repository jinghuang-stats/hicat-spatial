import shutil
import argparse
import os
from pathlib import Path
from time import time

from .histosweep_vendor.HistoSweep.computeMetrics import compute_metrics
from .histosweep_vendor.HistoSweep.densityFiltering import (
    compute_low_density_mask,
)
from .histosweep_vendor.HistoSweep.generateMask import generate_final_mask
from .histosweep_vendor.HistoSweep.ratioFiltering import run_ratio_filtering
from .histosweep_vendor.HistoSweep.textureAnalysis import run_texture_analysis
from .histosweep_vendor.HistoSweep.utils import load_image

# python Run.py --read_dir HE/demo/ --save_dir ./mask/

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--read_dir', type=str, default='AAAA',
                       help='dictionary to read dataset')
    parser.add_argument('--save_dir', type=str, default='BBBB',
                       help='Directory to save results')
    parser.add_argument('--pixel_size_raw',type=float,default = 0.5)
    parser.add_argument('--density_thresh',type=int,default = 100)
    parser.add_argument('--clean_background_flag', action='store_true', help='Wheter to preserve fibrous regions that are otherwise being incorrectly filtered out')
    parser.add_argument('--min_size',type=int,default = 10)
    parser.add_argument('--patch_size',type=int,default = 16)
    parser.add_argument('--pixel_size',type=float,default = 0.5)
    
    ##########################
    return parser.parse_args()
    
def main():
    args = get_args()
    print(args)
    
    
    # Flag for whether to rescale the image 
    need_scaling_flag = False  # True if image resolution ≠ 0.5µm (or desired size) per pixel
    # Flag for whether to preprocess the image 
    need_preprocessing_flag = False  # True if image dimensions are not divisible by patch_size
    HE_prefix = args.read_dir
    directory = args.save_dir
    pixel_size_raw = args.pixel_size_raw
    density_thresh = args.density_thresh
    clean_background_flag = args.clean_background_flag
    min_size = args.min_size
    patch_size = args.patch_size
    pixel_size = args.pixel_size
    
    
    # Read dataset
    if not os.path.exists(args.read_dir):
        raise ValueError(f"Path file {args.path_file} does not exist!")
    
    # Create save directory
    #os.makedirs(args.save_dir, exist_ok=True)
    

    ########################main code###################################
    # rescale and preprocess image
    image = load_image(os.path.join(HE_prefix, "he_processed.jpg"))
    print(image.shape)
    
#     if not os.path.exists(directory):
#         os.makedirs(directory)

#########################################################################################
#     saveParams(HE_prefix, need_scaling_flag, need_preprocessing_flag, pixel_size_raw,density_thresh,clean_background_flag,min_size,patch_size,pixel_size)
#########################################################################################

    he_std_norm_image_, he_std_image_, z_v_norm_image_, z_v_image_, ratio_norm_, ratio_norm_image_ = compute_metrics(image, patch_size=patch_size)
    
    # identify low density superpixels
    mask1_lowdensity = compute_low_density_mask(z_v_image_, he_std_image_, ratio_norm_, density_thresh=density_thresh)
    
    print('Total selected for density filtering: ', mask1_lowdensity.sum())
    
    # perform texture analysis 
    mask1_lowdensity_update = run_texture_analysis(prefix=HE_prefix, image=image, tissue_mask=mask1_lowdensity, patch_size=patch_size, glcm_levels=64)
    
    # identify low ratio superpixels
    mask2_lowratio, otsu_thresh = run_ratio_filtering(ratio_norm_, mask1_lowdensity_update)
    print(mask2_lowratio.shape)
    
    
    if not os.path.exists(os.path.join(f"{HE_prefix}/{directory}")):
        os.makedirs(os.path.join(f"{HE_prefix}/{directory}"))
    generate_final_mask(prefix=HE_prefix, he=image,output_dir=directory, 
                    mask1_updated = mask1_lowdensity_update, mask2 = mask2_lowratio, 
                    clean_background = clean_background_flag, 
                    super_pixel_size=patch_size, minSize = min_size)

    ###########################################################
    
    print("Running successfully!")
    
    print("copy mask-small.png to its parent folder...")
    file_path = os.path.join(f"{HE_prefix}/{directory}", 'mask-small.png')
    dest_path = os.path.join(f"{HE_prefix}/", 'mask-small.png')
    shutil.copy2(file_path, dest_path) 
    print(f"File copied to {dest_path}")
    print("Copying successfully!!!!")



# from Xueqi
def uni_generate_mask(
    sample,
    save_dir="mask",
    pixel_size_raw: float = 0.5,
    density_thresh: int = 100,
    clean_background_flag: bool = False,
    min_size: int = 10,
    patch_size: int = 16,
    pixel_size: float = 0.5
):
    """
    Generate tissue mask for UNI feature extraction using HistoSweep.

    Expected input
    --------------
    sample/
    └── he_processed.jpg

    Output
    ------
    sample/
    ├── mask-small.png

    sample/mask/
    ├── mask-small.png
    └── other HistoSweep outputs

    Parameters
    ----------
    sample : str
        Sample folder containing ``he_processed.jpg``.

    save_dir : str or pathlib.Path, default="mask"
        Relative subdirectory inside the sample folder, or an absolute output
        directory, where HistoSweep masks are saved.

    pixel_size_raw : float, default=0.5
        Raw image pixel size. Kept for compatibility with the original script.

    density_thresh : int, default=100
        Threshold used for low-density superpixel filtering.

    clean_background_flag : bool, default=False
        Whether to preserve fibrous regions that may otherwise be incorrectly
        filtered out as background.

    min_size : int, default=10
        Minimum object size used in final mask generation.

    patch_size : int, default=16
        Superpixel or patch size used for HistoSweep masking.

    pixel_size : float, default=0.5
        Target pixel size. Kept for compatibility with the original script.

    """

    image = load_image(f"{sample}/he_processed.jpg")

    he_std_norm_image_, he_std_image_, z_v_norm_image_, z_v_image_, ratio_norm_, ratio_norm_image_ = compute_metrics(
        image,
        patch_size=patch_size
    )

    mask1_lowdensity = compute_low_density_mask(
        z_v_image_,
        he_std_image_,
        ratio_norm_,
        density_thresh=density_thresh
    )

    # print("Total selected for density filtering:", mask1_lowdensity.sum())

    mask1_lowdensity_update = run_texture_analysis(
        prefix=sample,
        image=image,
        tissue_mask=mask1_lowdensity,
        patch_size=patch_size,
        glcm_levels=64
    )

    mask2_lowratio, otsu_thresh = run_ratio_filtering(
        ratio_norm_,
        mask1_lowdensity_update
    )

    sample_path = Path(sample)
    mask_output_dir = Path(save_dir).expanduser()
    if not mask_output_dir.is_absolute():
        mask_output_dir = sample_path / mask_output_dir
    mask_output_dir.mkdir(parents=True, exist_ok=True)

    generate_final_mask(
        prefix=sample,
        he=image,
        output_dir=mask_output_dir,
        mask1_updated=mask1_lowdensity_update,
        mask2=mask2_lowratio,
        clean_background=clean_background_flag,
        super_pixel_size=patch_size,
        minSize=min_size
    )

    src_mask = mask_output_dir / "mask-small.png"
    dst_mask = sample_path / "mask-small.png"

    if not os.path.exists(src_mask):
        raise FileNotFoundError(f"HistoSweep mask was not generated: {src_mask}")

    shutil.copy2(src_mask, dst_mask)

    print("----------Finished generating the mask----------")



if __name__ == '__main__':
    t0 = time()
    main()
    t1 = time()
    print(f"Running this file cost {t1-t0} s!!!")

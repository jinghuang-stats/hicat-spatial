import argparse
import os 
import numpy as np
from time import time
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from .histosweep_vendor.HistoSweep.image import crop_image
from .histosweep_vendor.HistoSweep.preprocess import adjust_margins
from .histosweep_vendor.HistoSweep.rescale import rescale_image
from .histosweep_vendor.HistoSweep.utils import load_image

def load_image2(filename, verbose=True):
    import tifffile
    ext = os.path.splitext(filename)[-1].lower()
    # use tifffile to open .tif / .tiff
    if ext in ['.tif', '.tiff']:
        img = tifffile.imread(filename)
        img = np.array(img)
    else:
        img = Image.open(filename)
        img = np.array(img)
    if img.ndim == 3 and img.shape[-1] == 4:
        img = img[..., :3]  # remove alpha channel
    if verbose:
        print(f'Image loaded from {filename}')
    return img


# python processImageData.py --image_path ./data/ --save_dir ./data/  --sample H1

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--sample',type=str,required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--scale_value',type = float, default = 1.0)
    parser.add_argument('--pad_value',type = int,default = 16)
    
    args = parser.parse_args()
    return args

def get_image_filename(prefix):
    print(prefix)
    file_exists = False
    for suffix in ['.jpg', '.png', '.tiff','.tif']:
        filename = prefix + suffix
        if os.path.exists(filename):
            file_exists = True
            break
    if not file_exists:
        raise FileNotFoundError('Image not found')
    return filename

def main():
    args = get_args()
    print(args)
    #pdb.set_trace()
    args.save_dir =  args.save_dir + f"{args.sample}/"
    if(not os.path.exists(f"{args.save_dir}")):
        os.makedirs(f"{args.save_dir}")
    print(args.save_dir)
        
    args.image_path =get_image_filename(args.image_path + f"{args.sample}/Image/he-raw")
    print(args.image_path)
    if(not os.path.exists(f"{args.image_path}")):
        print("image file don't exist")
        exit(1)
    image = load_image(args.image_path)
    print(f"image_raw.shape ={image.shape}")
    img_scale = image.astype(np.float32)
    print(f'Rescaling image (scale: {args.scale_value:.3f})...')
    t0 = time()
    img_scale = rescale_image(img_scale, args.scale_value)
    print(int(time() - t0), 'sec')
    img_scale = img_scale.astype(np.uint8)
    
    print(f"image_scale.shape ={img_scale.shape}")
    img_pad  = adjust_margins(img_scale, pad=args.pad_value, pad_value = None)
    print(f"image_padding.shape ={img_pad.shape}")
    print(f"image_grid.shape ={img_pad.shape[0]/16},{img_pad.shape[1]/16}")
    Image.fromarray(img_pad).save(f"{args.save_dir}/Image/he_processed.jpg")
    


# from Xueqi
def uni_preprocess_image(
    sample,
    scale_value: float = 1.0,
    pad_value: int = 16,
):
    """
    Preprocess raw H&E image for UNI feature extraction.

    Expected input:
        sample/he-raw.jpg, .png, or .tif

    Output:
        sample/he_processed.jpg

    Parameters
    ----------
    sample : str
        Sample name.

    scale_value : float, default=1.0
        Rescaling factor for the raw image.

    pad_value : int, default=16
        Padding unit. Output image height and width will be divisible by this value.

    Returns
    -------
    np.ndarray
        Padded and processed image.
    """

    if sample is None:
        raise ValueError("sample must be provided.")

    if not os.path.exists(sample):
        raise FileNotFoundError('Sample folder not found')

    if not 0 < scale_value <= 1.0:
        raise ValueError("scale value must be between 0 and 1.")

    raw_image_file = get_image_filename(f'{sample}/he-raw')
    image = load_image(raw_image_file)
    # print(f"image_raw.shape = {image.shape}")

    img_scale = image.astype(np.float32)

    # print(f"Rescaling image (scale: {scale_value:.3f})...")
    t0 = time()
    img_scale = rescale_image(img_scale, scale_value)
    # print(int(time() - t0), "sec")

    img_scale = img_scale.astype(np.uint8)
    # print(f"image_scale.shape = {img_scale.shape}")

    img_pad = adjust_margins(
        img_scale,
        pad=pad_value,
        pad_value=None,
    )

    # print(f"image_padding.shape = {img_pad.shape}")
    # print(f"image_grid.shape = {img_pad.shape[0] / 16}, {img_pad.shape[1] / 16}")

    Image.fromarray(img_pad).save(f"{sample}/he_processed.jpg")

    print('----------Finished preprocessing----------')



if __name__ == '__main__':
    t0 = time()
    main()
    t1 = time()
    print(f"running done,cost {t1-t0}s")

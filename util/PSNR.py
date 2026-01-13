from skimage.metrics import peak_signal_noise_ratio as psnr
import os
import cv2
import numpy as np

# from torchvision.transforms.functional import normalize
import torch
def rgb2ycbcr_pt(img, y_only=False):
    """Convert RGB images to YCbCr images (PyTorch version).

    It implements the ITU-R BT.601 conversion for standard-definition television. See more details in
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.

    Args:
        img (Tensor): Images with shape (n, 3, h, w), the range [0, 1], float, RGB format.
         y_only (bool): Whether to only return Y channel. Default: False.

    Returns:
        (Tensor): converted images with the shape (n, 3/1, h, w), the range [0, 1], float.
    """
    if y_only:
        weight = torch.tensor([[65.481], [128.553], [24.966]]).to(img)
        out_img = torch.matmul(img.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + 16.0
    else:
        weight = torch.tensor([[65.481, -37.797, 112.0], [128.553, -74.203, -93.786], [24.966, 112.0, -18.214]]).to(img)
        bias = torch.tensor([16, 128, 128]).view(1, 3, 1, 1).to(img)
        out_img = torch.matmul(img.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + bias

    out_img = out_img / 255.
    return out_img
def calculate_psnr_pt(img, img2, crop_border, test_y_channel=False):
    """Calculate PSNR (Peak Signal-to-Noise Ratio) (PyTorch version).

    Reference: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Args:
        img (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
        img2 (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: PSNR result.
    """

    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')

    if crop_border != 0:
        img = img[:, :, crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[:, :, crop_border:-crop_border, crop_border:-crop_border]

    if test_y_channel:
        img = rgb2ycbcr_pt(img, y_only=True)
        img2 = rgb2ycbcr_pt(img2, y_only=True)

    img = img.to(torch.float64)
    img2 = img2.to(torch.float64)

    mse = torch.mean((img - img2)**2, dim=[1, 2, 3])
    return 10. * torch.log10(1. / (mse + 1e-8))
def calculate_avg_psnr(folder1,folder2):
    
    image_path1= os.listdir(folder1)
    image_path2= os.listdir(folder2)
    # 初始化SSIM总和
    psnr_sum = 0.0
    # 计数器，用于计算平均SSIM
    image_count = len(image_path1)
    for i in range(len(image_path1)):
        image1 = cv2.imread(os.path.join(folder1,image_path1[i]),cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0
        image2 = cv2.imread(os.path.join(folder2,image_path2[i]),cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        # Convert NumPy arrays to PyTorch tensors and move them to the CPU
        image1_tensor = torch.from_numpy(image1).permute(2, 0, 1).unsqueeze(0).to('cpu')
        image2_tensor = torch.from_numpy(image2).permute(2, 0, 1).unsqueeze(0).to('cpu')
        psnr_sum += calculate_psnr_pt(image1_tensor, image2_tensor,0)
    return psnr_sum / image_count

if __name__ == '__main__':
    fake_path = "/home/bj/pix2pix/test_result/dists/fakes"
    real_path = "/home/bj/pix2pix/test_result/dists/reals"
    psnr_value = calculate_avg_psnr(fake_path, real_path)
    print(f'PSNR value: {psnr_value}')
# Example usage:
# psnr_value = calculate_psnr('path_to_image1.jpg', 'path_to_image2.jpg')
# print(f'PSNR value: {psnr_value}')

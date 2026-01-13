import cv2
import os
from skimage.metrics import structural_similarity as ssim
import numpy as np

def _ssim(img, img2):
    """Calculate SSIM (structural similarity) for one channel images.

    It is called by func:`calculate_ssim`.

    Args:
        img (ndarray): Images with range [0, 255] with order 'HWC'.
        img2 (ndarray): Images with range [0, 255] with order 'HWC'.

    Returns:
        float: SSIM result.
    """

    c1 = (0.01 * 255)**2
    c2 = (0.03 * 255)**2
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img, -1, window)[5:-5, 5:-5]  # valid mode for window size 11
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean()


def calculate_average_ssim(folder1, folder2):
    # 获取两个文件夹下的所有图片文件名
    images1 = [f for f in os.listdir(folder1) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    images2 = [f for f in os.listdir(folder2) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # 确保两个文件夹中的图片数量相同
    if len(images1) != len(images2):
        raise ValueError("两个文件夹中的图片数量不匹配")
    
    # 初始化SSIM总和
    ssim_sum = 0.0
    # 计数器，用于计算平均SSIM
    image_count = 0
    
    # 遍历文件夹中的图片
    for image_name in images1:
        # 读取图像，并转换为归一化浮点型
        _name = image_name.replace(".jpg", ".jpg")
        print(_name)
        img1 = cv2.imread(os.path.join(folder1, _name), cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0
        img2 = cv2.imread(os.path.join(folder2, _name), cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0
        
        # 确保图像被成功加载
        if img1 is None or img2 is None:
            print(f"图片 {image_name} 未找到或无法读取")
            continue
        
        # 计算图像尺寸
        height, width = img1.shape[:2]
        
        # 确保窗口大小是奇数，并且不大于图像的最小边长
        win_size = min(11, width, height)  # 选择11作为默认窗口大小，但不超过图像尺寸
        
        # 计算SSIM
        score, _ = ssim(img1, img2, full=True, win_size=win_size, data_range=1.0, channel_axis=-1, gaussian_weights=True, use_sample_covariance=False)
        ssim_sum += score
        image_count += 1
    
    # 计算平均SSIM值
    if image_count == 0:
        raise ValueError("没有有效的图片进行SSIM计算")
    average_ssim = ssim_sum / image_count
    
    return average_ssim


# 指定两个文件夹路径
folder_path1 = '/home/bj/pix2pix/test_result/dists/fakes'
folder_path2 = '/home/bj/pix2pix/test_result/dists/reals'

# 计算平均SSIM
average_ssim = calculate_average_ssim(folder_path2, folder_path1)

# 打印平均SSIM结果
print(f"pix2pixHD_vit_loss_maps_test的平均SSIM值: {average_ssim:.4f}")
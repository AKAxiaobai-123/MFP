import os
from pathlib import Path
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import torch


class ClipDataset(BaseDataset):
    """A dataset class for paired image dataset with an associated text description for the target.

    It assumes that the directory '/path/to/data/<phase>' contains three subdirectories:
    - train_A (or test_A): images in the input domain
    - train_B (or test_B): images in the target domain
    - text: text descriptions for the target images

    The images in train_A and train_B should have the same file names (e.g., 123.jpg).
    The text files in 'text' should have the same stem as the images (e.g., 123.txt).

    Options supported via `opt`:
    - opt.dataroot, opt.phase, opt.max_dataset_size (as usual)
    - opt.caption_ext (optional): default '.txt'
    """

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        # 假设目录结构为:
        # dataroot/phase/train_A
        # dataroot/phase/train_B
        # dataroot/phase/text
        # 或者直接是 dataroot/train_A, dataroot/train_B (取决于你的 dataroot 设置)
        
        # 这里我们假设 opt.dataroot 指向 dataset/train 或 dataset/val
        # 并且下面有 train_A, train_B, text 文件夹
        
        # 为了兼容性，我们尝试构建路径
        # 如果 opt.phase 是 'train'，我们期望子目录也是 'train_A', 'train_B'
        # 但有时候用户可能把 phase 设为 'train' 但文件夹叫 'A', 'B'，这里我们按用户描述的 train_A/train_B/text
        
        # 修正：通常 pix2pix 的 aligned dataset 是直接在 phase 目录下找图片
        # 这里改为分别找 A, B, text
        
        # 构造 A, B, text 的路径
        # 假设 opt.dataroot = .../dataset/train
        # 那么 dir_A = .../dataset/train/train_A
        
        # 注意：pix2pix 标准做法是 opt.dataroot 指向 dataset 根目录，opt.phase 指向 train/test
        # 但根据你之前的 shuffle.py，你的结构是 dataset/train/train_A
        # 所以如果 opt.dataroot=dataset, opt.phase=train
        # 路径应该是 dataset/train/train_A
        
        self.dir_A = os.path.join(opt.dataroot, opt.phase, 'train_A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase, 'train_B')
        self.dir_text = os.path.join(opt.dataroot, opt.phase, 'train_text')
        
        # 如果找不到 train_A，尝试找 A (兼容性)
        if not os.path.exists(self.dir_A):
             self.dir_A = os.path.join(opt.dataroot, opt.phase, 'A')
             self.dir_B = os.path.join(opt.dataroot, opt.phase, 'B')

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))
        
        # 确保 A 和 B 数量一致（或者至少 B 包含 A 需要的）
        # 这里简单起见，假设一一对应且排序后对齐
        
        assert(opt.load_size >= opt.crop_size)
        self.input_nc = opt.output_nc if opt.direction == 'BtoA' else opt.input_nc
        self.output_nc = opt.input_nc if opt.direction == 'BtoA' else opt.output_nc

        self.caption_ext = getattr(opt, 'caption_ext', '.txt')
        self.step = torch.zeros(1) - 1

    def __getitem__(self, index):
        self.step += 1

        # A path
        A_path = self.A_paths[index % len(self.A_paths)]
        
        # B path - 假设文件名相同
        # 如果 B 的数量和 A 不一样，或者排序不对，这里可能会错位
        # 更稳健的做法是根据 A 的文件名去 B 目录找
        stem = Path(A_path).stem
        suffix = Path(A_path).suffix
        
        # 尝试在 B 目录找同名文件 (假设扩展名也相同，或者尝试常见扩展名)
        B_path = os.path.join(self.dir_B, Path(A_path).name)
        if not os.path.exists(B_path):
            # 尝试找 jpg 如果 A 是 png
            if suffix == '.png':
                B_path = os.path.join(self.dir_B, stem + '.jpg')
            elif suffix == '.jpg':
                B_path = os.path.join(self.dir_B, stem + '.png')
        
        # 如果还是找不到，回退到按索引取（不推荐，但作为兜底）
        if not os.path.exists(B_path):
             B_path = self.B_paths[index % len(self.B_paths)]

        A = Image.open(A_path).convert('RGB')
        B = Image.open(B_path).convert('RGB')

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        A = A_transform(A)
        B = B_transform(B)

        # load caption text
        caption = ''
        caption_path = os.path.join(self.dir_text, stem + self.caption_ext)
        
        if os.path.exists(caption_path):
            try:
                with open(caption_path, 'r', encoding='utf-8') as f:
                    caption = f.read().strip()
            except Exception:
                caption = ''

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path, 'text': caption, 'step': self.step}

    def __len__(self):
        return len(self.A_paths)

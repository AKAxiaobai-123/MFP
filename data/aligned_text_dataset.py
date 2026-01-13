import os
from pathlib import Path
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import torch


class AlignedTextDataset(BaseDataset):
    """A dataset class for paired image dataset with an associated text description for the target.

    It assumes that the directory '/path/to/data/<phase>' contains paired images in the form {A,B}
    (left-right concatenated). For each AB image, this class will try to locate a text file
    with the same stem and extension (default '.txt') containing the target text description.

    Options supported via `opt`:
    - opt.dataroot, opt.phase, opt.max_dataset_size (as usual)
    - opt.caption_folder (optional): if provided, captions are searched under
        os.path.join(opt.dataroot, opt.caption_folder, <stem> + caption_ext)
      otherwise captions are searched next to the AB image (same directory) with the same stem.
    - opt.caption_ext (optional): default '.txt'
    """

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))
        assert(opt.load_size >= opt.crop_size)
        self.input_nc = opt.output_nc if opt.direction == 'BtoA' else opt.input_nc
        self.output_nc = opt.input_nc if opt.direction == 'BtoA' else opt.output_nc

        # caption configuration
        self.caption_folder = getattr(opt, 'caption_folder', None)
        self.caption_ext = getattr(opt, 'caption_ext', '.txt')

        # step counter similar to AlignedDataset
        self.step = torch.zeros(1) - 1

    def __getitem__(self, index):
        self.step += 1

        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')

        # split AB image into A and B
        w, h = AB.size
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h))
        B = AB.crop((w2, 0, w, h))

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        A = A_transform(A)
        B = B_transform(B)

        # load caption text if available
        stem = Path(AB_path).stem
        caption = ''

        if self.caption_folder:
            caption_path = os.path.join(self.opt.dataroot, self.caption_folder, stem + self.caption_ext)
            if os.path.exists(caption_path):
                try:
                    with open(caption_path, 'r', encoding='utf-8') as f:
                        caption = f.read().strip()
                except Exception:
                    caption = ''
        else:
            caption_path = str(Path(AB_path).with_suffix(self.caption_ext))
            if os.path.exists(caption_path):
                try:
                    with open(caption_path, 'r', encoding='utf-8') as f:
                        caption = f.read().strip()
                except Exception:
                    caption = ''

        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path, 'text': caption, 'step': self.step}

    def __len__(self):
        return len(self.AB_paths)

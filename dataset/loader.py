import cv2
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path
from json import load
from typing import Union
from dataset.util import Mode, calculate_aperture_embedding, generate_maps, build_input_dict, crop_to_divisible
import torch
import random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from pathlib import Path
from json import load
from typing import Union, Tuple
from enum import Enum


class RealBokeh(Dataset):
    def __init__(self, data_path: Union[str, Path], mode: Mode,
                 binary_bokeh: bool = False,
                 defocus_deblur_mode: bool = False,
                 device: str = 'cpu',
                 challenge: bool = False,
                 patch_size: int = 512,
                 use_full_resolution: bool = False  # 新增參數
                 ):
        self._data_path = Path(data_path)
        self._mode = mode
        self.device = device
        self.patch_size = patch_size
        self.challenge = challenge
        self.defocus_deblur_mode = defocus_deblur_mode
        self.use_full_resolution = use_full_resolution  # 新增
        self._iteration_mode = 'sample'
        
        # 讀取路徑與資料邏輯 (保持不變)
        self._mode_dir = self._data_path.joinpath(self._mode.value)
        self._scene_list = sorted([load(open(f)) for f in self._mode_dir.joinpath("metadata").glob("*.json")], key=lambda x: x['id'])
        
        self._sample_list = []
        for scene_id, metadata in enumerate(self._scene_list):
            for sample_id in range(len(metadata['target_images'])):
                self._sample_list.append((scene_id, sample_id))

    def __len__(self):
        return len(self._sample_list) if self._iteration_mode == 'sample' else len(self._scene_list)

    def __getitem__(self, index: int):
        # 1. 取得中繼資料
        metadata = self._scene_list[self._sample_list[index][0]]
        tgt_av = metadata['target_avs'][self._sample_list[index][1]]
        target_name = Path(metadata['target_images'][self._sample_list[index][1]]).stem

        # 2. 讀取原始大圖 (24-MP)
        source = Image.open(self._mode_dir.joinpath(metadata['source_image']))
        target = Image.open(self._mode_dir.joinpath(metadata['target_images'][self._sample_list[index][1]])) if not self.challenge else None

        # 3. 執行裁切 - 新增完整解析度選項
        if self.use_full_resolution:
            # 使用完整解析度，不裁切
            pass
        elif self._mode == Mode.TRAIN:
            # 訓練時：隨機裁切
            w, h = source.size
            i = random.randint(0, h - self.patch_size)
            j = random.randint(0, w - self.patch_size)
            
            source = TF.crop(source, i, j, self.patch_size, self.patch_size)
            if target is not None:
                target = TF.crop(target, i, j, self.patch_size, self.patch_size)
        elif self._mode == Mode.VAL:
            # 驗證時：中心裁切
            source = TF.center_crop(source, [self.patch_size, self.patch_size])
            if target is not None:
                target = TF.center_crop(target, [self.patch_size, self.patch_size])

        if self.defocus_deblur_mode:
            source, target = target, source

        # 4. 生成輔助圖層與 Embedding
        aperture_embedding = calculate_aperture_embedding(tgt_av)
        maps = generate_maps(source, aperture_embedding, target=target)

        # 5. 打包字典
        return_dict = build_input_dict(maps, aperture_embedding, target_name, self.device)

        return return_dict

class RealBokeh_depth(Dataset):
    def __init__(self, data_path, mode, binary_bokeh=False,
                 defocus_deblur_mode=False, device='cpu',
                 challenge=False, patch_size=512,
                 force_full_resolution=False):
        self._data_path = Path(data_path)
        self._mode = mode  # 這個 mode 是從外部傳入的 Mode.VAL (value='validation')
        self.device = device
        self.patch_size = patch_size
        self.challenge = challenge
        self.defocus_deblur_mode = defocus_deblur_mode
        self.force_full_resolution = force_full_resolution
        self._iteration_mode = 'sample'
        
        # 安全檢查
        if self._mode == Mode.TRAIN and self.force_full_resolution:
            raise ValueError(
                "❌ 訓練模式不能使用完整解析度！請設定 force_full_resolution=False"
            )
        
        # 讀取資料
        self._mode_dir = self._data_path.joinpath(self._mode.value)
        self._scene_list = sorted(
            [load(open(f)) for f in self._mode_dir.joinpath("metadata").glob("*.json")], 
            key=lambda x: x['id']
        )
        
        self._sample_list = []
        for scene_id, metadata in enumerate(self._scene_list):
            for sample_id in range(len(metadata['target_images'])):
                self._sample_list.append((scene_id, sample_id))

        # 顯示資訊
        resolution_info = "完整解析度" if self.force_full_resolution else f"{patch_size}×{patch_size}"
        print(f"RealBokeh 初始化成功 | 模式: {mode.value} | 樣本數: {len(self._sample_list)} | 解析度: {resolution_info}")
        print("Random rotation and flipping augmentation are applied!!!")
    def __len__(self):
        return len(self._sample_list) if self._iteration_mode == 'sample' else len(self._scene_list)

    def __getitem__(self, index: int):
        # 1. 取得中繼資料
        metadata = self._scene_list[self._sample_list[index][0]]
        tgt_av = metadata['target_avs'][self._sample_list[index][1]]
        target_name = Path(metadata['target_images'][self._sample_list[index][1]]).stem

        # 2. 讀取原始圖片
        source = Image.open(self._mode_dir.joinpath(metadata['source_image']))
        target = Image.open(
            self._mode_dir.joinpath(metadata['target_images'][self._sample_list[index][1]])
        ) if not self.challenge else None

        # ============================================================
        # 3. 裁切邏輯 - 添加更多 debug
        # ============================================================
        if index == 0:
            print(f"\n🔍 Crop Debug:")
            print(f"  self._mode = {self._mode} (value={self._mode.value})")
            print(f"  Mode.VAL = {Mode.VAL} (value={Mode.VAL.value})")
            print(f"  force_full_resolution = {self.force_full_resolution}")
            print(f"  Original image size: {source.size}")
        
        if self.force_full_resolution:
            # 使用完整解析度：不裁切
            if index == 0:
                print(f"  ✓ Using full resolution (no crop)")
            
        elif self._mode == Mode.TRAIN:
            # 訓練模式：隨機裁切
            if index == 0:
                print(f"  ✓ Mode is TRAIN, applying random crop + augmentation")
            
            w, h = source.size
            if h < self.patch_size or w < self.patch_size:
                raise ValueError(f"Image {source.size} < patch_size {self.patch_size}")
            
            i = random.randint(0, h - self.patch_size)
            j = random.randint(0, w - self.patch_size)
            
            source = TF.crop(source, i, j, self.patch_size, self.patch_size)
            if target is not None:
                target = TF.crop(target, i, j, self.patch_size, self.patch_size)
            
            # ============================================================
            # 新增：Random rotation and flipping for training
            # ============================================================
            # Random horizontal flip
            if random.random() > 0.5:
                source = TF.hflip(source)
                if target is not None:
                    target = TF.hflip(target)
            
            # Random vertical flip
            if random.random() > 0.5:
                source = TF.vflip(source)
                if target is not None:
                    target = TF.vflip(target)
            
            # Random rotation (0, 90, 180, 270 degrees)
            angle = random.choice([0, 90, 180, 270])
            if angle != 0:
                source = TF.rotate(source, angle)
                if target is not None:
                    target = TF.rotate(target, angle)
        
        elif self._mode == Mode.VAL:
            # 驗證模式：中心裁切
            if index == 0:
                print(f"  ✓ Mode is VAL, applying center crop")
            
            source = TF.center_crop(source, [self.patch_size, self.patch_size])
            if target is not None:
                target = TF.center_crop(target, [self.patch_size, self.patch_size])
        
        else:
            if index == 0:
                print(f"  ⚠️ No crop condition matched! Mode = {self._mode}")
        
        if index == 0:
            print(f"  Final image size: {source.size}\n")

        # 4. Defocus deblur 模式
        if self.defocus_deblur_mode:
            source, target = target, source

        # 5. 生成輔助圖層
        aperture_embedding = calculate_aperture_embedding(tgt_av)
        maps = generate_maps(source, aperture_embedding, target=target)

        # 6. 打包字典
        return_dict = build_input_dict(maps, aperture_embedding, target_name, self.device)

        return return_dict

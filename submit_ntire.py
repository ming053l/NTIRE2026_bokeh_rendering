import os
from os import makedirs
from statistics import mean
from torch import load, no_grad, clamp, Tensor
from torch.cuda import Event, synchronize, get_device_name
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
from pathlib import Path
from shutil import make_archive, rmtree
from datetime import datetime

# Local Imports
from dataset.loader import RealBokeh
from dataset.util import Mode
from util.parser import get_ntire_parser
from method.config import bokehlicious_size_builder
from method.model import Bokehlicious

def preprocess_batch(batch):
    for k, v in batch.items():
        if isinstance(v, Tensor):
            batch[k] = v.unsqueeze(0).cuda()
    return batch

if __name__ == "__main__":
    parser = get_ntire_parser()
    # 添加 size 參數
    parser.add_argument('-size', type=str, default='large', 
                       choices=['small', 'medium', 'large'],
                       help='Model size (default: small)')
    # 添加完整解析度選項
    parser.add_argument('--full_resolution', action='store_true',
                       help='Use full resolution instead of center crop (works with validation data)')
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    dataset_path = args.dataset_root_dir / 'Bokeh_NTIRE2026'
    
    # 檢查資料集路徑
    print(f"Dataset path: {dataset_path}")
    print(f"Dataset exists: {dataset_path.exists()}")
    
    if not dataset_path.exists():
        print(f"\n❌ Error: Dataset path does not exist!")
        exit(1)
    
    # 根據 full_resolution 參數決定使用的模式
    if args.full_resolution:
        # 使用 validation 資料但不裁切（模擬 TEST 行為）
        mode = Mode.VAL
        use_full_res = True
        print(f"Using FULL RESOLUTION mode with validation data")
    else:
        # 正常模式
        mode = Mode.VAL if args.phase == 'dev' else Mode.TEST
        use_full_res = False
        print(f"Using {'512x512 center crop' if mode == Mode.VAL else 'full resolution'}")
    
    mode_dir = dataset_path / mode.value
    print(f"Mode directory: {mode_dir}")
    print(f"Mode directory exists: {mode_dir.exists()}")
    
    if not mode_dir.exists():
        print(f"\n❌ Error: Mode directory '{mode.value}' does not exist!")
        print(f"Available directories in {dataset_path}:")
        for item in sorted(dataset_path.iterdir()):
            print(f"  - {item.name}")
        exit(1)
    
    # 建立輸出目錄
    output_directory = Path(args.out_path) / args.name
    if output_directory.exists():
        rmtree(output_directory)
    makedirs(output_directory, exist_ok=True)

    # 初始化模型
    config = bokehlicious_size_builder(args.size)
    model = Bokehlicious(**config)
    
    ckpt = load(checkpoint)
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict)
    

    model.to(args.device)
    model.eval()

    parameters = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"\nLoaded {args.size} model from {checkpoint.name}")
    print(f"Parameters: {parameters:.2f}M")

    # 初始化 Dataloader - 嘗試傳入 use_full_resolution 參數
    print(f"\nInitializing dataloader...")
    try:
        # 嘗試使用 use_full_resolution 參數
        dataloader = RealBokeh(
            data_path=dataset_path, 
            mode=mode,
            device=args.device, 
            challenge=True,
            use_full_resolution=use_full_res
        )
    except TypeError:
        # 如果 RealBokeh 不支援 use_full_resolution 參數
        if use_full_res:
            print("Warning: RealBokeh doesn't support use_full_resolution parameter")
            print("Please update dataset/loader.py to add this parameter")
            print("See the provided realbokeh_fullres.py for reference")
            exit(1)
        else:
            dataloader = RealBokeh(
                data_path=dataset_path, 
                mode=mode,
                device=args.device, 
                challenge=True
            )

    print(f"Number of images: {len(dataloader)}")
    
    if len(dataloader) == 0:
        print(f"\n❌ Error: No images found in dataloader!")
        exit(1)

    start_events = [Event(enable_timing=True) for _ in range(len(dataloader))]
    end_events = [Event(enable_timing=True) for _ in range(len(dataloader))]

    print(f"\nStarting Inference for {len(dataloader)} images...")
    print(f"Resolution mode: {'Full resolution' if use_full_res else '512x512'}")
    
    with no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            # 取得正確的檔名 Key
            img_name = batch['image_name']
            processed_batch = preprocess_batch(batch)

            start_events[i].record()
            # 執行前向傳播
            output = model(**processed_batch)
            end_events[i].record()

            # 儲存結果
            output_tensor = clamp(output.squeeze(0), 0, 1)
            output_img = to_pil_image(output_tensor.cpu())
            output_img.save(output_directory / f"{img_name}.png")

    synchronize()
    
    # 計算元數據
    if len(start_events) > 0:
        avg_time = mean([s.elapsed_time(e) for s, e in zip(start_events, end_events)]) / 1000.0
    else:
        avg_time = 0.0

    # 寫入 readme.txt
    metadata_file = output_directory / "readme.txt"
    with open(metadata_file, "w") as f:
        f.write(f"Architecture Name: {args.name}\n")
        f.write(f"Parameters: {parameters:.2f}M\n")
        f.write(f"Device: {get_device_name()}\n")
        f.write(f"Runtime: {avg_time:.3f}s\n")
        f.write(f"Extra Data: {'Yes' if args.extra_data else 'No'}\n")

    # 建立壓縮檔
    archive_name = f"SUBMISSION_{args.name}_{'fullres' if use_full_res else args.phase}"
    make_archive(archive_name, 'zip', root_dir=output_directory)
    
    print(f"\n{'='*60}")
    print(f"✓ Submission file created: {archive_name}.zip")
    print(f"✓ Model: {args.size} ({parameters:.2f}M parameters)")
    print(f"✓ Average Runtime: {avg_time:.3f}s per image")
    print(f"✓ Total Images: {len(dataloader)}")
    print(f"✓ Resolution: {'Full' if use_full_res else '512x512'}")
    print(f"✓ Output directory: {output_directory}")
    print(f"{'='*60}")
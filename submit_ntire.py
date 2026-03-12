# Package Imports
from os import makedirs
from statistics import mean
from torch import load, no_grad, clamp, Tensor
from torch.cuda import Event, synchronize, get_device_name
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
from pathlib import Path
from shutil import make_archive, unpack_archive, copytree, rmtree
from datetime import datetime
from warnings import warn

# Local Imports
from dataset.loader import RealBokeh
from dataset.util import Mode
from util.parser import get_ntire_parser

# TODO Architecture imports, replace with your model!
from method.config import bokehlicious_size_builder
from method.model import Bokehlicious

"""
!!!!! NTIRE CHALLENGE README: !!!!!
This script produces a submission ready .zip archive to be uploaded at 
https://www.codabench.org/competitions/12764/#/participate-tab for evaluation by our server.
You will have to replace our Baseline network with your own solution, makes sure to check the relevant #TODO comments!
The image input archives are expected to be at the -dataset_root_dir (default is './dataset').

Run `python submit_ntire.py -h` to see an overview of the script arguments.
Run `python submit_ntire.py -c small.pt -n 'BokN_S_(Baseline)'` to generate a sample submission based on the challenge baseline method.
"""

def unsqueeeze_batch(batch):
    for k, v in batch.items():
        if isinstance(v, Tensor):
            batch[k] = v.unsqueeze(0).cuda()
    return batch

if __name__ == "__main__":
    parser = get_ntire_parser()
    args = parser.parse_args()

    # put the path to your checkpoint file here, or use the parser argument -checkpoint / -c
    checkpoint = Path(f"./checkpoints/{args.checkpoint}")

    # Setup and sanity checking
    assert args.phase in ['dev', 'test'], (f"Unknown argument for phase (-phase, -p): {args.phase}, "
                                           f"only ['dev', 'test'] are supported for this script.")
    assert checkpoint.is_file(), f"Checkpoint {checkpoint} is not a file."
    if args.image_format != 'png':
        warn(f"Image format {args.image_format} might lead to a lower score du to default compression behaviour, "
             f"we recomment .png for final submission!")
    dataset_path = args.dataset_root_dir / 'Bokeh_NTIRE2026'
    assert args.dataset_root_dir.is_dir(), f"Path {args.dataset_root_dir} is not a directory."
    if args.phase == 'dev':
        if (dataset_path / 'validation').exists():
            print(f"Found NTIRE 2026 Bokeh Challenge Development inputs in: {dataset_path.absolute()}")
        else: # development set is missing
            print(f"Could not locate NTIRE 2026 Bokeh Challenge Development inputs (\'validation\' folder) in "
                  f"{dataset_path.absolute()}, attempting to extract them from \'Bokeh_NTIRE2026_Development_Inputs.zip\'")
            assert (args.dataset_root_dir / 'Bokeh_NTIRE2026_Development_Inputs.zip').is_file(), \
                (f'{f"Could not find the \'validation\' split (which is used in the dev phase) at {dataset_path.absolute()}"
                if dataset_path.is_dir() else
                f"Could not find the \'Bokeh_NTIRE26\' folder at {args.dataset_root_dir.absolute()}"} '
                 f'OR the development inputs archive \'Bokeh_NTIRE2026_Development_Inputs.zip\'.')
            unpack_archive(args.dataset_root_dir / 'Bokeh_NTIRE2026_Development_Inputs.zip', args.dataset_root_dir)
            copytree(args.dataset_root_dir / 'Bokeh_NTIRE2026_Development_Inputs',
                     args.dataset_root_dir, dirs_exist_ok=True)
            rmtree(args.dataset_root_dir / 'Bokeh_NTIRE2026_Development_Inputs')
            print("Successfully finished development dataset setup!")
    else: # args.phase = 'test'
        if (dataset_path / 'test').exists():
            print(f"Found NTIRE 2026 Bokeh Challenge Test inputs in: {dataset_path.absolute()}")
        else:  # Test set is missing
            print(f"Could not locate NTIRE 2026 Bokeh Challenge Test inputs (\'test\' folder) in "
                  f"{dataset_path.absolute()}, attempting to extract them from \'Bokeh_NTIRE2026_Test_Inputs.zip\'")
            assert (args.dataset_root_dir / 'Bokeh_NTIRE2026_Test_Inputs.zip').is_file(), \
                (f'{f"Could not find the \'test\' split (which is used in the final test phase) at {dataset_path.absolute()}" 
                    if dataset_path.is_dir() else 
                    f"Could not find the \'Bokeh_NTIRE26\' folder at {args.dataset_root_dir.absolute()}"} '
                 f'OR the test inputs archive \'Bokeh_NTIRE2026_Test_Inputs.zip\'.')
            unpack_archive(args.dataset_root_dir / 'Bokeh_NTIRE2026_Test_Inputs.zip', args.dataset_root_dir)
            copytree(args.dataset_root_dir / 'Bokeh_NTIRE2026_Test_Inputs ', args.dataset_root_dir,
                     dirs_exist_ok=True)
            rmtree(args.dataset_root_dir / 'Bokeh_NTIRE2026_Test_Inputs ')
            print("Successfully finished test dataset setup!")
    # Setup dirs for saving results
    output_directory = args.out_path / args.name / 'NTIRE2026BokehChallenge' / args.phase
    # Clean output directory
    try:
        rmtree(output_directory)
    except FileNotFoundError:
        pass
    makedirs(output_directory, exist_ok=False)
    print(f"Saving outputs to {output_directory.absolute()}")

    # Network Name is used in the Codalab leaderboard, set as desired with the -name argument
    print(f"Running Architecture {args.name} on {'Development' if args.phase == 'dev' else 'Test'} set...")

    # We initialize our own method here, replace this with the appropriate code to initialize your network!
    # TODO: Load your own model here for evaluation!
    config = bokehlicious_size_builder('small')

    model = Bokehlicious(**config)

    print(f"Initialized model on {args.device}")

    # This code should load your checkpoint and set up the model for evaluation
    state_dict = load(checkpoint)
    model.load_state_dict(state_dict)
    model.to(args.device)
    model.eval()
    print(f"Loaded weights from {checkpoint.absolute()}")

    # Initialize evaluation dataset, use '-phase test' argument for the final test phase
    dataloader = RealBokeh(data_path=dataset_path, mode=Mode.VAL if args.phase=='dev' else Mode.TEST, device=args.device, challenge=True)
    print(f"Initialized RealBokeh (NTIRE 2026 challenge) {'Development' if args.phase == 'dev' else 'Test'} phase dataloader")

    # We use cuda events for timing network inference times
    start_events = [Event(enable_timing=True) for _ in range(len(dataloader))]
    end_events = [Event(enable_timing=True) for _ in range(len(dataloader))]

    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        with no_grad():
            # TODO: Depending on whatever inputs your model uses you might need to modify the batch pre-processing
            batch = unsqueeeze_batch(batch) # (c h w) to (b c h w) for all tensors in batch dict

            synchronize() # wait for GPU to complete any current workload
            start_events[i].record() # log prediction start time
            ### ONLY THE MODEL FORWARD CALL SHOULD BE BETWEEN start_events[i].record()
            output = model(**batch) # unpack batch dict for network forward call
            ### AND nd_events[i].record()
            end_events[i].record() # log prediction end time

            output = clamp(output, 0, 1)
            to_pil_image(output.squeeze(0).cpu()).save(output_directory / f"{batch['image_name']}.{args.image_format}")

    print("Finished prediction!")

    # record important metadata
    avg_time = mean([s.elapsed_time(e) for s, e in zip(start_events, end_events)]) # avg is in ms
    print(f"Average time taken for {args.phase} phase: {avg_time / 1000:.3f} seconds on {get_device_name()}.")
    parameters = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {parameters / 1e6:.2f}M")

    metadata_file = output_directory / "readme.txt"
    with open(metadata_file, "w") as readme:
        readme.write(f"This file contains relevant metadata for the challenge leaderboard, "
                     f"everything should be automatically generated by the submit.py script, "
                     f"but feel free to double check!\n")
        readme.write(f"Architecture Name:{args.name}\n")
        readme.write(f"Parameters:{parameters / 1e6:.2f}M\n")
        readme.write(f"Runtime:{avg_time / 1000:.3f}s\n")
        readme.write(f"Device:{get_device_name()}\n")
        readme.write(f"Extra data:{'Yes' if args.extra_data else 'No'}\n")

    print(f"Wrote metadata to {metadata_file.absolute()}:")
    print("")
    with open(metadata_file, "r") as readme:
        print(readme.read())

    print(f"Creating zip archive for Codabench submission...")
    archive_file = args.out_path / f'{args.name}_{args.phase}_{datetime.now().strftime("%Y-%m-%d_%H:%M")}'
    make_archive(archive_file, 'zip', root_dir=output_directory)

    print(f"Please upload your submission file found at {archive_file.absolute()}.zip to https://www.codabench.org/competitions/12764/#/participate-tab!")











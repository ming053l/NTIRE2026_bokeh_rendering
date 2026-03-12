# Bokehlicious: <br/> Photorealistic Bokeh Rendering with Controllable Apertures

This is a slightly reduced version (no large checkpoints) of our official [Bokehlicious GitHub repo](https://github.com/TimSeizinger/Bokehlicious).

### [Project Page](https://timseizinger.github.io/BokehliciousProjectPage/)

### NTIRE Challenge Instructions:

In collaboration with [NTIRE (New Trends in Image Restoration and Enhancement) Workshop @ CVPR 2026](https://cvlai.net/ntire/2026/) we are hosting a Challenge on Controllable Aperture Bokeh Rendering! The goal is to beat our Baseline method from this repository, with the top teams invited to present their solution at NTIRE @ CVPR 2026

**Sign up here: https://www.codabench.org/competitions/12764/**

The Baseline solution of the challenge is included in this starting kit, you can either try to improve our architecture, or propose an entirely new and novel solution! :D

As RealBokeh is too large to be hosted directly on CodaBench you will need to download the training data from [Huggingface](https://huggingface.co/datasets/timseizinger/RealBokeh_3MP/tree/main/train)!
All scripts expect the RealBokeh dataset at *./dataset/*.
E.g. the training data location would be *./dataset/RealBokeh/train*

This repo includes a script (```submit_ntire.py```) for easy submission of your results to the [NTIRE 2026 Challenge on Controllable Bokeh Rendering](https://www.codabench.org/competitions/12764/)!
NTIRE submission instructions can be found in ```submit_ntire.py```.

BTW this starting kit already includes the Development phase input, no need to manually download!

## Installation

```
pip install -r requirements.txt
```

## Usage

predict.py lets you run Bokehlicious (or your own method after modifying the code) on a single image.
For example:
```
python predict.py -img_path ./examples/collie.jpg -size small -av 2.8
```

Here _-img\_path_ is the path to the image you want to render, _-size_ is the size of the model you want to use (small or large) and _-av_ is the aperture f-stop to control the strength of bokeh (between 2.0 and 20.0).

## Evaluation

We recommend using the *RealBokeh* validation and test sets for evaluating your method privately during the challenge, but optionally *EBB! Val294* and *EBB400* are also supported by ```evaluate.py```.

Before running the evaluation script you need to download the [validation set](https://huggingface.co/datasets/timseizinger/RealBokeh_3MP/tree/main/validation) or [test set](https://huggingface.co/datasets/timseizinger/RealBokeh_3MP/tree/main/test) of RealBokeh and copy it to the ./dataset/RealBokeh folder.
The same applies to EBB! Val294 (./dataset/EBB_Val294) and EBB400 (./dataset/EBB400).

To run the evaluation script use:
```
python evaluate.py -dataset RealBokeh -size small --save_outputs
```

Here _-dataset_ is the dataset you want to evaluate on (RealBokeh, RealBokeh_bin, EBB_Val294, EBB400), _-size_ is the size of the model you want to use (small or large) and _--save_outputs_ is a flag to save the rendered images.

## RealBokeh Dataset

You can find our RealBokeh Dataset on Huggingface!

[![Dataset on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-lg.svg)](https://huggingface.co/datasets/timseizinger/RealBokeh_3MP)

## Citation

If you find our work useful for your research work please cite:

```
@inproceedings{seizinger2025bokehlicious,
  author    = {Seizinger, Tim and Vasluianu, Florin-Alexandru and Conde, Marcos and Wu, Zongwei and Timofte, Radu},
  title     = {Bokehlicious: Photorealistic Bokeh Rendering with Controllable Apertures},
  booktitle = {ICCV},
  year      = {2025},
}
```

# CRAFT: Video Diffusion for Bimanual Robot Data Generation

Jason Chen, I-Chun Arthur Liu, Gaurav Sukhatme, Daniel Seita

University of Southern California

[Project page](https://craftaug.github.io/) | [arXiv](https://arxiv.org/abs/2604.03552) | [Paper (PDF)](https://arxiv.org/pdf/2604.03552)

## Overview

This is the code release for CRAFT, accepted at IROS. Given a canny edge video from a robot scene, a reference image and a text prompt, it generates a new video of the same motion in a different scene. The demo here uses the stack two bowls task.

It is built on VideoX-Fun, and we changed it to run on Wan2.2-Fun-A14B-Control instead of Wan2.1. Wan2.2 is a much better model, and the generations are noticeably cleaner. We made the switch because people have been emailing us asking for Wan2.2.

The method is not tied to Wan2.2. Many newer video generation models that support canny edge conditioning should work too, so feel free to try others.

The examples below cover the default generation, a different background (curtains in the reference image), green lighting, pink bowls and a multiview observation. All of them are in `run.py`.

## Setup

This repo needs the VideoX-Fun code next to it, since `run.py` imports `videox_fun` from the parent directory. Clone this repo inside a VideoX-Fun checkout:

```bash
git clone https://github.com/aigc-apps/VideoX-Fun.git
cd VideoX-Fun
git clone https://github.com/craftaug/craft.git
cd craft
```

All the commands below run from the `craft` folder.

Make a conda environment first. We ran this on Python 3.12 with PyTorch 2.10 (CUDA 12.8) on one RTX 5090, but any recent Python 3.10+ and a CUDA build of PyTorch that fits your GPU should work. You also need `ffmpeg` on your path.

```bash
conda create -n craft python=3.12 -y
conda activate craft
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # pick the CUDA version for your GPU
```

The model is [Wan2.2-Fun-A14B-Control](https://huggingface.co/alibaba-pai/Wan2.2-Fun-A14B-Control) from the [VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun) project, which builds on [Wan2.2](https://github.com/Wan-Video/Wan2.2). Install the code and download the checkpoint into `models/Diffusion_Transformer/`:

```bash
pip install -r ../requirements.txt
pip install -U "huggingface_hub[cli]"
huggingface-cli download alibaba-pai/Wan2.2-Fun-A14B-Control \
  --local-dir ../models/Diffusion_Transformer/Wan2.2-Fun-A14B-Control
```

The download is large (two 14B transformers plus the text encoder and VAE), so check your disk space first. If you put the checkpoint somewhere else, pass `--model_dir /path/to/it` to `run.py`.

Videos are saved to `outputs/`, which already has our example results.

## Options

| Flag | Default | What it does |
|---|---|---|
| `--ref` | `all` | Which setup to run: `default`, `curtains`, `green`, `pink`, `multiview`, or `all`. You can list several, like `--ref pink curtains`. |
| `--model_dir` | `../models/Diffusion_Transformer/Wan2.2-Fun-A14B-Control` | Where the Wan2.2 checkpoint is. |
| `--out` | `outputs/` | Folder the videos are saved to. |
| `--steps` | `100` | Number of denoising steps. Fewer is faster and rougher. |
| `--seed` | per setup | Overrides the seed for every setup. Without it each setup uses its own seed from `SEEDS` in `run.py` (43 if not listed). |
| `--guidance` | per setup | Overrides the guidance scale. Without it, each setup uses `GUIDANCE` in `run.py` (6.0 if not listed). Lower values follow the reference image more, higher values follow the text more. |
| `--sampler` | per setup | `Flow`, `Flow_Unipc` or `Flow_DPM++`. Without it, each setup uses `SAMPLERS` in `run.py` (`Flow` if not listed). The sampler changes the overall exposure and look more than the other settings. |
| `--shift` | `5.0` | Noise schedule shift. Only used by the `Flow` sampler. |
| `--size` | `512` | Output height and width in pixels. Use `1024` for multiview so each tile is 512x512. It is much slower and needs more memory. |
| `--no_teacache` | off | Turns off TeaCache, which skips some steps to save time. Slower, and it did not change our results. |
| `--fast` | off | Uses model CPU offload instead of sequential offload. Faster but needs more GPU memory. |

The video length (81 frames at 16 fps) is fixed in `run.py`. We run multiview as `--size 1024 --steps 50 --guidance 5`, as shown below.

## Default generation

Canny edge video (`assets/control_canny.mp4`), the reference image (`assets/ref_default.png`) and the prompt in `run.py`.

```bash
python run.py --ref default
```

## Different background

Same canny video and prompt, but the reference image has different curtains.

```bash
python run.py --ref curtains
```

## Green lighting

The reference image is lit green.

```bash
python run.py --ref green
```

## Pink bowls

The reference is an empty scene with no objects, and the prompt asks for pink bowls.

```bash
python run.py --ref pink
```

## Multiview observation

A 2x2 layout with a third-person view and two wrist cameras. It has its own canny video and prompt.

```bash
python run.py --ref multiview --size 1024 --steps 50 --guidance 5
```

The multiview tiles are small, so we generate this one at 1024x1024 so each tile is 512x512. It is slow (about 80 minutes for 50 steps on a 5090 with the default offloading). Dropping to the default 512 works too, but the wide view comes out soft.

Running `python run.py` with no arguments generates all of them.

## Making your own canny video

Use this with any simulator. Render a video, then:

```bash
python make_canny.py my_video.mp4 assets/control_canny.mp4
```

It outputs 512x512, 81 frames at 16 fps. `--low` and `--high` set the Canny thresholds (80 and 200 by default).

## Notes

**Doing this zero shot takes a lot of prompt fine-tuning.** This is a limitation of our method, and people often underestimate it. Some tips:

1. Put the camera close to the robot scene. Far away, the canny edges get noisy. Adjust the Canny thresholds for the camera distance and how many objects are in view.
2. Expect to spend real time on the prompt. If you get artifacts, keep editing it until they go away, and use the negative prompt too. The prompts in `run.py` worked for our setup after a lot of tuning. Use them as a starting point, but they probably won't work as is on a new scene. Shorter isn't always worse: a very long prompt for the multiview layout gave us pure noise, and a short one worked.
3. You may also need to change the hyperparameters (guidance scale, number of steps, seed, TeaCache threshold) depending on the task.
4. Ideally take the reference image with the RealSense camera in the lab. Pick a frame where the robot arms are touching the objects, and make sure the whole arm is in the image. Seeing the full arm tells the model what the arm's texture and material look like, and the gripper contact with the object helps reduce artifacts there.

Prompting won't always get you a good enough video, because current video generation models have limits. Transparent objects are one example: if your robot scene has them, expect trouble, since these models struggle to render them. If you really want to try, you can collect play data of your scene and fine-tune the video generation model on it, but only for very few scenarios.

## Simulator data generation

As mentioned in the paper, the DexMimicGen code isn't fully released, so we implemented it as best we could. Once its data generation code is out, we'll integrate it into this codebase. Until then, you choose the simulator and render the videos yourself, then turn them into canny edge videos with `make_canny.py`.

## Questions

Contact jchen567@usc.edu with questions. He may be able to help, and he'd be happy to collaborate if you want to extend this line of work.

## Citation

```bibtex
@inproceedings{chen2026craft,
  title={{CRAFT: Video Diffusion for Bimanual Robot Data Generation}},
  author={Jason Chen and I-Chun Arthur Liu and Gaurav Sukhatme and Daniel Seita},
  booktitle={International Conference on Intelligent Robots and Systems (IROS)},
  Year={2026}
}
```

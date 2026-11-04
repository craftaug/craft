"""Wan2.2-Fun-A14B-Control: canny video + reference image -> stack-two-bowls video."""
import argparse
import os
import sys

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from videox_fun.models import (AutoencoderKLWan, AutoTokenizer, WanT5EncoderModel,
                               Wan2_2Transformer3DModel)
from videox_fun.models.cache_utils import get_teacache_coefficients
from videox_fun.pipeline import Wan2_2FunControlPipeline
from videox_fun.utils.fm_solvers import FlowDPMSolverMultistepScheduler
from videox_fun.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from videox_fun.utils.fp8_optimization import replace_parameters_by_name
from videox_fun.utils.utils import (filter_kwargs, get_image_latent, get_image_to_video_latent,
                                    get_video_to_video_latent, save_videos_grid)

PROMPT = "A top-down overhead view of two white articulated robotic arms with white segmented links and gray joint accents, mounted on a table covered with black fabric, in front of dark curtains, smoothly and precisely picking up a solid matte gray ceramic bowl from the black fabric surface and stacking it on top of a second identical solid matte gray ceramic bowl, both bowls uniformly gray inside and outside. The robotic arms are neutral pure white from the very first frame to the last, in natural neutral colors under neutral white balance. The gripper hands are plain solid pure white blocks with a smooth flat front panel. The curtains in the background hang perfectly still and static. They move with fluid, deliberate mechanical motion, gripper claws grasping the bowl securely. The workspace is lit with soft indoor lighting, showing the dark fabric texture of the table. The second bowl remains stationary on the table as the first bowl is lowered and nested perfectly on top of it. The motion is slow, controlled, and precise. Shot from directly above, static camera angle, realistic robotics laboratory setting, high detail mechanical components, photorealistic, 4K quality, smooth motion, continuous take."
NEGATIVE = "Blurry, shaky camera, handheld footage, low resolution, pixelated, cartoon, animated, CGI-looking robots, humanoid robot, robotic arms disappearing mid-frame, bowls clipping through table, bowls falling over, incorrect stacking, extra objects on table, people in frame, hands, humans, text overlay, watermark, logo, lens flare, overexposed, underexposed, jerky motion, fast motion, time-lapse, jump cuts, multiple camera angles, side view, distorted geometry, unrealistic physics, floating objects, duplicate bowls, wrong colors, black robot arms, dark robot arms, blue tint, teal tint, cyan tint, colored tint on the robot arms, white-rimmed bowls, black bowl interior, color cast, window or cutout or slot on the gripper hand, gray gripper hand, blue gripper hand, moving curtains, swaying fabric, wind, curtain motion, waving cloth, flowing fabric, rippling background."
MULTIVIEW_PROMPT = "A 2x2 grid of four camera views of two white robot arms stacking one gray bowl on top of another gray bowl on a matte black cloth-covered table. Top-left is a wide view, top-right and bottom-left are close-up wrist camera views, bottom-right is solid black. The top-left wide view is sharp, clean and high detail, with pure white robot arms on a matte black cloth-covered table with no reflections and no glossy shine, in front of dark curtains. Both robot arms are entirely white, including the forearms and wrists. Photorealistic, smooth motion."
CURTAINS_PROMPT = PROMPT.replace("in front of dark curtains", "in front of a brightly lit open robotics lab with shelves, equipment, a blue pallet and windows clearly visible behind the table")
CURTAINS_PROMPT += " There are exactly two gray bowls on the table at the start, and no other objects on the table. The lab behind the table is softly lit with balanced exposure, slightly dimmer and softly blurred, so the bright white robot arms and grippers stand out clearly against it with defined shaded edges."
CURTAINS_NEGATIVE = NEGATIVE + ", extra bowls, four bowls, more than two bowls on the table, clutter on the table, overexposed background, blown out windows, washed out highlights, white robot arms blending into the white background, low contrast"
MULTIVIEW_NEGATIVE = NEGATIVE + ", glossy table, reflections on the table, shiny surface, mirror reflection, polished floor"
PINK_PROMPT = PROMPT.replace("solid matte gray ceramic bowl", "solid matte pink ceramic bowl").replace(", both bowls uniformly gray inside and outside", ", both bowls uniformly pink inside and outside. Only the two bowls are pink, the robot arms and grippers stay white and gray")
PINK_NEGATIVE = NEGATIVE + ", pink robot arms, pink gripper, pink wrist, pink tint on the robot"
SAMPLERS = {"curtains": "Flow_Unipc"}  # per-setup sampler, everything else uses Flow
GUIDANCE = {"curtains": 4.5}  # per-setup guidance scale, everything else uses 6.0 (lower = follows the reference image more)
SEEDS = {"default": 45, "pink": 44, "curtains": 47}  # per-setup seeds, everything else uses 43
# name: (reference image, canny control video, prompt)
REFS = {
    "default":   ("ref_default.png",   "control_canny.mp4",           PROMPT),
    "curtains":  ("ref_curtains.png",  "control_canny.mp4",           CURTAINS_PROMPT),
    "green":     ("ref_green.png",     "control_canny.mp4",           PROMPT),
    "pink":      ("ref_no_objects.png", "control_canny.mp4",           PINK_PROMPT),
    "multiview": ("ref_multiview.png", "control_canny_multiview.mp4", MULTIVIEW_PROMPT),
}

p = argparse.ArgumentParser()
p.add_argument("--ref", nargs="+", default=["all"], choices=["all", *REFS], help="one or more setups")
p.add_argument("--model_dir", default=os.path.join(ROOT, "models/Diffusion_Transformer/Wan2.2-Fun-A14B-Control"))
p.add_argument("--out", default=os.path.join(HERE, "outputs"))
p.add_argument("--steps", type=int, default=100)
p.add_argument("--seed", type=int, default=None, help="override the seed for every setup")
p.add_argument("--size", type=int, default=512, help="output height and width, e.g. 1024 for sharper multiview tiles (much slower)")
p.add_argument("--guidance", type=float, default=None, help="override the guidance scale for every setup")
p.add_argument("--shift", type=float, default=5.0, help="noise schedule shift (5 for 720p-style, lower keeps more fine detail)")
p.add_argument("--sampler", default=None, choices=["Flow", "Flow_Unipc", "Flow_DPM++"], help="override the sampler for every setup")
p.add_argument("--no_teacache", action="store_true", help="turn off TeaCache (slower, avoids caching artifacts)")
p.add_argument("--fast", action="store_true", help="model CPU offload instead of sequential (faster, needs more GPU memory)")
args = p.parse_args()

size, frames, fps, dtype, device = [args.size, args.size], 81, 16, torch.bfloat16, "cuda"
config = OmegaConf.load(os.path.join(ROOT, "config/wan2.2/wan_civitai_i2v.yaml"))
tk = config["transformer_additional_kwargs"]
sub = lambda d, k, default: os.path.join(args.model_dir, d.get(k, default))

transformer = Wan2_2Transformer3DModel.from_pretrained(
    sub(tk, "transformer_low_noise_model_subpath", "transformer"),
    transformer_additional_kwargs=OmegaConf.to_container(tk), low_cpu_mem_usage=True, torch_dtype=dtype)
transformer_2 = Wan2_2Transformer3DModel.from_pretrained(
    sub(tk, "transformer_high_noise_model_subpath", "transformer"),
    transformer_additional_kwargs=OmegaConf.to_container(tk), low_cpu_mem_usage=True, torch_dtype=dtype)
vae = AutoencoderKLWan.from_pretrained(
    sub(config["vae_kwargs"], "vae_subpath", "vae"),
    additional_kwargs=OmegaConf.to_container(config["vae_kwargs"])).to(dtype)
tokenizer = AutoTokenizer.from_pretrained(sub(config["text_encoder_kwargs"], "tokenizer_subpath", "tokenizer"))
text_encoder = WanT5EncoderModel.from_pretrained(
    sub(config["text_encoder_kwargs"], "text_encoder_subpath", "text_encoder"),
    additional_kwargs=OmegaConf.to_container(config["text_encoder_kwargs"]),
    low_cpu_mem_usage=True, torch_dtype=dtype).eval()
def make_scheduler(sampler):
    cls = {"Flow": FlowMatchEulerDiscreteScheduler, "Flow_Unipc": FlowUniPCMultistepScheduler,
           "Flow_DPM++": FlowDPMSolverMultistepScheduler}[sampler]
    kw = OmegaConf.to_container(config["scheduler_kwargs"])
    # Flow reads its shift from the scheduler config; Unipc and DPM++ take it from the pipeline call
    kw["shift"] = args.shift if sampler == "Flow" else 1
    return cls(**filter_kwargs(cls, kw))


scheduler = make_scheduler(args.sampler or "Flow")

pipe = Wan2_2FunControlPipeline(transformer=transformer, transformer_2=transformer_2, vae=vae,
                                tokenizer=tokenizer, text_encoder=text_encoder, scheduler=scheduler)
if not args.fast:
    for t in (transformer, transformer_2):
        replace_parameters_by_name(t, ["modulation"], device=device)
        t.freqs = t.freqs.to(device=device)
    pipe.enable_sequential_cpu_offload(device=device)
else:
    pipe.enable_model_cpu_offload(device=device)

coeffs = None if args.no_teacache else get_teacache_coefficients(args.model_dir)
if coeffs is not None:
    pipe.transformer.enable_teacache(coeffs, args.steps, 0.10, num_skip_start_steps=5, offload=False)
    pipe.transformer_2.share_teacache(transformer=pipe.transformer)

inpaint_video, inpaint_mask, _ = get_image_to_video_latent(None, None, video_length=frames, sample_size=size)

os.makedirs(args.out, exist_ok=True)
for name in (REFS if "all" in args.ref else args.ref):
    img, ctrl, prompt = REFS[name]
    pipe.scheduler = make_scheduler(args.sampler or SAMPLERS.get(name, "Flow"))
    control, _, _, _ = get_video_to_video_latent(
        os.path.join(HERE, "assets", ctrl), video_length=frames, sample_size=size, fps=fps, ref_image=None)
    ref = get_image_latent(os.path.join(HERE, "assets", img), sample_size=size) if img else None
    with torch.no_grad():
        sample = pipe(
            prompt, num_frames=frames, negative_prompt={"pink": PINK_NEGATIVE, "curtains": CURTAINS_NEGATIVE, "multiview": MULTIVIEW_NEGATIVE}.get(name, NEGATIVE), height=size[0], width=size[1],
            generator=torch.Generator(device=device).manual_seed(args.seed if args.seed is not None else SEEDS.get(name, 43)),
            guidance_scale=args.guidance if args.guidance is not None else GUIDANCE.get(name, 6.0), num_inference_steps=args.steps,
            video=inpaint_video, mask_video=inpaint_mask, control_video=control,
            ref_image=ref, boundary=tk.get("boundary", 0.875), shift=args.shift,
        ).videos
    path = os.path.join(args.out, f"{name}.mp4")
    save_videos_grid(sample, path, fps=fps)
    print("saved", path)

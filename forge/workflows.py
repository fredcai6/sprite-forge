"""ComfyUI API graphs, one builder per model. Each returns a prompt ending in a SaveImage."""


def _common_flux2(pos):
    return {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": "flux-2-klein-4b.safetensors", "weight_dtype": "default"}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_3_4b.safetensors", "type": "flux2"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": "flux2-vae.safetensors"}},
        "pos0": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": pos}},
    }


def klein(prompt, refs, seed, width, height, prefix, steps=4):
    g = _common_flux2(prompt)
    cond = "pos0"
    for i, ref in enumerate(refs):
        g[f"img{i}"] = {"class_type": "LoadImage", "inputs": {"image": ref}}
        g[f"enc{i}"] = {"class_type": "VAEEncode", "inputs": {"pixels": [f"img{i}", 0], "vae": ["vae", 0]}}
        g[f"ref{i}"] = {"class_type": "ReferenceLatent", "inputs": {"conditioning": [cond, 0], "latent": [f"enc{i}", 0]}}
        cond = f"ref{i}"
    g |= {
        "neg": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["pos0", 0]}},
        "guider": {"class_type": "CFGGuider", "inputs": {"model": ["unet", 0], "positive": [cond, 0], "negative": ["neg", 0], "cfg": 1.0}},
        "noise": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "sampler": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "sched": {"class_type": "Flux2Scheduler", "inputs": {"steps": steps, "width": width, "height": height}},
        "latent": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "sample": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["noise", 0], "guider": ["guider", 0], "sampler": ["sampler", 0], "sigmas": ["sched", 0], "latent_image": ["latent", 0]}},
        "decode": {"class_type": "VAEDecode", "inputs": {"samples": ["sample", 0], "vae": ["vae", 0]}},
        "save": {"class_type": "SaveImage", "inputs": {"images": ["decode", 0], "filename_prefix": prefix}},
    }
    return g


def qwen_edit(prompt, refs, seed, width, height, prefix, steps=4):
    imgs = {}
    g = {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": "qwen_image_edit_2511_fp8mixed.safetensors", "weight_dtype": "default"}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors", "type": "qwen_image"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
        "lora": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["unet", 0], "lora_name": "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors", "strength_model": 1.0}},
        "shift": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["lora", 0], "shift": 3.1}},
        "norm": {"class_type": "CFGNorm", "inputs": {"model": ["shift", 0], "strength": 1.0}},
    }
    for i, ref in enumerate(refs[:3]):
        g[f"img{i}"] = {"class_type": "LoadImage", "inputs": {"image": ref}}
        imgs[f"image{i + 1}"] = [f"img{i}", 0]
    g |= {
        "pos": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"clip": ["clip", 0], "vae": ["vae", 0], "prompt": prompt, **imgs}},
        "neg": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"clip": ["clip", 0], "vae": ["vae", 0], "prompt": "", **imgs}},
        "posm": {"class_type": "FluxKontextMultiReferenceLatentMethod", "inputs": {"conditioning": ["pos", 0], "reference_latents_method": "index_timestep_zero"}},
        "negm": {"class_type": "FluxKontextMultiReferenceLatentMethod", "inputs": {"conditioning": ["neg", 0], "reference_latents_method": "index_timestep_zero"}},
        "latent": {"class_type": "EmptySD3LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "sample": {"class_type": "KSampler", "inputs": {"model": ["norm", 0], "positive": ["posm", 0], "negative": ["negm", 0], "latent_image": ["latent", 0], "seed": seed, "steps": steps, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "decode": {"class_type": "VAEDecode", "inputs": {"samples": ["sample", 0], "vae": ["vae", 0]}},
        "save": {"class_type": "SaveImage", "inputs": {"images": ["decode", 0], "filename_prefix": prefix}},
    }
    return g


MODELS = {"klein": klein, "qwen-edit": qwen_edit}

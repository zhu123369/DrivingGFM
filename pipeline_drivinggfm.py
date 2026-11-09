# -*- coding: utf-8 -*-
"""
Usage:
    from pipeline_drivinggfm import DrivinggfmPipeline
    import torch
    from PIL import Image

    pipe = DrivinggfmPipeline.from_pretrained("ZehongZhu/drivinggfm", torch_dtype=torch.float16)
    pipe.to("cuda")
    img = pipe.drive(Image.open("scene.png"), "turn the road into a rainy night scene")

Loading notes:
    `model_index.json` declares `_class_name` as `Drivinggfm`, a class provided by this
    file and therefore not registered inside diffusers. Load it through
    `DrivinggfmPipeline.from_pretrained(...)` (not `DiffusionPipeline.from_pretrained`),
    or use `custom_pipeline="pipeline_drivinggfm.py"` together with `trust_remote_code=True`.
"""

from __future__ import annotations

import inspect
from typing import List, Optional, Union

import torch
from PIL import Image
from tqdm.auto import tqdm

from diffusers import DiffusionPipeline
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput
from diffusers.utils import logging

try:  # diffusers >= 0.30
    from diffusers.utils.torch_utils import randn_tensor
except ImportError:  # older layout
    from diffusers.utils import randn_tensor

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

__all__ = ["DrivinggfmPipeline"]

# Default sampling parameters matching the released checkpoint
DEFAULTS = dict(
    num_inference_steps=20,
    image_guidance_scale=1.5,
    guidance_scale=7.5,
)


class DrivinggfmPipeline(DiffusionPipeline):
    """DrivingGFM v1.0 inference pipeline.

    Takes a scene image plus a natural-language instruction and returns the edited
    scene image. Batch and multi-image inputs are both supported.
    """

    model_cpu_offload_seq = "text_encoder->unet->vae"

    def __init__(
        self,
        vae,
        text_encoder,
        tokenizer,
        unet,
        scheduler,
        requires_safety_checker: bool = False,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
        )

        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        )
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.register_to_config(requires_safety_checker=requires_safety_checker)

    # ------------------------------------------------------------------ helpers
    def _encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]],
        num_images_per_prompt: int,
        do_guidance: bool,
        device: torch.device,
    ) -> torch.Tensor:
        """Encode the prompt, plus the empty negative prompt when guidance is active.

        Returns a tripled batch ordered ``[text, negative, negative]``: the first two
        slots feed text guidance, the last slot feeds the image-guidance branch.
        """
        batch_size = 1 if isinstance(prompt, str) else len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        attention_mask = (
            text_inputs.attention_mask.to(device)
            if getattr(self.text_encoder.config, "use_attention_mask", False)
            else None
        )
        prompt_embeds = self.text_encoder(text_inputs.input_ids.to(device), attention_mask=attention_mask)[0]
        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        if not do_guidance:
            return prompt_embeds

        if negative_prompt is None:
            uncond_tokens: List[str] = [""] * batch_size
        elif isinstance(negative_prompt, str):
            uncond_tokens = [negative_prompt]
        else:
            uncond_tokens = negative_prompt

        uncond_input = self.tokenizer(
            uncond_tokens,
            padding="max_length",
            max_length=seq_len,
            truncation=True,
            return_tensors="pt",
        )
        uncond_mask = (
            uncond_input.attention_mask.to(device)
            if getattr(self.text_encoder.config, "use_attention_mask", False)
            else None
        )
        negative_prompt_embeds = self.text_encoder(
            uncond_input.input_ids.to(device), attention_mask=uncond_mask
        )[0]
        negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        neg_seq_len = negative_prompt_embeds.shape[1]
        negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
        negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, neg_seq_len, -1)

        return torch.cat([prompt_embeds, negative_prompt_embeds, negative_prompt_embeds])

    def _prepare_image_latents(self, image: torch.Tensor, do_guidance: bool) -> torch.Tensor:
        """VAE-encode the conditioning image and expand it to the guidance batch layout.

        The three slots are ``[image, image, zeros]``: the middle slot carries the image
        guidance branch, the last slot is the unconditional branch with a zeroed image
        condition.
        """
        image_latents = self.vae.encode(image).latent_dist.mode()

        if do_guidance:
            uncond_image_latents = torch.zeros_like(image_latents)
            image_latents = torch.cat([image_latents, image_latents, uncond_image_latents], dim=0)
        return image_latents

    def _prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        # scale the initial noise by the standard deviation required by the scheduler
        return latents * self.scheduler.init_noise_sigma

    def _extra_step_kwargs(self, generator, eta: float = 0.0) -> dict:
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra = {}
        if accepts_eta:
            extra["eta"] = eta
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra["generator"] = generator
        return extra

    # ------------------------------------------------------------------- forward
    @torch.no_grad()
    def __call__(
        self,
        image: Union[Image.Image, List[Image.Image], torch.Tensor],
        prompt: Union[str, List[str]],
        num_inference_steps: int = DEFAULTS["num_inference_steps"],
        image_guidance_scale: float = DEFAULTS["image_guidance_scale"],
        guidance_scale: float = DEFAULTS["guidance_scale"],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: str = "pil",
        height: Optional[int] = None,
        width: Optional[int] = None,
        show_progress: bool = True,
        **kwargs,
    ) -> StableDiffusionPipelineOutput:
        device = self._execution_device
        do_guidance = guidance_scale > 1.0

        if image is None:
            raise ValueError("`image` input cannot be undefined.")

        batch_size = 1 if isinstance(prompt, str) else len(prompt)

        # 1. text conditioning
        prompt_embeds = self._encode_prompt(prompt, negative_prompt, num_images_per_prompt, do_guidance, device)

        # 2. preprocess the conditioning image
        image = self.image_processor.preprocess(image, height=height, width=width)
        image = image.to(device=device, dtype=prompt_embeds.dtype)

        # 3. timesteps and the image-condition latents
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        image_latents = self._prepare_image_latents(image, do_guidance)

        latent_height = image_latents.shape[-2] * self.vae_scale_factor
        latent_width = image_latents.shape[-1] * self.vae_scale_factor

        # 4. initial noise
        num_channels_latents = self.vae.config.latent_channels
        latents = self._prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            latent_height,
            latent_width,
            prompt_embeds.dtype,
            device,
            generator,
        )

        # 5. channel sanity check
        if num_channels_latents + image_latents.shape[1] != self.unet.config.in_channels:
            raise ValueError(
                f"`unet` expects {self.unet.config.in_channels} input channels but received "
                f"{num_channels_latents} latent channels + {image_latents.shape[1]} image-condition channels."
            )

        extra_step_kwargs = self._extra_step_kwargs(generator, eta)

        # 6. denoising loop: one UNet pass over the tripled batch per step, then
        #    recombine the three predictions with the two guidance scales.
        for t in tqdm(timesteps, desc="Denoising", disable=not show_progress):
            latent_model_input = torch.cat([latents] * 3) if do_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            latent_model_input = torch.cat([latent_model_input, image_latents], dim=1)

            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
            )[0]

            if do_guidance:
                noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)
                noise_pred = (
                    noise_pred_uncond
                    + guidance_scale * (noise_pred_text - noise_pred_image)
                    + image_guidance_scale * (noise_pred_image - noise_pred_uncond)
                )

            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

        # 7. decode
        if output_type == "latent":
            image_out = latents
        else:
            image_out = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            image_out = self.image_processor.postprocess(
                image_out, output_type=output_type, do_denormalize=[True] * image_out.shape[0]
            )

        self.maybe_free_model_hooks()
        return StableDiffusionPipelineOutput(images=image_out, nsfw_content_detected=None)

    # ------------------------------------------------------- convenience wrapper
    def drive(
        self,
        image: Union[Image.Image, List[Image.Image]],
        prompt: Union[str, List[str]],
        num_inference_steps: Optional[int] = None,
        image_guidance_scale: Optional[float] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> Union[Image.Image, List[Image.Image]]:
        """DrivingGFM unified entry point: image + instruction -> edited image.

        Args:
            image: PIL image or a list of images.
            prompt: editing instruction (one per image).
            num_inference_steps: denoising steps, default 20.
            image_guidance_scale: structural fidelity to the source image, default 1.5.
            guidance_scale: instruction adherence, default 7.5.
            seed: random seed for reproducibility.
        """
        steps = DEFAULTS["num_inference_steps"] if num_inference_steps is None else num_inference_steps
        igs = DEFAULTS["image_guidance_scale"] if image_guidance_scale is None else image_guidance_scale
        gs = DEFAULTS["guidance_scale"] if guidance_scale is None else guidance_scale

        generator = None
        if seed is not None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            generator = torch.Generator(device=device).manual_seed(int(seed))

        out: StableDiffusionPipelineOutput = self(
            image=image,
            prompt=prompt,
            num_inference_steps=steps,
            image_guidance_scale=igs,
            guidance_scale=gs,
            generator=generator,
            **kwargs,
        )
        return out.images[0] if len(out.images) == 1 else out.images


# Alias matching `_class_name` in model_index.json, used by diffusers when resolving
# this module through `custom_pipeline`.
Drivinggfm = DrivinggfmPipeline

import logging
from collections.abc import Iterator
from typing import Literal

import torch

from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.protocols import DiffusionStepProtocol
from ltx_core.model.audio_vae import encode_audio as vae_encode_audio
from ltx_core.model.upsampler import upsample_video
from ltx_core.model.video_vae import TilingConfig
from ltx_core.model.video_vae import decode_video as vae_decode_video
from ltx_core.quantization import QuantizationPolicy
from ltx_core.types import Audio, AudioLatentShape, LatentState, VideoPixelShape
from ltx_pipelines.utils import ModelLedger
from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_pipelines.utils.constants import DEFAULT_IMAGE_CRF, DISTILLED_SIGMA_VALUES, STAGE_2_DISTILLED_SIGMA_VALUES
from ltx_pipelines.utils.helpers import (
    assert_resolution,
    cleanup_memory,
    denoise_video_only,
    encode_prompts,
    get_device,
    image_conditionings_by_replacing_latent,
    simple_denoising_func,
)
from ltx_pipelines.utils.media_io import decode_audio_from_file
from ltx_pipelines.utils.samplers import euler_denoising_loop
from ltx_pipelines.utils.types import PipelineComponents

device = get_device()

CleanupMode = Literal["aggressive", "balanced", "none"]


class DistilledA2VPipeline:
    """Two-stage distilled audio-to-video pipeline (image + audio → video).

    Stage 1 generates video at half resolution with frozen audio conditioning,
    then Stage 2 upsamples by 2x and refines with a distilled sigma schedule.

    Notes on VRAM:
    - This pipeline does *not* do CPU/disk offloading.
    - It aggressively deletes large modules between stages (configurable via
      ``cleanup_mode``) to reduce peak VRAM.
    """

    def __init__(
        self,
        distilled_checkpoint_path: str,
        gemma_root: str,
        spatial_upsampler_path: str,
        loras: tuple = (),
        device: torch.device = device,
        quantization: QuantizationPolicy | None = None,
        cleanup_mode: CleanupMode = "aggressive",
    ) -> None:
        self.device = device
        self.dtype = torch.bfloat16
        self.cleanup_mode: CleanupMode = cleanup_mode

        self.model_ledger = ModelLedger(
            dtype=self.dtype,
            device=device,
            checkpoint_path=distilled_checkpoint_path,
            gemma_root_path=gemma_root,
            spatial_upsampler_path=spatial_upsampler_path,
            loras=loras,
            quantization=quantization,
        )
        self.pipeline_components = PipelineComponents(dtype=self.dtype, device=device)

    @torch.inference_mode()
    def __call__(  # noqa: PLR0913
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[tuple[str, int, float]],
        audio_path: str,
        audio_start_time: float = 0.0,
        audio_max_duration: float | None = None,
        tiling_config: TilingConfig | None = None,
        enhance_prompt: bool = False,
        *,
        progress: bool = True,
        progress_leave: bool = True,
        progress_position: int = 0,
        progress_vram: bool = True,
        progress_vram_every: int = 1,
        progress_bar_format: str | None = None,
    ) -> tuple[Iterator[torch.Tensor], Audio]:
        assert_resolution(height=height, width=width, is_two_stage=True)
        if not images:
            raise ValueError("At least one image conditioning input is required for DistilledA2VPipeline.")

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        stepper = EulerDiffusionStep()
        dtype = torch.bfloat16

        (ctx_p,) = encode_prompts(
            [prompt],
            self.model_ledger,
            enhance_first_prompt=enhance_prompt,
            enhance_prompt_image=images[0][0],
        )
        video_context, audio_context = ctx_p.video_encoding, ctx_p.audio_encoding

        # --- Audio encode ----------------------------------------------------
        decoded_audio = decode_audio_from_file(audio_path, self.device, audio_start_time, audio_max_duration)
        assert decoded_audio is not None, "Audio file contains no audio stream"
        audio_encoder = self.model_ledger.audio_encoder()
        encoded_audio_latent = vae_encode_audio(decoded_audio, audio_encoder)
        del audio_encoder
        if self.cleanup_mode != "none":
            cleanup_memory()

        audio_shape = AudioLatentShape.from_duration(batch=1, duration=num_frames / frame_rate, channels=8, mel_bins=16)
        target_frames = audio_shape.frames
        if encoded_audio_latent.shape[2] < target_frames:
            pad_size = target_frames - encoded_audio_latent.shape[2]
            encoded_audio_latent = torch.nn.functional.pad(encoded_audio_latent, (0, 0, 0, pad_size))
        else:
            encoded_audio_latent = encoded_audio_latent[:, :, :target_frames]

        # Materialize image inputs in the expected NamedTuple shape.
        ltx_images = [ImageConditioningInput(path, frame_idx, strength, DEFAULT_IMAGE_CRF) for path, frame_idx, strength in images]

        # --- Stage 1: Half-resolution denoise (video only; audio frozen) -----
        stage_1_sigmas = torch.tensor(DISTILLED_SIGMA_VALUES, device=self.device)
        stage_1_output_shape = VideoPixelShape(
            batch=1,
            frames=num_frames,
            width=width // 2,
            height=height // 2,
            fps=frame_rate,
        )

        video_encoder = self.model_ledger.video_encoder()
        stage_1_conditionings = image_conditionings_by_replacing_latent(
            images=ltx_images,
            height=stage_1_output_shape.height,
            width=stage_1_output_shape.width,
            video_encoder=video_encoder,
            dtype=dtype,
            device=self.device,
        )

        if self.cleanup_mode in ("aggressive", "balanced"):
            torch.cuda.synchronize()
            del video_encoder
            cleanup_memory()

        transformer = self.model_ledger.transformer()

        def stage_1_denoising_loop(
            sigmas: torch.Tensor, video_state: LatentState, audio_state: LatentState, stepper: DiffusionStepProtocol
        ) -> tuple[LatentState, LatentState]:
            return euler_denoising_loop(
                sigmas=sigmas,
                video_state=video_state,
                audio_state=audio_state,
                stepper=stepper,
                denoise_fn=simple_denoising_func(
                    video_context=video_context,
                    audio_context=audio_context,
                    transformer=transformer,  # noqa: F821
                ),
                progress=progress,
                progress_desc="Stage 1 denoise (distilled A2V)",
                progress_leave=progress_leave,
                progress_position=progress_position,
                progress_vram=progress_vram,
                progress_vram_every=progress_vram_every,
                progress_bar_format=progress_bar_format,
            )

        video_state = denoise_video_only(
            output_shape=stage_1_output_shape,
            conditionings=stage_1_conditionings,
            noiser=noiser,
            sigmas=stage_1_sigmas,
            stepper=stepper,
            denoising_loop_fn=stage_1_denoising_loop,
            components=self.pipeline_components,
            dtype=dtype,
            device=self.device,
            initial_audio_latent=encoded_audio_latent,
        )

        if self.cleanup_mode in ("aggressive", "balanced"):
            torch.cuda.synchronize()
            del transformer
            cleanup_memory()

        # --- Upsample to full resolution in latent space --------------------
        if self.cleanup_mode in ("aggressive", "balanced"):
            video_encoder = self.model_ledger.video_encoder()
        upsampler = self.model_ledger.spatial_upsampler()
        upscaled_video_latent = upsample_video(
            latent=video_state.latent[:1],
            video_encoder=video_encoder,
            upsampler=upsampler,
        )
        del upsampler

        # --- Stage 2: Full-resolution refinement (video only; audio frozen) --
        stage_2_sigmas = torch.tensor(STAGE_2_DISTILLED_SIGMA_VALUES, device=self.device)
        stage_2_output_shape = VideoPixelShape(batch=1, frames=num_frames, width=width, height=height, fps=frame_rate)
        stage_2_conditionings = image_conditionings_by_replacing_latent(
            images=ltx_images,
            height=stage_2_output_shape.height,
            width=stage_2_output_shape.width,
            video_encoder=video_encoder,
            dtype=dtype,
            device=self.device,
        )

        if self.cleanup_mode != "none":
            torch.cuda.synchronize()
            del video_encoder
            cleanup_memory()

        if self.cleanup_mode == "none":
            # Reuse transformer from stage 1 to save time (higher peak VRAM).
            pass
        else:
            transformer = self.model_ledger.transformer()

        def stage_2_denoising_loop(
            sigmas: torch.Tensor, video_state: LatentState, audio_state: LatentState, stepper: DiffusionStepProtocol
        ) -> tuple[LatentState, LatentState]:
            return euler_denoising_loop(
                sigmas=sigmas,
                video_state=video_state,
                audio_state=audio_state,
                stepper=stepper,
                denoise_fn=simple_denoising_func(
                    video_context=video_context,
                    audio_context=audio_context,
                    transformer=transformer,  # noqa: F821
                ),
                progress=progress,
                progress_desc="Stage 2 denoise (distilled A2V)",
                progress_leave=progress_leave,
                progress_position=progress_position,
                progress_vram=progress_vram,
                progress_vram_every=progress_vram_every,
                progress_bar_format=progress_bar_format,
            )

        video_state = denoise_video_only(
            output_shape=stage_2_output_shape,
            conditionings=stage_2_conditionings,
            noiser=noiser,
            sigmas=stage_2_sigmas,
            stepper=stepper,
            denoising_loop_fn=stage_2_denoising_loop,
            components=self.pipeline_components,
            dtype=dtype,
            device=self.device,
            noise_scale=float(stage_2_sigmas[0].item()),
            initial_video_latent=upscaled_video_latent,
            initial_audio_latent=encoded_audio_latent,
        )

        if self.cleanup_mode != "none":
            torch.cuda.synchronize()
            del transformer
            cleanup_memory()

        decoded_video = vae_decode_video(video_state.latent, self.model_ledger.video_decoder(), tiling_config, generator)

        # Trim waveform to target video duration so the muxed output doesn't
        # extend beyond the generated video frames.
        max_samples = round(num_frames / frame_rate * decoded_audio.sampling_rate)
        trimmed_waveform = decoded_audio.waveform.squeeze(0)[..., :max_samples]
        original_audio = Audio(waveform=trimmed_waveform, sampling_rate=decoded_audio.sampling_rate)

        logging.getLogger(__name__).info("Generation complete.")
        return decoded_video, original_audio


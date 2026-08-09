"""
vla_inference.py — OpenVLA model loading and inference wrapper.

Provides a clean interface for running OpenVLA-7B inference in the
AirSim benchmark pipeline. Standalone (no ROS dependency).

The model takes a camera image + language instruction and outputs
a 7-DoF action delta vector. For drone navigation, we extract the
first 3 components (dx, dy, dz) as a directional signal.
"""

import logging
import time
import warnings
from typing import Optional, Tuple

import numpy as np

warnings.filterwarnings(
    "ignore",
    message="Argument 'interpolation' of type int is deprecated",
    category=UserWarning,
)

logger = logging.getLogger(__name__)

try:
    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor
    _HAS_VLA = True
except ImportError as e:
    _HAS_VLA = False
    logger.warning(f"torch/transformers not available — VLA inference disabled ({e})")


class OpenVLAInference:
    """Wrapper for OpenVLA-7B inference.

    Loads the model once at construction, then provides fast inference calls.

    Args:
        model_path: Local path or HuggingFace model ID (e.g. "openvla/openvla-7b").
        device: CUDA device string (default: "cuda:0").
        use_flash_attn: Whether to use Flash Attention 2 (faster, requires flash-attn).
    """

    PROMPT_TEMPLATE = "In: What action should the robot take to {instruction}?\nOut:"

    def __init__(
        self,
        model_path: str = "openvla/openvla-7b",
        device: str = "cuda:0",
        use_flash_attn: bool = False,
    ):
        if not _HAS_VLA:
            raise RuntimeError(
                "PyTorch and transformers are required for VLA inference. "
                "Install with: pip install torch transformers timm accelerate"
            )

        self._device = device
        self._model_path = model_path
        self._last_inference_time: float = 0.0
        self._inference_count: int = 0

        logger.info(f"Loading OpenVLA from '{model_path}' on {device}...")
        t0 = time.time()

        self._processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )

        # PyTorch < 2.0 doesn't support bfloat16/float16 for some CPU ops
        # (linspace, triu) during model construction. Strategy:
        # - PyTorch >= 2: load directly in bfloat16
        # - PyTorch < 2: load in float32, then cast to float16 on GPU
        torch_major = int(torch.__version__.split('.')[0])
        if torch_major >= 2:
            load_dtype = torch.bfloat16
            self._dtype = torch.bfloat16
            dtype_name = "bfloat16"
        else:
            load_dtype = torch.float32
            self._dtype = torch.float16
            dtype_name = "float16 (loaded as fp32, cast after)"
            logger.info(f"PyTorch {torch.__version__} — loading in fp32, will cast to fp16 on GPU")

        attn_impl = "flash_attention_2" if use_flash_attn else "eager"
        model_kwargs = dict(
            torch_dtype=load_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        try:
            self._model = AutoModelForVision2Seq.from_pretrained(
                model_path, attn_implementation=attn_impl, **model_kwargs
            )
        except TypeError:
            logger.warning("attn_implementation not supported — loading without it")
            self._model = AutoModelForVision2Seq.from_pretrained(
                model_path, **model_kwargs
            )

        # Move to device (and cast to fp16 for PyTorch < 2.0)
        if torch_major < 2:
            self._model = self._model.half().to(device)
        else:
            self._model = self._model.to(device)

        load_time = time.time() - t0
        logger.info(f"OpenVLA loaded in {load_time:.1f}s "
                    f"(attn={attn_impl}, dtype={dtype_name})")

    @property
    def inference_count(self) -> int:
        return self._inference_count

    @property
    def last_inference_time(self) -> float:
        return self._last_inference_time

    def predict_action(
        self, image: np.ndarray, instruction: str, max_retries: int = 2
    ) -> np.ndarray:
        """Run a single VLA inference pass with retry on transient errors.

        Args:
            image: RGB image as numpy array (H, W, 3), uint8.
            instruction: Natural language task instruction.
            max_retries: Number of retries on RuntimeError (e.g. tensor shape bugs).

        Returns:
            7-element numpy array: (dx, dy, dz, droll, dpitch, dyaw, gripper).
            For drone navigation, only the first 3 (position deltas) are used.
        """
        from PIL import Image as PILImage

        prompt = self.PROMPT_TEMPLATE.format(instruction=instruction)
        pil_img = PILImage.fromarray(image)

        inputs = self._processor(prompt, pil_img).to(
            self._device, dtype=self._dtype
        )

        # predict_action() may append a special token (29871) to input_ids,
        # making it 1 longer than the processor's attention_mask. Drop the
        # mask so generate() builds a fresh one matching the actual length.
        inputs.pop("attention_mask", None)

        last_error: Optional[Exception] = None
        for attempt in range(1 + max_retries):
            try:
                t0 = time.time()
                with torch.inference_mode():
                    action = self._model.predict_action(
                        **inputs,
                        unnorm_key="bridge_orig",
                        do_sample=False,
                    )
                self._last_inference_time = time.time() - t0
                self._inference_count += 1

                action_np = np.array(action, dtype=np.float64)

                if self._inference_count <= 3 or self._inference_count % 10 == 0:
                    logger.debug(
                        f"VLA inference #{self._inference_count} "
                        f"({self._last_inference_time:.3f}s): "
                        f"delta=({action_np[0]:.4f}, {action_np[1]:.4f}, {action_np[2]:.4f})"
                    )

                return action_np

            except RuntimeError as e:
                last_error = e
                if attempt < max_retries:
                    logger.warning(
                        f"VLA inference attempt {attempt + 1} failed: {e} — retrying"
                    )
                    torch.cuda.empty_cache()
                    time.sleep(0.1)

        logger.error(f"VLA inference failed after {max_retries + 1} attempts: {last_error}")
        return np.zeros(7, dtype=np.float64)

    def predict_direction(self, image: np.ndarray, instruction: str) -> np.ndarray:
        """Extract 3D directional signal from VLA output.

        Returns:
            3-element numpy array (dx, dy, dz) — the position delta component.
        """
        action = self.predict_action(image, instruction)
        return action[:3]

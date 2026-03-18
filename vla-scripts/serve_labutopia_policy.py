#!/usr/bin/env python3
"""Serve a LabUtopia-finetuned UniVLA checkpoint over the websocket protocol used by LabUtopia."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import threading
from typing import Any, List

import msgpack
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
from websockets.exceptions import ConnectionClosed
from websockets.sync.server import serve

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.models.policy.transformer_utils import MAPBlock


def compose_camera_arrays(frames: List[np.ndarray], layout: str) -> np.ndarray:
    if len(frames) == 0:
        raise ValueError("`frames` must contain at least one image array.")

    if len(frames) == 1 or layout == "single":
        return frames[0]

    if layout not in {"grid_2x2", "horizontal", "vertical"}:
        raise ValueError(f"Unsupported image layout: {layout}")

    base_h, base_w = frames[0].shape[:2]
    normalized_frames = []
    for frame in frames:
        if frame.shape[:2] != (base_h, base_w):
            pil = Image.fromarray(frame)
            frame = np.asarray(pil.resize((base_w, base_h)))
        normalized_frames.append(frame)
    frames = normalized_frames

    if layout == "horizontal":
        return np.concatenate(frames, axis=1)

    if layout == "vertical":
        return np.concatenate(frames, axis=0)

    padded_frames = list(frames[:4])
    while len(padded_frames) < 4:
        padded_frames.append(np.zeros_like(frames[0]))

    top = np.concatenate([padded_frames[0], padded_frames[1]], axis=1)
    bottom = np.concatenate([padded_frames[2], padded_frames[3]], axis=1)
    return np.concatenate([top, bottom], axis=0)


def _pack_array(obj: Any) -> Any:
    if (isinstance(obj, (np.ndarray, np.generic))) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")

    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }

    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }

    return obj


def _unpack_array(obj: dict[str, Any]) -> Any:
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])

    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])

    return obj


def packb(obj: Any) -> bytes:
    return msgpack.packb(obj, default=_pack_array)


def unpackb(payload: bytes) -> Any:
    return msgpack.unpackb(payload, object_hook=_unpack_array)


class ActionDecoder(nn.Module):
    """Action decoder matching `vla-scripts/finetune_labutopia.py`."""

    def __init__(self, window_size: int = 12, action_dim: int = 8, hidden_dim: int = 512, vla_hidden_dim: int = 4096):
        super().__init__()
        self.window_size = window_size
        self.action_dim = action_dim
        self.attn_pool = MAPBlock(
            n_latents=1,
            vis_dim=vla_hidden_dim,
            embed_dim=hidden_dim,
            n_heads=hidden_dim // 64,
        )
        self.visual_pool = MAPBlock(
            n_latents=1,
            vis_dim=vla_hidden_dim,
            embed_dim=hidden_dim,
            n_heads=hidden_dim // 64,
        )
        self.proprio_proj = nn.Sequential(
            nn.Linear(7, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, window_size * action_dim),
        )

    def forward(self, latent_action_tokens: torch.Tensor, visual_embed: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        proprio = self.proprio_proj(proprio)
        visual_embed = self.visual_pool(visual_embed)
        action = self.proj(
            torch.cat([self.attn_pool(latent_action_tokens, init_embed=visual_embed), proprio], dim=-1)
        )
        return action


class LabUtopiaUniVLAService:
    def __init__(
        self,
        model_path: str,
        decoder_path: str,
        camera_keys: List[str],
        image_layout: str,
        window_size: int,
        action_dim: int,
        device: str,
        do_sample: bool,
        temperature: float,
        top_p: float,
    ) -> None:
        self.model_path = str(Path(model_path).expanduser())
        self.decoder_path = str(Path(decoder_path).expanduser())
        self.camera_keys = camera_keys
        self.image_layout = image_layout
        self.window_size = window_size
        self.action_dim = action_dim
        self.device = torch.device(device)
        self.model_dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p
        self._lock = threading.Lock()
        self._fallback_camera_logged = False

        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self.vla = self._load_model(self.model_path)
        self.vla.eval()

        self.action_decoder = ActionDecoder(window_size=self.window_size, action_dim=self.action_dim)
        decoder_state = torch.load(self.decoder_path, map_location="cpu")
        self.action_decoder.load_state_dict(decoder_state, strict=True)
        self.action_decoder.to(self.device)
        self.action_decoder.eval()

    def _load_model(self, model_path: str) -> torch.nn.Module:
        load_kwargs: dict[str, Any] = {
            "torch_dtype": self.model_dtype,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
        }
        if self.device.type == "cuda":
            load_kwargs["attn_implementation"] = "flash_attention_2"

        try:
            model = AutoModelForVision2Seq.from_pretrained(model_path, **load_kwargs)
        except Exception as exc:
            if load_kwargs.pop("attn_implementation", None) is None:
                raise
            logging.warning("Falling back to default attention because FlashAttention failed: %s", exc)
            model = AutoModelForVision2Seq.from_pretrained(model_path, **load_kwargs)

        return model.to(self.device)

    def metadata(self) -> dict[str, Any]:
        return {
            "server": "univla-labutopia-ws",
            "model_path": self.model_path,
            "decoder_path": self.decoder_path,
            "camera_keys": self.camera_keys,
            "image_layout": self.image_layout,
            "window_size": self.window_size,
            "action_dim": self.action_dim,
            "device": str(self.device),
        }

    def infer(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        image = self._extract_image(obs)
        instruction = self._extract_instruction(obs)
        proprio = self._extract_proprio(obs)
        prompt = self._build_prompt(instruction)

        inputs = self.processor(prompt, image).to(self.device, dtype=self.model_dtype)
        proprio_tensor = torch.from_numpy(proprio).unsqueeze(0).to(self.device, dtype=torch.float32)

        with self._lock, torch.inference_mode():
            latent_action, visual_embed, _ = self.vla.predict_latent_action(
                **inputs,
                do_sample=self.do_sample,
                temperature=self.temperature,
                top_p=self.top_p,
            )
            actions = self.action_decoder(
                latent_action.to(torch.float32),
                visual_embed.to(torch.float32),
                proprio_tensor,
            )
            actions = actions.reshape(-1, self.window_size, self.action_dim)[0].detach().cpu().numpy().astype(np.float32)

        return {"action": actions}

    def _extract_instruction(self, obs: dict[str, Any]) -> str:
        instruction = obs.get("language_instruction") or obs.get("prompt")
        if not instruction:
            raise ValueError("Missing `language_instruction` or `prompt` in request payload.")
        return str(instruction).strip()

    def _extract_proprio(self, obs: dict[str, Any]) -> np.ndarray:
        if "state" not in obs:
            raise ValueError("Missing `state` in request payload.")
        state = np.asarray(obs["state"], dtype=np.float32).reshape(-1)
        if state.size < 7:
            raise ValueError(f"Expected at least 7 proprio values, got shape {state.shape}.")
        return state[:7]

    def _extract_image(self, obs: dict[str, Any]) -> Image.Image:
        def normalize_array(image_key: str) -> np.ndarray:
            array = np.asarray(obs[image_key])
            if array.ndim == 4:
                array = array[-1]
            if array.ndim == 3 and array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
                array = np.transpose(array, (1, 2, 0))
            if np.issubdtype(array.dtype, np.floating):
                scale = 255.0 if array.max() <= 1.0 else 1.0
                array = np.clip(array * scale, 0, 255).astype(np.uint8)
            elif array.dtype != np.uint8:
                array = np.clip(array, 0, 255).astype(np.uint8)
            if array.ndim == 2:
                array = np.repeat(array[:, :, None], 3, axis=2)
            if array.ndim != 3:
                raise ValueError(f"Unsupported image shape for `{image_key}`: {array.shape}")
            if array.shape[2] == 1:
                array = np.repeat(array, 3, axis=2)
            elif array.shape[2] > 3:
                array = array[:, :, :3]
            return array

        missing_keys = [key for key in self.camera_keys if key not in obs]
        if missing_keys:
            raise ValueError(
                f"Missing camera observations {missing_keys}. Available keys: {list(obs.keys())}"
            )

        frames = [normalize_array(image_key) for image_key in self.camera_keys]
        composed = compose_camera_arrays(frames, self.image_layout)
        return Image.fromarray(composed)

    @staticmethod
    def _build_prompt(instruction: str) -> str:
        prompt_builder = PurePromptBuilder("openvla")
        prompt_builder.add_turn("human", f"What action should the robot take to {instruction.lower()}?")
        return prompt_builder.get_prompt()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a LabUtopia-finetuned UniVLA checkpoint over websocket.")
    parser.add_argument("--model-path", required=True, help="Merged Hugging Face UniVLA checkpoint directory.")
    parser.add_argument("--decoder-path", required=True, help="Path to action_decoder-<step>.pt.")
    parser.add_argument("--bind-host", default="0.0.0.0", help="Server bind host.")
    parser.add_argument("--port", type=int, default=20001, help="Server port.")
    parser.add_argument(
        "--camera-keys",
        default="camera_1_rgb,camera_2_rgb,camera_3_rgb",
        help="Comma-separated LabUtopia observation keys to compose into one image.",
    )
    parser.add_argument(
        "--image-layout",
        default="grid_2x2",
        choices=["single", "horizontal", "vertical", "grid_2x2"],
        help="How to compose multiple camera views into one image.",
    )
    parser.add_argument("--window-size", type=int, default=12, help="Action chunk length used during finetuning.")
    parser.add_argument("--action-dim", type=int, default=8, help="Continuous action dimension.")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu", help="Torch device.")
    parser.add_argument("--temperature", type=float, default=0.75, help="Sampling temperature for latent-action decoding.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling parameter.")
    parser.add_argument(
        "--do-sample",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable sampling during latent-action generation.",
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="[%(asctime)s] %(levelname)s %(message)s")

    service = LabUtopiaUniVLAService(
        model_path=args.model_path,
        decoder_path=args.decoder_path,
        camera_keys=[key.strip() for key in args.camera_keys.split(",") if key.strip()],
        image_layout=args.image_layout,
        window_size=args.window_size,
        action_dim=args.action_dim,
        device=args.device,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    logging.info("Loaded UniVLA LabUtopia service: %s", service.metadata())

    def handler(conn) -> None:
        peer = getattr(conn, "remote_address", "unknown")
        logging.info("Connection opened from %s", peer)
        conn.send(packb(service.metadata()))
        try:
            while True:
                payload = conn.recv()
                if isinstance(payload, str):
                    conn.send("Expected binary msgpack payload from client.")
                    continue
                response = service.infer(unpackb(payload))
                conn.send(packb(response))
        except ConnectionClosed:
            logging.info("Connection closed from %s", peer)
        except Exception as exc:
            logging.exception("Inference loop failed for %s", peer)
            try:
                conn.send(f"{type(exc).__name__}: {exc}")
            except Exception:
                pass

    with serve(handler, args.bind_host, args.port, compression=None, max_size=None) as server:
        logging.info("Serving websocket policy on ws://%s:%s", args.bind_host, args.port)
        server.serve_forever()


if __name__ == "__main__":
    main()

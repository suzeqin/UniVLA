"""
UniVLA 微调脚本 - 使用 LeRobot V3.0 格式的 LabUtopia 数据

基于 finetune_realworld.py 改造，支持从 LeRobot V3.0 格式数据集
(parquet + mp4) 直接加载 LabUtopia 数据，无需转换为 HDF5/RLDS 格式。

架构说明：
    UniVLA 微调分两部分：
    1. VLA 骨干（OpenVLA = Llama-2-7B + DINOSigLIP-224px）+ LoRA 微调
       - 输入：图像 + 语言指令 + 历史 latent action tokens
       - 输出：预测 latent action tokens (codebook_size=16)
    2. ActionDecoder（小型 Transformer + MLP）
       - 输入：VLA 最后一层隐状态中的 latent action tokens + visual embeddings
       - 输出：连续动作序列（window_size × action_dim）

数据流：
    图像帧 → LAM(VQ-VAE) → latent action token indices → VLA label
    同时：图像帧 → VLA vision encoder → hidden states
    VLA hidden states 中 latent action 对应位置 → ActionDecoder → 连续动作

使用方法:
    cd /data1/rbc/UniVLA
    bash vla-scripts/finetune_labutopia.sh
"""

import os
import pickle
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import draccus
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torchvision.transforms as T
import tqdm
from accelerate import PartialState, Accelerator, DistributedDataParallelKwargs
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoConfig,
    AutoImageProcessor,
    BitsAndBytesConfig,
)

import wandb
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import (
    PrismaticImageProcessor,
    PrismaticProcessor,
)
from prismatic.models.policy.transformer_utils import MAPBlock

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ========================= LeRobot V3.0 数据集加载 =========================

class LeRobotLabUtopiaDataset(Dataset):
    """
    从 LeRobot V3.0 格式数据集加载 LabUtopia 数据用于 UniVLA 微调。

    数据结构:
      {root}/{repo_id}/
        ├── meta/
        │   ├── info.json          # 数据集元信息
        │   ├── tasks.parquet      # 任务描述
        │   ├── stats.json         # 统计信息
        │   └── episodes/          # episode 元信息
        ├── data/
        │   └── chunk-000/
        │       └── file-000.parquet  # 状态/动作数据
        └── videos/
            └── observation.images.*/  # 视频文件

    每个 parquet 行包含: observation.state, action, frame_index, episode_index, task_index, ...
    图像通过视频解码获取（本实现使用 torchvision.io 或 decord）。
    """

    def __init__(
        self,
        dataset_root: str,
        repo_id: str,
        camera_names: List[str],
        image_transform,
        window_size: int = 12,
        image_size: int = 224,
        action_dim: int = 8,
        state_dim: int = 8,
    ):
        import pandas as pd
        import json

        self.dataset_path = Path(dataset_root) / repo_id
        self.camera_names = camera_names
        self.image_transform = image_transform
        self.window_size = window_size
        self.image_size = image_size
        self.action_dim = action_dim
        self.state_dim = state_dim

        # 加载元信息
        with open(self.dataset_path / "meta" / "info.json") as f:
            self.info = json.load(f)

        # 加载任务描述
                # 加载任务描述
        tasks_df = pd.read_json(self.dataset_path / "meta" / "tasks.jsonl", lines=True)
        self.task_descriptions = {}
        for _, row in tasks_df.iterrows():
            if "task_index" in tasks_df.columns and "task" in tasks_df.columns:
                self.task_descriptions[row["task_index"]] = row["task"]
            else:
                self.task_descriptions[_] = row.name

        # 加载所有 parquet 数据
        data_files = sorted((self.dataset_path / "data").rglob("*.parquet"))
        dfs = [pd.read_parquet(f) for f in data_files]
        self.data = pd.concat(dfs, ignore_index=True)
        self.total_frames = len(self.data)

        # 构建 episode 边界
        self.episode_starts = {}
        self.episode_ends = {}
        for ep_idx in self.data["episode_index"].unique():
            ep_mask = self.data["episode_index"] == ep_idx
            ep_indices = self.data.index[ep_mask]
            self.episode_starts[ep_idx] = ep_indices[0]
            self.episode_ends[ep_idx] = ep_indices[-1]

        # 加载视频解码器（使用 decord 或 torchvision）
        self._setup_video_readers()

        # 图像变换：UniVLA 使用 224x224，DINOSigLIP 格式
        self.resize_transform = T.Compose([
            T.Resize((image_size, image_size), antialias=True),
        ])

        print(f"[LeRobotLabUtopiaDataset] Loaded {self.total_frames} frames, "
              f"{len(self.episode_starts)} episodes from {self.dataset_path}")

    def _setup_video_readers(self):
        """设置视频读取器，为每个相机和 chunk 建立映射。"""
        self.video_paths = {}
        video_dir = self.dataset_path / "videos"
        for cam in self.camera_names:
            cam_key = f"observation.images.{cam}"
            cam_dir = video_dir / cam_key
            if cam_dir.exists():
                for mp4 in sorted(cam_dir.rglob("*.mp4")):
                    # 从路径解析 chunk_index 和 file_index
                    chunk_idx = int(mp4.parent.name.split("-")[-1])
                    file_idx = int(mp4.stem.split("-")[-1])
                    self.video_paths[(cam, chunk_idx, file_idx)] = str(mp4)

        # 确定每个帧属于哪个 chunk 和 file
        chunks_size = self.info.get("chunks_size", 1000)
        self.chunks_size = chunks_size

    def _load_frame_image(self, global_idx: int, camera: str) -> torch.Tensor:
        """从视频文件中解码指定帧的图像。"""
        chunks_size = self.chunks_size
        chunk_idx = global_idx // chunks_size
        file_idx = 0  # 通常单文件
        video_key = (camera, chunk_idx, file_idx)

        if video_key not in self.video_paths:
            # 尝试找到正确的文件
            for k, v in self.video_paths.items():
                if k[0] == camera and k[1] == chunk_idx:
                    video_key = k
                    break

        video_path = self.video_paths.get(video_key)
        if video_path is None:
            # 返回黑色图像作为 fallback
            return torch.zeros(3, self.image_size, self.image_size)

        frame_in_video = global_idx % chunks_size

        try:
            import decord
            decord.bridge.set_bridge("torch")
            vr = decord.VideoReader(video_path)
            if frame_in_video >= len(vr):
                frame_in_video = len(vr) - 1
            frame = vr[frame_in_video]  # (H, W, C) torch tensor
            frame = frame.permute(2, 0, 1).float() / 255.0  # -> (C, H, W) [0, 1]
        except ImportError:
            # Fallback to torchvision
            import torchvision.io as tvio
            video, _, _ = tvio.read_video(video_path, pts_unit="sec")
            if frame_in_video >= len(video):
                frame_in_video = len(video) - 1
            frame = video[frame_in_video].permute(2, 0, 1).float() / 255.0

        frame = self.resize_transform(frame)
        return frame

    def __len__(self):
        return self.total_frames

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        ep_idx = int(row["episode_index"])
        frame_idx = int(row["frame_index"])
        task_idx = int(row["task_index"])

        # 语言指令
        instruction = self.task_descriptions.get(task_idx, "perform the task")

        # 状态和动作
        state = np.array(row["state"], dtype=np.float32)
        action_current = np.array(row["actions"], dtype=np.float32)

        # 构建动作窗口 (window_size 步)
        ep_start = self.episode_starts[ep_idx]
        ep_end = self.episode_ends[ep_idx]
        actions = []
        for w in range(self.window_size):
            future_idx = idx + w
            if future_idx <= ep_end:
                future_row = self.data.iloc[future_idx]
                if int(future_row["episode_index"]) == ep_idx:
                    actions.append(np.array(future_row["actions"], dtype=np.float32))
                else:
                    actions.append(actions[-1] if actions else action_current)
            else:
                actions.append(actions[-1] if actions else action_current)
        actions = np.stack(actions, axis=0)  # (window_size, action_dim)

        # 加载当前帧图像（第一个相机作为主相机）
        primary_cam = self.camera_names[0]
        current_image = self._load_frame_image(idx, primary_cam)

        # 加载下一帧图像用于 LAM 编码
        next_idx = min(idx + 1, ep_end)
        if int(self.data.iloc[next_idx]["episode_index"]) != ep_idx:
            next_idx = idx  # episode 末尾则使用当前帧
        next_image = self._load_frame_image(next_idx, primary_cam)

        # 将图像通过 image_transform 处理为 VLA 输入格式 (224x224)
        # UniVLA 的 processor 需要 PIL Image 输入
        from PIL import Image
        current_pil = Image.fromarray(
            (current_image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        )

        # 为 LAM 准备：224x224 tensor (C, H, W) [0, 1]
        lam_resize = T.Resize((224, 224), antialias=True)
        initial_pixel = lam_resize(current_image)
        target_pixel = lam_resize(next_image)

        return {
            "image": current_pil,
            "instruction": instruction,
            "actions": torch.tensor(actions, dtype=torch.float32),
            "proprio": torch.tensor(state[:7] if len(state) >= 7 else state, dtype=torch.float32),
            "initial_pixel_values": initial_pixel,
            "target_pixel_values": target_pixel,
        }


def collate_fn_labutopia(batch, processor, action_tokenizer):
    """
    自定义 collate 函数：将 batch 中的样本组装为 VLA 训练所需的格式。
    注意：latent action labels 在训练循环中动态生成（需要 LAM 前向传播）。
    """
    images = [item["image"] for item in batch]
    instructions = [item["instruction"] for item in batch]
    actions = torch.stack([item["actions"] for item in batch])
    proprios = torch.stack([item["proprio"] for item in batch])
    initial_pixels = torch.stack([item["initial_pixel_values"] for item in batch])
    target_pixels = torch.stack([item["target_pixel_values"] for item in batch])

    # 使用 processor 处理图像为 pixel_values
    pixel_values_list = []
    for img in images:
        pv = processor.image_processor.apply_transform(img)
        if isinstance(pv, np.ndarray):
            pv = torch.from_numpy(pv)
        pixel_values_list.append(pv)
    pixel_values = torch.stack(pixel_values_list)

    return {
        "pixel_values": pixel_values,
        "instructions": instructions,
        "actions": actions,
        "proprio": proprios,
        "initial_pixel_values": initial_pixels,
        "target_pixel_values": target_pixels,
    }


# ========================= 模型组件 =========================

class ActionDecoder(nn.Module):
    """
    低层动作解码器：从 VLA 隐状态中的 latent action tokens 预测连续动作。

    输入：
        latent_action_tokens: VLA 最后一层隐状态中 latent action 位置的特征
        visual_embed: VLA 最后一层隐状态中视觉 patch 特征
        proprio: 本体感知状态 (7-dim)
    输出：
        actions: (batch, window_size * action_dim) 连续动作
    """

    def __init__(self, window_size=12, action_dim=8, hidden_dim=512, vla_hidden_dim=4096):
        super().__init__()
        self.attn_pool = MAPBlock(
            n_latents=1, vis_dim=vla_hidden_dim, embed_dim=hidden_dim,
            n_heads=hidden_dim // 64,
        )
        self.visual_pool = MAPBlock(
            n_latents=1, vis_dim=vla_hidden_dim, embed_dim=hidden_dim,
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

    def forward(self, latent_action_tokens, visual_embed, proprio):
        proprio = self.proprio_proj(proprio)
        visual_embed = self.visual_pool(visual_embed)
        action = self.proj(
            torch.cat(
                [self.attn_pool(latent_action_tokens, init_embed=visual_embed), proprio],
                dim=-1,
            )
        )
        return action


class WrappedModel(nn.Module):
    """
    封装 VLA + ActionDecoder 的联合模型。

    训练模式下：
      1. VLA 前向：预测 latent action tokens (CE loss)
      2. ActionDecoder 前向：从 VLA 隐状态解码连续动作 (L1 loss)
      3. 总 loss = VLA CE loss + ActionDecoder L1 loss
    """

    def __init__(self, vla, freeze_vla=False, window_size=12, action_dim=8):
        super().__init__()
        self.vla = vla
        self.window_size = window_size
        self.action_dim = action_dim
        self.action_decoder = ActionDecoder(
            window_size=window_size, action_dim=action_dim,
        )
        if freeze_vla:
            self.vla.requires_grad_(False)

    def forward(self, batch):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            vla_output = self.vla(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch["pixel_values"],
                labels=batch["labels"],
                output_hidden_states=True,
            )
        loss, loss_one_step, latent_action_tokens = self.action_decoder_forward(
            batch, vla_output
        )
        return vla_output, loss, loss_one_step, latent_action_tokens

    def action_decoder_forward(self, batch, vla_output):
        # 提取视觉 patch embedding（前 num_patches 个 token）
        num_patches = self.vla.vision_backbone.featurizer.patch_embed.num_patches
        visual_embed = vla_output.hidden_states[-1][:, :num_patches].to(torch.float)
        # 提取语言 + latent action tokens（num_patches 之后）
        latent_tokens = vla_output.hidden_states[-1][:, num_patches:]
        action_gt = batch["labels"].to(latent_tokens.device)
        mask = action_gt > 32000  # latent action tokens 的词表 ID > 32000

        latent_action_tokens = []
        for idx, per_sample_tokens in enumerate(latent_tokens):
            per_sample_lat = per_sample_tokens[mask[idx], :]
            latent_action_tokens.append(per_sample_lat)
        latent_action_tokens = torch.stack(latent_action_tokens).to(torch.float)

        pred_action = self.action_decoder(
            latent_action_tokens, visual_embed, batch["proprio"]
        ).reshape(-1, self.window_size, self.action_dim)

        loss = torch.nn.functional.l1_loss(
            pred_action, batch["actions"], reduction="none"
        )
        loss_one_step = loss[:, 0].mean()
        loss = loss.mean()
        return loss, loss_one_step, latent_action_tokens


# ========================= 配置 =========================

@dataclass
class FinetuneConfig:
    # ---- 模型路径 ----
    vla_path: str = "/data/rbc/Embodied_models/univla-7b"
    lam_path: str = "latent_action_model/logs/task_centric_lam_stage2/epoch=0-step=200000.ckpt"

    # ---- 数据路径 ----
    dataset_root: str = "/data1/rbc/lerobot/.cache"
    repo_id: str = "LabUtopia/Level3_TransportBeaker"
    camera_names: str = "camera_1_rgb"  # 逗号分隔的相机名列表
    dataset_name: str = "labutopia_level3_transport_beaker"

    # ---- 输出路径 ----
    run_root_dir: Path = Path("runs")
    adapter_tmp_dir: Path = Path("adapter-tmp")

    # ---- 训练超参 ----
    batch_size: int = 4
    max_steps: int = 20000
    save_steps: int = 5000
    learning_rate: float = 3.5e-4
    grad_accumulation_steps: int = 2
    image_aug: bool = False
    num_workers: int = 4

    # ---- LAM 配置（与预训练 LAM 一致）----
    codebook_size: int = 16
    lam_model_dim: int = 768
    lam_latent_dim: int = 128
    lam_num_latents: int = 32
    lam_patch_size: int = 14
    lam_enc_blocks: int = 12
    lam_dec_blocks: int = 12
    lam_num_heads: int = 12

    # ---- 动作配置 ----
    window_size: int = 12
    action_dim: int = 8        # LabUtopia: 7 joints + 1 gripper

    # ---- LoRA / VLA 配置 ----
    freeze_vla: bool = False
    use_lora: bool = True
    lora_rank: int = 32
    lora_dropout: float = 0.0
    use_quantization: bool = False
    save_latest_checkpoint_only: bool = True

    # ---- Wandb ----
    wandb_project: str = "univla-finetune-labutopia"
    wandb_entity: str = "scivla"
    run_id_note: Optional[str] = None


@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    print(f"Fine-tuning UniVLA `{cfg.vla_path}` on `{cfg.dataset_name}`")

    assert torch.cuda.is_available(), "Fine-tuning requires GPU!"
    distributed_state = PartialState()

    if distributed_state.is_main_process:
        print("Main process (rank 0)")
    else:
        print(f"Worker process (rank {distributed_state.process_index})")

    torch.cuda.set_device(device_id := distributed_state.local_process_index)
    torch.cuda.empty_cache()

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(mixed_precision="bf16", kwargs_handlers=[ddp_kwargs])

    # ---- 实验 ID ----
    exp_id = (
        f"{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}"
        f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
        f"+lr-{cfg.learning_rate}"
    )
    if cfg.use_lora:
        exp_id += f"+lora-r{cfg.lora_rank}"
    exp_id += f"+ws-{cfg.window_size}+ad-{cfg.action_dim}"

    run_dir = cfg.run_root_dir / exp_id
    adapter_dir = cfg.adapter_tmp_dir / exp_id
    os.makedirs(run_dir, exist_ok=True)

    # ---- 注册 OpenVLA 到 HF AutoClasses ----
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    # ---- 加载 Processor & VLA ----
    quantization_config = None
    if cfg.use_quantization:
        assert cfg.use_lora
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )

    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    if cfg.use_quantization:
        vla = prepare_model_for_kbit_training(vla)
    else:
        vla = vla.to(device_id)

    # ---- LoRA ----
    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_config)
        vla.print_trainable_parameters()

    # ---- ActionTokenizer ----
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    # ---- 封装模型 ----
    wrapped_model = WrappedModel(
        vla=vla,
        freeze_vla=cfg.freeze_vla,
        window_size=cfg.window_size,
        action_dim=cfg.action_dim,
    ).to(device_id)

    trainable_params_count = sum(
        p.numel() for p in wrapped_model.parameters() if p.requires_grad
    )
    print(f"Total Trainable Params: {trainable_params_count:,}")

    # ---- 优化器 ----
    trainable_params = [p for p in wrapped_model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=int(cfg.max_steps * 0.5), gamma=0.1
    )

    # ---- 加载 LAM (Stage-2 Controllable DINO LAM) ----
    from latent_action_model.genie.modules.lam import ControllableDINOLatentActionModel

    latent_action_model = ControllableDINOLatentActionModel(
        in_dim=3,
        model_dim=cfg.lam_model_dim,
        latent_dim=cfg.lam_latent_dim,
        num_latents=cfg.codebook_size,
        patch_size=cfg.lam_patch_size,
        enc_blocks=cfg.lam_enc_blocks,
        dec_blocks=cfg.lam_dec_blocks,
        num_heads=cfg.lam_num_heads,
        dropout=0.0,
    )

    lam_ckpt = torch.load(cfg.lam_path, map_location="cpu")["state_dict"]
    new_ckpt = {k.replace("lam.", ""): v for k, v in lam_ckpt.items()}
    latent_action_model.load_state_dict(new_ckpt, strict=True)
    latent_action_model = latent_action_model.to(device_id).eval()

    # ---- 数据集 ----
    camera_list = [c.strip() for c in cfg.camera_names.split(",")]
    dataset = LeRobotLabUtopiaDataset(
        dataset_root=cfg.dataset_root,
        repo_id=cfg.repo_id,
        camera_names=camera_list,
        image_transform=processor.image_processor.apply_transform,
        window_size=cfg.window_size,
        image_size=224,
        action_dim=cfg.action_dim,
        state_dim=8,
    )

    from functools import partial

    collate = partial(collate_fn_labutopia, processor=processor, action_tokenizer=action_tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate,
        drop_last=True,
        pin_memory=True,
    )

    # ---- Accelerator prepare ----
    wrapped_model, latent_action_model, optimizer, scheduler, dataloader = accelerator.prepare(
        wrapped_model, latent_action_model, optimizer, scheduler, dataloader
    )

    # ---- Wandb ----
    if distributed_state.is_main_process:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=f"ft+{exp_id}",
            mode=os.environ.get("WANDB_MODE", "offline"),
        )

    # ---- 训练循环 ----
    recent_losses = deque(maxlen=cfg.grad_accumulation_steps)
    recent_accs = deque(maxlen=cfg.grad_accumulation_steps)

    with tqdm.tqdm(total=cfg.max_steps, leave=False, desc="Training") as progress:
        wrapped_model.train()
        optimizer.zero_grad()
        current_step = 0

        while current_step < cfg.max_steps:
            for batch_idx, batch in enumerate(dataloader):
                batch["initial_pixel_values"] = batch["initial_pixel_values"].to(device_id)
                batch["target_pixel_values"] = batch["target_pixel_values"].to(device_id)
                batch["pixel_values"] = batch["pixel_values"].to(torch.bfloat16).to(device_id)
                batch["actions"] = batch["actions"].to(device_id)
                batch["proprio"] = batch["proprio"].to(device_id)

                # ---- 使用 LAM 编码 latent action tokens ----
                with torch.no_grad():
                    video = torch.stack(
                        [batch["initial_pixel_values"], batch["target_pixel_values"]], dim=1
                    )
                    latent_action_idx_batch = latent_action_model.module.vq_encode(video)[
                        "indices"
                    ].squeeze()

                # ---- 构建 VLA 输入（input_ids + labels） ----
                input_ids_list = []
                labels_list = []
                for idx, latent_action_idx in enumerate(latent_action_idx_batch):
                    action_vocab = [f"<ACT_{i.item()}>" for i in latent_action_idx]
                    action_tokens = "".join(action_vocab)

                    prompt_builder = PurePromptBuilder("openvla")
                    conversation = [
                        {
                            "from": "human",
                            "value": f"What action should the robot take to {batch['instructions'][idx].lower()}?",
                        },
                        {"from": "gpt", "value": action_tokens},
                    ]
                    for turn in conversation:
                        prompt_builder.add_turn(turn["from"], turn["value"])

                    input_ids = processor.tokenizer(
                        prompt_builder.get_prompt(), add_special_tokens=True
                    ).input_ids
                    labels = list(input_ids)

                    input_ids = torch.tensor(input_ids)
                    labels = torch.tensor(labels)
                    labels[: -(len(action_vocab) + 1)] = -100

                    input_ids_list.append(input_ids)
                    labels_list.append(labels)

                input_ids = pad_sequence(
                    input_ids_list, batch_first=True,
                    padding_value=processor.tokenizer.pad_token_id,
                )
                labels = pad_sequence(labels_list, batch_first=True, padding_value=-100)

                input_ids = input_ids[:, : processor.tokenizer.model_max_length]
                labels = labels[:, : processor.tokenizer.model_max_length]
                attention_mask = input_ids.ne(processor.tokenizer.pad_token_id)

                batch["input_ids"] = input_ids
                batch["attention_mask"] = attention_mask
                batch["labels"] = labels

                # ---- 前向传播 ----
                output, act_loss, loss_one_step, lat_tokens = wrapped_model(batch)
                loss = act_loss if cfg.freeze_vla else act_loss + output.loss
                normalized_loss = loss / cfg.grad_accumulation_steps

                torch.nn.utils.clip_grad_norm_(wrapped_model.parameters(), max_norm=0.3)
                normalized_loss.backward()

                # ---- 计算 accuracy ----
                num_patches = wrapped_model.module.vla.vision_backbone.featurizer.patch_embed.num_patches
                action_logits = output.logits[:, num_patches:-1]
                action_preds = action_logits.argmax(dim=2)
                action_gt = batch["labels"][:, 1:].to(action_preds.device)
                mask = action_gt > 32000
                correct = (action_preds == action_gt) & mask
                accuracy = correct.sum().float() / mask.sum().float() if mask.sum() > 0 else torch.tensor(0.0)

                recent_losses.append(loss.item())
                recent_accs.append(accuracy.item())

                gradient_step_idx = batch_idx // cfg.grad_accumulation_steps

                # ---- 日志 ----
                if distributed_state.is_main_process:
                    wandb.log(
                        {
                            "train_loss": sum(recent_losses) / len(recent_losses),
                            "action_accuracy": sum(recent_accs) / len(recent_accs),
                            "action_l1_loss": act_loss.item(),
                            "action_l1_loss_1step": loss_one_step.item(),
                            "vla_ce_loss": output.loss.item() if not cfg.freeze_vla else 0,
                            "lr": optimizer.state_dict()["param_groups"][0]["lr"],
                        },
                        step=gradient_step_idx + current_step,
                    )

                # ---- 优化器步进 ----
                if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    progress.update()

                # ---- 保存检查点 ----
                global_step = gradient_step_idx + current_step
                if global_step > 0 and global_step % cfg.save_steps == 0:
                    if distributed_state.is_main_process:
                        print(f"Saving checkpoint at step {global_step}")
                        save_dir = str(run_dir) + f"/{global_step}"
                        os.makedirs(save_dir, exist_ok=True)

                        if not cfg.freeze_vla:
                            processor.save_pretrained(save_dir)
                            wrapped_model.module.vla.save_pretrained(
                                str(adapter_dir) + f"/{global_step}" if cfg.use_lora else save_dir
                            )

                        torch.save(
                            wrapped_model.module.action_decoder.state_dict(),
                            os.path.join(save_dir, f"action_decoder-{global_step}.pt"),
                        )

                    dist.barrier()

                    # ---- Merge LoRA ----
                    if cfg.use_lora and distributed_state.is_main_process:
                        base_vla = AutoModelForVision2Seq.from_pretrained(
                            cfg.vla_path,
                            torch_dtype=torch.bfloat16,
                            low_cpu_mem_usage=True,
                            trust_remote_code=True,
                        )
                        merged = PeftModel.from_pretrained(
                            base_vla, str(adapter_dir) + f"/{global_step}"
                        )
                        merged = merged.merge_and_unload()
                        merged.save_pretrained(str(run_dir) + f"/{global_step}")
                        print(f"Merged LoRA checkpoint saved at: {run_dir}/{global_step}")

                    dist.barrier()

                if global_step >= cfg.max_steps:
                    break

            current_step = gradient_step_idx + 1 + current_step
            if current_step >= cfg.max_steps:
                print(f"Max step {cfg.max_steps} reached!")
                if distributed_state.is_main_process:
                    wandb.finish()
                break


if __name__ == "__main__":
    finetune()

import copy
import multiprocessing
import os
import os.path
import shutil
import subprocess
import time
from datetime import datetime
from multiprocessing import Pool
from typing import List, Callable, Dict, Optional

import imageio
import numpy as np
import torch
from codetiming import Timer
from torch import Tensor

from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.agentic.agentic_config import AgenticConfig, RewardNormalizationConfig
from roll.pipeline.rlvr.utils import DUMPING_FUNC
from roll.utils.logging import get_logger
from roll.utils.functionals import (
    masked_whiten,
    compute_gae_advantage_return,
    compute_clip_fraction,
    compute_reinforce_return,
)

logger = get_logger()


def dump_rollout_render(save_dir, step, frames: List[List], env_ids: List, tags: List, episode_scores: List):
    with Timer(name="dump", logger=None) as timer:
        try:
            local_save_dir = f'/tmp/rollout_render/{datetime.now().strftime("%Y%m%d-%H%M%S")}'
            os.makedirs(local_save_dir, exist_ok=True)
            os.makedirs(save_dir, exist_ok=True)

            args_list = [
                (os.path.join(local_save_dir, f"{step}", f"{env_id}_{tag}_{episode_score:.1f}.gif"), frame_list)
                for frame_list, env_id, tag, episode_score in zip(frames, env_ids, tags, episode_scores)
                if len(frame_list) > 0
            ]
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            with Pool(processes=16) as pool:
                pool.starmap(dump_frames_as_gif, args_list)

            rar_file_path = os.path.join(
                "/tmp", f'rollout_render_{datetime.now().strftime("%Y%m%d-%H%M%S")}_{step}.zip'
            )
            command = ["zip", "-rq", rar_file_path, local_save_dir]
            subprocess.run(command, check=True)
            shutil.move(rar_file_path, save_dir)
            shutil.rmtree(local_save_dir, ignore_errors=True)
        except Exception as e:
            logger.error(f"dump rollout render failed: {e}")
    logger.info(f"dump_rollout_render_cost: {timer.last}")


@torch.no_grad()
def compute_discounted_returns(batch: DataProto, adv_estimator, gamma=1.0) -> DataProto:
    """
    Compute discounted returns for each trajectory in the batch.

    Args:
        batch (DataProto): A `DataProto` instance containing trajectories.
        adv_estimator (str): Advantage estimator type; only `"gigpo"` triggers computation here.
        gamma (float, optional): Discount factor applied to future rewards. Defaults to 1.0.

    Returns:
        DataProto: Updated batch where each trajectory contains an extra tensor key
                   `"step_rewards"` holding the computed discounted returns.
    """
    if adv_estimator in ["gigpo", "step_reinforce"]:
        batch.batch["sample_order_placeholder"] = torch.arange(batch.batch.batch_size[0], device=batch.batch.device)
        batch_group_by_traj: Dict[str, DataProto] = batch.group_by(keys="traj_id")
        for traj_id, traj_batch in batch_group_by_traj.items():

            indices: Tensor = torch.argsort(torch.from_numpy(traj_batch.non_tensor_batch["step"].astype(np.int64)))
            traj_batch.reorder(indices)
            step_scores = traj_batch.non_tensor_batch["step_scores"].astype(np.float32)
            rewards = torch.as_tensor(step_scores).float()
            discounts = torch.empty_like(rewards)
            running_return = 0.0
            for t in reversed(range(len(rewards))):
                running_return = rewards[t] + gamma * running_return
                discounts[t] = running_return
            traj_batch.batch["step_rewards"] = discounts

        merged = DataProto.concat(list(batch_group_by_traj.values()))
        merged.reorder(indices=torch.argsort(merged.batch["sample_order_placeholder"]))
        merged.pop("sample_order_placeholder")
        return merged
    else:
        return batch


# TODO: 这里的功能性和rlvr比较接近，但因为后续agentic会有潜在的修改需求，所以就先拎出来
@torch.no_grad()
def agentic_reward_norm(batch: "DataProto", reward_normalization: RewardNormalizationConfig) -> torch.Tensor:
    batch.batch["sample_order_placeholder"] = torch.arange(batch.batch.batch_size[0], device=batch.batch.device)
    grouping = reward_normalization.grouping
    norm_mean_type = reward_normalization.norm_mean_type
    norm_std_type = reward_normalization.norm_std_type

    all_scores = batch.batch["scores"].float()
    batch_mean = None
    batch_std = None
    if norm_mean_type == "batch":
        batch_mean = all_scores.mean()
    if norm_std_type == "batch":
        batch_std = all_scores.std()

    batch_list = []
    batch_grouped: Dict[str, DataProto] = {"default": batch}
    if grouping != "batch":
        batch_grouped = batch.group_by(keys=grouping)
    for group_name, group_batch in batch_grouped.items():
        scores = group_batch.batch["scores"]
        original_dtype = scores.dtype
        scores_float = scores.float()

        if norm_mean_type == "batch":
            reward_mean = batch_mean
        elif norm_mean_type == "group":
            reward_mean = scores_float.mean()
        else:
            reward_mean = 0.0

        if norm_std_type == "batch":
            reward_std = batch_std
        elif norm_std_type == "group":
            reward_std = scores_float.std()
        else:
            reward_std = None

        if reward_std is not None:
            # 处理单个元素或标准差为0的情况，避免除以0
            if scores_float.numel() > 1 and reward_std.abs() > 1e-6:
                normalized_scores = (scores_float - reward_mean) / (reward_std + 1e-6)
            else:
                normalized_scores = torch.zeros_like(scores_float)
        else:
            normalized_scores = scores_float - reward_mean

        normalized_scores = normalized_scores.to(dtype=original_dtype)
        group_batch.batch["grouped_rewards"] = normalized_scores
        batch_list.append(group_batch)

    batch = DataProto.concat(batch_list)
    batch.reorder(indices=torch.argsort(batch.batch["sample_order_placeholder"]))
    batch.pop("sample_order_placeholder")
    return batch.batch.pop("grouped_rewards")


def build_state_group(batch: "DataProto") -> "DataProto":
    batch.batch["sample_order_placeholder"] = torch.arange(batch.batch.batch_size[0], device=batch.batch.device)
    batch_group_by_traj_group: Dict[str, DataProto] = batch.group_by(keys="traj_group_id")
    merged = []
    for traj_group_id, traj_group_batch in batch_group_by_traj_group.items():
        batch_group_by_state: Dict[str, DataProto] = traj_group_batch.group_by(keys="state_hash")
        for state, state_batch in batch_group_by_state.items():
            state_batch.non_tensor_batch["state_group_id"] = np.array(
                [state] * state_batch.batch.batch_size[0], dtype=object
            )
            merged.append(state_batch)
    state_batch_size = [len(m) for m in merged]
    merged = DataProto.concat(merged)
    merged.reorder(indices=torch.argsort(merged.batch["sample_order_placeholder"]))
    merged.pop("sample_order_placeholder")
    metrics = merged.meta_info.pop("metrics", {})
    metrics["system/state_batch_size/max"] = np.max(state_batch_size)
    metrics["system/state_batch_size/mean"] = np.mean(state_batch_size)
    metrics["system/state_batch_size/min"] = np.min(state_batch_size)
    merged.meta_info["metrics"] = metrics
    return merged


@torch.no_grad()
def compute_response_level_rewards(batch: "DataProto", pipeline_config: AgenticConfig) -> "DataProto":
    reward_metrics = {}
    if pipeline_config.adv_estimator == "gigpo":
        # ref: https://github.com/langfengQ/verl-agent/blob/e03bd502667c45172e8c093cc506db8438ae8ab5/gigpo/core_gigpo.py#L109
        # step 1
        episode_scores = torch.from_numpy(batch.non_tensor_batch["episode_scores"].astype(np.float32))
        scores_to_group = DataProto.from_dict({"scores": episode_scores})
        scores_to_group.non_tensor_batch = batch.non_tensor_batch
        episode_rewards: torch.Tensor = agentic_reward_norm(scores_to_group, reward_normalization=pipeline_config.reward_normalization)
        # fallback: if normalization produced all zeros (e.g. single-sample group), keep raw episode scores
        try:
            if torch.max(torch.abs(episode_rewards)) < 1e-6:
                logger.warning(f"agentic_reward_norm produced near-zero episode_rewards, falling back to raw scores. episode_rewards_sample={episode_rewards.flatten()[:8].tolist()}")
                # dump a small debug file for offline inspection
                try:
                    os.makedirs("output/debug", exist_ok=True)
                    dump_path = os.path.join("output/debug", f"fallback_episode_rewards_{int(time.time())}.pt")
                    torch.save({"grouped": episode_rewards, "raw": scores_to_group.batch["scores"]}, dump_path)
                    logger.info(f"wrote fallback debug to {dump_path}")
                except Exception:
                    logger.exception("failed to write fallback debug file")
                episode_rewards = scores_to_group.batch["scores"].clone().detach()
        except Exception:
            logger.exception("error checking episode_rewards fallback condition")

        # step 2
        batch = build_state_group(batch=batch)

        # step 3
        scores_to_group = DataProto.from_dict({"scores": batch.batch["step_rewards"]})
        scores_to_group.non_tensor_batch = batch.non_tensor_batch
        step_rewards: torch.Tensor = agentic_reward_norm(batch=scores_to_group,
                                                         reward_normalization=RewardNormalizationConfig(grouping="state_group_id",
                                                                                                        method=pipeline_config.reward_normalization.method))
        try:
            if torch.max(torch.abs(step_rewards)) < 1e-6:
                logger.warning(f"agentic_reward_norm produced near-zero step_rewards, falling back to raw scores. step_rewards_sample={step_rewards.flatten()[:8].tolist()}")
                try:
                    os.makedirs("output/debug", exist_ok=True)
                    dump_path = os.path.join("output/debug", f"fallback_step_rewards_{int(time.time())}.pt")
                    torch.save({"grouped": step_rewards, "raw": scores_to_group.batch["scores"]}, dump_path)
                    logger.info(f"wrote fallback debug to {dump_path}")
                except Exception:
                    logger.exception("failed to write fallback debug file")
                step_rewards = scores_to_group.batch["scores"].clone().detach()
        except Exception:
            logger.exception("error checking step_rewards fallback condition")

        batch.batch["response_level_rewards"] = (
            pipeline_config.episode_reward_weight * episode_rewards + pipeline_config.step_reward_weight * step_rewards
        )
        batch.batch["episode_rewards_norm"] = episode_rewards
        batch.batch["step_rewards_norm"] = step_rewards
    elif pipeline_config.adv_estimator == "step_reinforce":
        scores_to_group = DataProto.from_dict({"scores": batch.batch["step_rewards"]})
        scores_to_group.non_tensor_batch = batch.non_tensor_batch
        # compute grouped normalized rewards; fallback to raw scores if normalization zeroes them
        grouped = agentic_reward_norm(scores_to_group, reward_normalization=pipeline_config.reward_normalization)
        try:
            if torch.max(torch.abs(grouped)) < 1e-6:
                logger.warning(f"agentic_reward_norm produced near-zero grouped rewards, falling back to raw scores. grouped_sample={grouped.flatten()[:8].tolist()}")
                try:
                    os.makedirs("output/debug", exist_ok=True)
                    dump_path = os.path.join("output/debug", f"fallback_grouped_rewards_{int(time.time())}.pt")
                    torch.save({"grouped": grouped, "raw": scores_to_group.batch["scores"]}, dump_path)
                    logger.info(f"wrote fallback debug to {dump_path}")
                except Exception:
                    logger.exception("failed to write fallback debug file")
                grouped = scores_to_group.batch["scores"].clone().detach()
        except Exception:
            logger.exception("error checking grouped fallback condition")
        batch.batch["response_level_rewards"] = grouped
    else:
        # 优先使用环境已写入的 response_level_rewards（已包含 episode_score-episode_cost）
        if "response_level_rewards" in batch.batch:
            base_rewards = batch.batch["response_level_rewards"].clone().detach()
        else:
            base_rewards = batch.batch["scores"].clone().sum(dim=-1)

        scores_to_group = DataProto.from_dict({"scores": base_rewards})
        scores_to_group.non_tensor_batch = batch.non_tensor_batch
        grouped = agentic_reward_norm(scores_to_group, reward_normalization=pipeline_config.reward_normalization)

        # 当归一化导致奖励几乎全为 0 时，回退到原始奖励，避免梯度为 0
        try:
            if torch.max(torch.abs(grouped)) < 1e-6:
                logger.warning(
                    f"agentic_reward_norm produced near-zero grouped rewards, falling back to raw scores. "
                    f"grouped_sample={grouped.flatten()[:8].tolist()}"
                )
                grouped = base_rewards.clone().detach()
        except Exception:
            logger.exception("error checking grouped fallback condition")

        batch.batch["response_level_rewards"] = grouped

    # 加上clip
    if pipeline_config.reward_clip:
        reward_metrics["critic/reward_clip_frac"] = compute_clip_fraction(
            values=batch.batch["response_level_rewards"],
            clip_min=-pipeline_config.reward_clip,
            clip_max=pipeline_config.reward_clip,
        )
        batch.batch["response_level_rewards"] = torch.clamp(
            batch.batch["response_level_rewards"], min=-pipeline_config.reward_clip, max=pipeline_config.reward_clip
        )

    return batch, reward_metrics


@torch.no_grad()
def get_agentic_response_level_mask(data: "DataProto", pipeline_config: AgenticConfig):
    batch_size = data.batch["response_mask"].size(0)
    mask_metrics = {}

    # mask相关策略
    data.batch["origin_response_mask"] = data.batch["response_mask"].clone()
    response_mask = data.batch["response_mask"][:, 1:].clone()

    final_sample_mask = torch.ones(batch_size, device=response_mask.device)

    if getattr(pipeline_config, "max_len_mask", False):
        # TODO 当前是混合多个的action/state，需要去判别，或者用别的方式过滤
        final_sample_mask = final_sample_mask
        mask_metrics["actor/max_len_mask_ratio"] = 1.0
    else:
        mask_metrics["actor/max_len_mask_ratio"] = 1.0

    expanded_sample_mask = final_sample_mask.unsqueeze(-1).expand_as(response_mask)
    final_response_mask = response_mask * expanded_sample_mask
    mask_metrics["actor/final_mask_ratio"] = final_sample_mask.mean().item()
    mask_metrics["actor/samples_used"] = final_sample_mask.sum().item()
    mask_metrics["actor/samples_total"] = float(batch_size)

    data.batch["final_response_mask"] = final_response_mask
    return data, mask_metrics


print_only_once = False

# Map base rollout paths to a timestamped subdirectory created for this run.
# Ensures we create one timestamped folder per base `path` and reuse it for
# subsequent dump calls so files from different runs don't overwrite each other.
_rollout_path_map = {}


def dump_frames_as_gif(filename, frames, duration=0.2):
    global print_only_once
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with imageio.get_writer(filename, mode="v", duration=duration) as writer:
            for frame in frames:
                writer.append_data(frame.astype(np.uint8))

    except Exception as e:
        if not print_only_once:
            print(f"Error saving gif: {e}")
        print_only_once = True
        pass


def dump_rollout_trajectories(path, global_step, data: DataProto):
    """
    Dumps rollout trajectories to persistent storage.

    The data is written using a column-based configuration defined in COLUMNS_CONFIG.
    Each column is specified as a list [column_name, data_type], where:
    - column_name: string identifier for the column
    - data_type: data type specification ('bigint', 'string', 'double', etc.)

    Example configuration:
    columns_config = [
        ['global_step', 'bigint'],
        ['id', 'string'],
        ['source', 'string'],
        # ... additional columns
    ]
    """
    if not path:
        return

    columns_config: Optional[List] = data.meta_info.get("COLUMNS_CONFIG", None)
    if columns_config is None:
        return

    # Operate on a deep copy to avoid mutating the original DataProto
    write_data = copy.deepcopy(data.non_tensor_batch)
    # NOTE: previous implementation removed keys from `data.non_tensor_batch` here,
    # which caused downstream code that expects fields like 'traj_id' to fail.
    # We must not mutate the original `data` in-place; only remove from the copy
    # if needed by the writer implementation. Keep original intact.

    data_cnt = len(data)
    write_data['global_step'] = [global_step] * data_cnt
    columns_config.append(['global_step','bigint'])

    # Ensure we don't overwrite a previous run's rollouts: create (once per
    # base `path`) a timestamped subdirectory and use it for all writes in
    # this process. This keeps the external API (accepting `path`) unchanged
    # while avoiding clobbering files across runs.
    try:
        base_path = path
        if base_path not in _rollout_path_map:
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            target = os.path.join(base_path, ts)
            os.makedirs(target, exist_ok=True)
            _rollout_path_map[base_path] = target
        target_path = _rollout_path_map[base_path]

        for checker, func in DUMPING_FUNC:
            if checker(target_path):
                p = multiprocessing.Process(target=func, args=(target_path, write_data, columns_config), daemon=False)
                p.start()
    except Exception as e:
        logger.error(f"failed to schedule dump_rollout_trajectories: {e}")


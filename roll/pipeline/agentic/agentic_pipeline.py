import json
import os.path
import random
from collections import deque
from typing import Any, Dict, List

import numpy as np
import ray
import torch
from codetiming import Timer
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from ray.util.timer import _Timer

from roll.datasets.global_dataset import GlobalDatasetManager
from roll.distributed.scheduler.rollout_scheduler import RolloutScheduler
from roll.distributed.executor.cluster import Cluster
from roll.distributed.scheduler.protocol import DataProto
from roll.models.model_providers import default_tokenizer_provider
from roll.pipeline.agentic.agentic_config import AgenticConfig, EnvManagerConfig
from roll.pipeline.agentic.utils import (dump_rollout_render, compute_discounted_returns,
                                         compute_response_level_rewards, dump_rollout_trajectories)
from roll.pipeline.base_pipeline import BasePipeline
from roll.utils.constants import RAY_NAMESPACE
from roll.utils.functionals import (
    apply_kl_penalty,
    compute_advantage,
    reduce_metrics,
    masked_mean,
    RunningMoments,
    compute_clip_fraction,
    agg_loss,
)
from roll.utils.kl_controller import get_kl_controller
from roll.utils.logging import get_logger

logger = get_logger()


class AgenticPipeline(BasePipeline):
    def __init__(self, pipeline_config: AgenticConfig):
        super().__init__(pipeline_config)
        self.pipeline_config: AgenticConfig

        self.pipeline_config.set_max_steps(max_steps=self.pipeline_config.max_steps)

        self.kl_ctrl = get_kl_controller(
            init_kl_coef=self.pipeline_config.init_kl_coef,
            target_kl=self.pipeline_config.target_kl,
            kl_horizon=self.pipeline_config.kl_horizon,
        )

        self.actor_train: Any = Cluster(
            name=self.pipeline_config.actor_train.name,
            worker_cls=self.pipeline_config.actor_train.worker_cls,
            resource_manager=self.resource_manager,
            worker_config=self.pipeline_config.actor_train,
        )
        self.actor_infer: Any = Cluster(
            name=self.pipeline_config.actor_infer.name,
            worker_cls=self.pipeline_config.actor_infer.worker_cls,
            resource_manager=self.resource_manager,
            worker_config=self.pipeline_config.actor_infer,
        )
        self.reference: Any = Cluster(
            name=self.pipeline_config.reference.name,
            worker_cls=self.pipeline_config.reference.worker_cls,
            resource_manager=self.resource_manager,
            worker_config=self.pipeline_config.reference,
        )

        download_clusters = [self.actor_train, self.actor_infer, self.reference]
        if self.pipeline_config.adv_estimator == "gae":
            self.critic: Any = Cluster(
                name=self.pipeline_config.critic.name,
                worker_cls=self.pipeline_config.critic.worker_cls,
                resource_manager=self.resource_manager,
                worker_config=self.pipeline_config.critic,
            )
            download_clusters.append(self.critic)
        
        # 新增：Cost Critic Cluster (for Safe PPO with cost constraint)
        if self.pipeline_config.enable_cost_constraint:
            if self.pipeline_config.adv_estimator != "gae":
                raise ValueError(
                    "enable_cost_constraint requires adv_estimator='gae'. "
                    "Cost critic needs GAE to compute cost advantages."
                )
            self.cost_critic: Any = Cluster(
                name=self.pipeline_config.cost_critic.name,
                worker_cls=self.pipeline_config.cost_critic.worker_cls,
                resource_manager=self.resource_manager,
                worker_config=self.pipeline_config.cost_critic,
            )
            download_clusters.append(self.cost_critic)
            logger.info(f"Cost Critic Cluster initialized: {self.cost_critic.cluster_name}")
        
        self.download_models(*download_clusters)
        self.tokenizer = default_tokenizer_provider(model_args=self.pipeline_config.actor_train.model_args)

        self.train_rollout_scheduler = ray.remote(RolloutScheduler).options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=ray.get_runtime_context().get_node_id(),
                soft=False)).remote(
            config=self.pipeline_config,
            env_manager_config=self.pipeline_config.train_env_manager,
            resource_manager=self.resource_manager,
            infer_cluster=self.actor_infer,
            mode="train",
        )
        self.val_rollout_scheduler = ray.remote(RolloutScheduler).options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=ray.get_runtime_context().get_node_id(),
                soft=False)).remote(
            config=self.pipeline_config,
            env_manager_config=self.pipeline_config.val_env_manager,
            resource_manager=self.resource_manager,
            infer_cluster=self.actor_infer,
            mode="val",
        )
        self.val_dataset_manager = GlobalDatasetManager.options(name=f"val_dataset_manager",
                                                                get_if_exists=True,
                                                                namespace=RAY_NAMESPACE).remote()
        refs: List[ray.ObjectRef] = []
        refs.extend(self.actor_train.initialize(pipeline_config=self.pipeline_config, blocking=False))
        if self.pipeline_config.adv_estimator == "gae":
            refs.extend(self.critic.initialize(pipeline_config=self.pipeline_config, blocking=False))
        
        # 新增：初始化Cost Critic
        if self.pipeline_config.enable_cost_constraint:
            refs.extend(self.cost_critic.initialize(pipeline_config=self.pipeline_config, blocking=False))
        
        ray.get(refs)

        self.actor_infer.initialize(pipeline_config=self.pipeline_config, blocking=True)

        refs.extend(self.reference.initialize(pipeline_config=self.pipeline_config, blocking=True))
        self.set_model_update_pair(
            src_cluster=self.actor_train,
            tgt_cluster=self.actor_infer,
            frequency=self.pipeline_config.actor_train.model_update_frequency,
        )

        if self.pipeline_config.adv_estimator == "gae":
            checkpoint_clusters = [self.actor_train, self.critic]
            # 新增：Cost Critic也需要checkpoint
            if self.pipeline_config.enable_cost_constraint:
                checkpoint_clusters.append(self.cost_critic)
            self.set_checkpoint_clusters(*checkpoint_clusters)
        else:
            self.set_checkpoint_clusters(self.actor_train)

        # 新增：Lagrange乘子初始化（仅在主进程）
        if self.pipeline_config.enable_cost_constraint:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.log_lambda = torch.nn.Parameter(
                torch.tensor(
                    np.log(self.pipeline_config.lambda_init),
                    device=device,
                    dtype=torch.float32
                ),
                requires_grad=True,
            )
            self.log_lambda_max = np.log(self.pipeline_config.lambda_max) if self.pipeline_config.lambda_max else None
            self.log_lambda_optimizer = torch.optim.SGD(
                [self.log_lambda],
                lr=self.pipeline_config.lambda_lr
            )
            
            # 早期验证：在初始化时测试lambda更新逻辑，提前发现问题
            if self.pipeline_config.enable_cost_constraint:
                logger.info(f"[DEBUG] Lambda优化器初始化完成，将在step {self.pipeline_config.lambda_update_delay_steps}开始更新")
                # 测试lambda loss计算是否正常
                try:
                    test_cost_mean = 1.0
                    test_cost_diff = torch.tensor(
                        test_cost_mean - self.pipeline_config.cost_limit,
                        dtype=torch.float32,
                        device=self.log_lambda.device,
                        requires_grad=False
                    )
                    test_lambda_loss = -test_cost_diff * torch.exp(self.log_lambda)
                    logger.info(f"[DEBUG] Lambda loss计算测试通过: loss={test_lambda_loss.item():.4f}")
                except Exception as e:
                    logger.error(f"[DEBUG] Lambda loss计算测试失败: {e}")
                    raise
            self.episode_costs = deque(
                maxlen=self.pipeline_config.episode_cost_window_size
            )
            logger.info(
                f"Lagrange multiplier initialized: log_lambda={self.log_lambda.item():.4f}, "
                f"lambda={torch.exp(self.log_lambda).item():.4f}"
            )

        self.running = RunningMoments()

    @torch.no_grad()
    def run(self):
        # Calculate tokens-per-second system throughput
        tps_timer = _Timer(window_size=5)
        import time
        total_start_time = time.time()

        for global_step in range(self.pipeline_config.max_steps):
            if global_step <= self.state.step:
                global_step += 1
                continue
            
            step_start_time = time.time()
            logger.info(f"\n{'='*80}")
            logger.info(f"[DEBUG] ========== Training Step {global_step}/{self.pipeline_config.max_steps} 开始 ==========")
            logger.info(f"[DEBUG] 总运行时间: {time.time() - total_start_time:.2f}秒")
            logger.info(f"{'='*80}\n")
            logger.info(f"pipeline rollout global step {global_step} start...")
            metrics = {}
            with tps_timer:
                if self.pipeline_config.adv_estimator == "gae":
                    self.critic.offload_states(blocking=True)
                    # 新增：Cost Critic也需要offload states
                    if self.pipeline_config.enable_cost_constraint:
                        self.cost_critic.offload_states(blocking=True)
                self.actor_train.offload_states(blocking=True)

                ray.get(self.train_rollout_scheduler.suspend.remote())
                if self.pipeline_config.async_generation_ratio > 0:
                    self.actor_infer.stop_server()
                model_update_metrics: Dict = self.model_update(global_step)
                metrics.update(model_update_metrics)
                if self.pipeline_config.async_generation_ratio > 0:
                    self.actor_infer.start_server(data=DataProto(meta_info={"global_step": global_step, "is_offload_states": False}))
                else:
                    self.actor_infer.start_server(data=DataProto(meta_info={"global_step": global_step, "is_offload_states": True}))

                batch: DataProto = DataProto()
                batch.meta_info = {"global_step": global_step}

                if global_step % self.pipeline_config.eval_steps == 0:
                    metrics.update(self.val(global_step=global_step))

                with Timer(name="rollout", logger=None) as rollout_timer:
                    logger.info(f"[DEBUG] [Step {global_step}] 开始rollout阶段...")
                    batch.meta_info["is_offload_states"] = True
                    logger.info(f"[DEBUG] [Step {global_step}] 调用rollout_scheduler.get_batch (batch_size={self.pipeline_config.rollout_batch_size})...")
                    batch = ray.get(self.train_rollout_scheduler.get_batch.remote(batch, self.pipeline_config.rollout_batch_size))
                    logger.info(f"[DEBUG] [Step {global_step}] rollout完成，batch形状: {batch.batch.batch_size if hasattr(batch.batch, 'batch_size') else 'N/A'}")
                    dump_rollout_trajectories(self.pipeline_config.rollout_dump_dir, global_step, batch)

                metrics["time/rollout"] = rollout_timer.last
                metrics.update(reduce_metrics(batch.meta_info.pop("metrics", {})))
                batch.meta_info["global_step"] = global_step
                if not (self.pipeline_config.async_generation_ratio > 0):
                    self.actor_infer.stop_server()

                batch = compute_discounted_returns(batch, self.pipeline_config.adv_estimator, self.pipeline_config.step_reward_gamma)

                batch = self.adjust_batch(batch, mode=self.pipeline_config.batch_adjust_mode)
                metrics.update(reduce_metrics(batch.meta_info.pop("metrics", {})))

                with Timer(name="cal_ref_log_probs", logger=None) as cal_timer:
                    ref_log_probs_refs: List[ray.ObjectRef] = self.reference.compute_log_probs(batch, blocking=False)
                    ref_log_probs = DataProto.materialize_concat(data_refs=ref_log_probs_refs)
                    ref_log_probs.rename(old_keys="log_probs", new_keys="ref_log_probs")
                    # 先处理metrics，避免union时冲突
                    ref_log_probs_metrics = ref_log_probs.meta_info.pop("metrics", {})
                    batch_metrics = batch.meta_info.pop("metrics", {})
                    batch = batch.union(ref_log_probs)
                    avg_ref_log_prob = masked_mean(batch.batch["ref_log_probs"], batch.batch["response_mask"][:, 1:])
                    metrics.update(reduce_metrics(batch_metrics))
                    metrics.update(reduce_metrics(ref_log_probs_metrics))
                    metrics.update({"critic/ref_log_prob/mean": avg_ref_log_prob.item()})
                metrics["time/ref_log_probs_values_reward"] = cal_timer.last

                with Timer(name="cal_old_log_probs_values", logger=None) as cal_old_logpb_timer:
                    # TODO: use engine log_probs as old_log_probs
                    batch.meta_info["is_offload_states"] = False
                    old_log_probs_refs: List[ray.ObjectRef] = self.actor_train.compute_log_probs(batch, blocking=False)
                    if self.pipeline_config.adv_estimator == "gae":
                        values_refs: List[ray.ObjectRef] = self.critic.compute_values(batch, blocking=False)
                        # 新增：计算Cost Critic的values（用于Cost advantage计算）
                        if self.pipeline_config.enable_cost_constraint:
                            cost_values_refs: List[ray.ObjectRef] = self.cost_critic.compute_values(batch, blocking=False)
                    old_log_probs = DataProto.materialize_concat(data_refs=old_log_probs_refs)
                    if self.pipeline_config.adv_estimator == "gae":
                        values = DataProto.materialize_concat(data_refs=values_refs)
                        # 先处理metrics，避免union时冲突
                        values_metrics = values.meta_info.pop("metrics", {})
                        batch_metrics = batch.meta_info.pop("metrics", {})
                        batch = batch.union(values)
                        metrics.update(reduce_metrics(batch_metrics))
                        metrics.update(reduce_metrics(values_metrics))
                        # 新增：收集Cost Critic的values并添加到batch
                        if self.pipeline_config.enable_cost_constraint:
                            cost_values = DataProto.materialize_concat(data_refs=cost_values_refs)
                            # 使用不同的key来存储cost values，避免与reward values混淆
                            cost_values.rename(old_keys="values", new_keys="cost_values")
                            # 先处理metrics，避免union时冲突
                            cost_values_metrics = cost_values.meta_info.pop("metrics", {})
                            batch_metrics = batch.meta_info.pop("metrics", {})
                            batch = batch.union(cost_values)
                            # 保存旧的cost_values用于PPO clipping
                            batch.batch["old_cost_values"] = batch.batch["cost_values"].clone()
                            metrics.update(reduce_metrics(batch_metrics))
                            metrics.update(reduce_metrics(cost_values_metrics))
                    batch.batch["old_log_probs"] = old_log_probs.batch["log_probs"]
                    avg_old_log_prob = masked_mean(batch.batch["old_log_probs"], batch.batch["response_mask"][:, 1:])
                    metrics.update({"critic/old_log_prob/mean": avg_old_log_prob.item()})

                    agg_entropy = agg_loss(
                        loss_mat=old_log_probs.batch["entropy"],
                        loss_mask=batch.batch["response_mask"][:, 1:],
                        loss_agg_mode="token-mean",
                    )
                    metrics.update({"critic/entropy/mean": agg_entropy.item()})

                    metrics.update(reduce_metrics(old_log_probs.meta_info.pop("metrics", {})))
                metrics["time/old_log_probs_values"] = cal_old_logpb_timer.last

                with Timer(name="adv", logger=None) as timer:
                    # Rewards need to be processed after grouping
                    # We can group by tag(env_type)/traj_group_id(group)/batch(rollout_batch)... to compute rewards / advantages
                    # The compute_response_level_rewards function injects a response_level_rewards key into batch.batch.
                    batch = compute_response_level_rewards(batch=batch, pipeline_config=self.pipeline_config)
                    
                    # 记录 response_level_rewards 的统计信息（用于监控训练）
                    if "response_level_rewards" in batch.batch:
                        resp_rewards = batch.batch["response_level_rewards"]
                        logger.info(
                            f"[Training Step {global_step}] Response Level Rewards Stats: "
                            f"mean={resp_rewards.mean().item():.4f}, "
                            f"min={resp_rewards.min().item():.4f}, "
                            f"max={resp_rewards.max().item():.4f}, "
                            f"std={resp_rewards.std().item():.4f}, "
                            f"shape={resp_rewards.shape}"
                        )
                    
                    metrics.update(reduce_metrics(batch.meta_info.pop("metrics", {})))

                    if self.pipeline_config.reward_clip:
                        reward_clip_frac = compute_clip_fraction(
                            values=batch.batch["response_level_rewards"],
                            clip_max=self.pipeline_config.reward_clip,
                            clip_min=-self.pipeline_config.reward_clip,
                        )
                        metrics["critic/reward_clip_frac"] = reward_clip_frac
                        batch.batch["response_level_rewards"] = torch.clamp(
                            batch.batch["response_level_rewards"],
                            min=-self.pipeline_config.reward_clip,
                            max=self.pipeline_config.reward_clip,
                        )

                    # Expand compute_response_level_rewards and add kl_penalty.
                    batch, kl_metrics = apply_kl_penalty(data=batch, kl_ctrl=self.kl_ctrl, kl_penalty=self.pipeline_config.kl_penalty)

                    # Is the advantage calculated globally across the batch, or within each group?
                    # 计算Reward侧的advantage（原有逻辑）
                    logger.info(f"[DEBUG] [Step {global_step}] 开始计算Reward侧的advantage...")
                    batch = compute_advantage(
                        data=batch,
                        gamma=self.pipeline_config.gamma,
                        lambd=self.pipeline_config.lambd,
                        adv_estimator=self.pipeline_config.adv_estimator,
                        advantage_clip=self.pipeline_config.advantage_clip,
                        whiten_advantages=self.pipeline_config.whiten_advantages,
                        whiten_rewards=self.pipeline_config.whiten_rewards,
                        cost_mode=False,  # Reward模式
                    )
                    metrics.update(reduce_metrics(batch.meta_info.pop("metrics", {})))
                    logger.info(f"[DEBUG] [Step {global_step}] Reward侧advantage计算完成")
                    
                    # 新增：计算Cost侧的advantage（仅当启用cost约束时）
                    if self.pipeline_config.enable_cost_constraint:
                        logger.info(f"[DEBUG] [Step {global_step}] 开始计算Cost侧的advantage...")
                        batch = compute_advantage(
                            data=batch,
                            gamma=self.pipeline_config.gamma,
                            lambd=self.pipeline_config.lambd,
                            adv_estimator=self.pipeline_config.adv_estimator,
                            advantage_clip=self.pipeline_config.advantage_clip,
                            whiten_advantages=self.pipeline_config.whiten_advantages,
                            whiten_rewards=self.pipeline_config.whiten_rewards,
                            cost_mode=True,  # Cost模式
                        )
                        metrics.update(reduce_metrics(batch.meta_info.pop("metrics", {})))
                        logger.info(f"[DEBUG] [Step {global_step}] Cost侧advantage计算完成")

                metrics.update(kl_metrics)
                metrics["time/adv"] = timer.last

                # Lambda更新逻辑（在critic训练之前，以便lambda可用于后续的actor训练）
                if self.pipeline_config.enable_cost_constraint:
                    # 收集episode costs并更新lambda
                    if "episode_costs" in batch.non_tensor_batch:
                        episode_costs = batch.non_tensor_batch["episode_costs"]
                        # 转换为numpy array并计算平均值
                        if isinstance(episode_costs, torch.Tensor):
                            avg_cost = episode_costs.mean().item()
                        else:
                            # 处理numpy array或list
                            costs_flat = []
                            for cost in episode_costs:
                                if isinstance(cost, (list, np.ndarray)):
                                    costs_flat.extend(cost if isinstance(cost, list) else cost.tolist())
                                else:
                                    costs_flat.append(float(cost))
                            avg_cost = np.mean(costs_flat) if costs_flat else 0.0
                        
                        self.episode_costs.append(avg_cost)
                        
                        # 计算移动平均
                        if len(self.episode_costs) > 0:
                            episode_cost_mean = np.mean(list(self.episode_costs))
                            
                            # 更新lambda（仅在达到延迟步数后）
                            if global_step >= self.pipeline_config.lambda_update_delay_steps:
                                try:
                                    # 【关键修复】整个run方法被@torch.no_grad()装饰，需要使用torch.enable_grad()临时启用梯度追踪
                                    with torch.enable_grad():
                                        # 确保self.log_lambda需要梯度
                                        if not self.log_lambda.requires_grad:
                                            logger.warning(f"[DEBUG] [Step {global_step}] log_lambda不需要梯度，强制设置requires_grad=True")
                                            self.log_lambda.requires_grad_(True)
                                        
                                        # 验证log_lambda的状态
                                        exp_log_lambda = torch.exp(self.log_lambda)
                                        if not exp_log_lambda.requires_grad:
                                            logger.error(f"[DEBUG] [Step {global_step}] torch.exp(log_lambda)不需要梯度！")
                                            logger.error(f"[DEBUG] [Step {global_step}] log_lambda.requires_grad={self.log_lambda.requires_grad}")
                                            logger.error(f"[DEBUG] [Step {global_step}] log_lambda.is_leaf={self.log_lambda.is_leaf}")
                                            logger.error(f"[DEBUG] [Step {global_step}] log_lambda.grad_fn={self.log_lambda.grad_fn}")
                                            logger.error(f"[DEBUG] [Step {global_step}] 是否在torch.enable_grad()上下文中: {torch.is_grad_enabled()}")
                                            raise RuntimeError(f"[Step {global_step}] torch.exp(log_lambda)不需要梯度，无法计算lambda_loss")
                                        
                                        # 计算cost_diff（Python float，不会影响梯度计算）
                                        cost_diff_value = float(episode_cost_mean - self.pipeline_config.cost_limit)
                                        
                                        # 直接使用self.log_lambda计算lambda_loss，确保计算图正确
                                        # cost_diff_value是Python float，不会影响梯度计算
                                        lambda_loss = -cost_diff_value * exp_log_lambda
                                        
                                        # 验证lambda_loss是否需要梯度
                                        if not lambda_loss.requires_grad:
                                            logger.error(f"[DEBUG] [Step {global_step}] lambda_loss不需要梯度！")
                                            logger.error(f"[DEBUG] [Step {global_step}] log_lambda.requires_grad={self.log_lambda.requires_grad}")
                                            logger.error(f"[DEBUG] [Step {global_step}] log_lambda.is_leaf={self.log_lambda.is_leaf}")
                                            logger.error(f"[DEBUG] [Step {global_step}] log_lambda.grad_fn={self.log_lambda.grad_fn}")
                                            logger.error(f"[DEBUG] [Step {global_step}] cost_diff_value={cost_diff_value}")
                                            logger.error(f"[DEBUG] [Step {global_step}] torch.exp(self.log_lambda).requires_grad={torch.exp(self.log_lambda).requires_grad}")
                                            logger.error(f"[DEBUG] [Step {global_step}] 是否在torch.enable_grad()上下文中: {torch.is_grad_enabled()}")
                                            raise RuntimeError(f"[Step {global_step}] lambda_loss不需要梯度，无法进行反向传播")
                                        
                                        self.log_lambda_optimizer.zero_grad()
                                        lambda_loss.backward()
                                        
                                        # 检查梯度是否存在
                                        if self.log_lambda.grad is None:
                                            logger.error(f"[DEBUG] [Step {global_step}] log_lambda的梯度为None")
                                            logger.error(f"[DEBUG] [Step {global_step}] lambda_loss.requires_grad={lambda_loss.requires_grad}")
                                            logger.error(f"[DEBUG] [Step {global_step}] 是否在torch.enable_grad()上下文中: {torch.is_grad_enabled()}")
                                            raise RuntimeError(f"[Step {global_step}] log_lambda的梯度为None，无法更新")
                                        
                                        self.log_lambda_optimizer.step()
                                        
                                        logger.info(f"[DEBUG] [Step {global_step}] Lambda更新成功: log_lambda={self.log_lambda.item():.6f}, grad={self.log_lambda.grad.item() if self.log_lambda.grad is not None else None}")
                                except Exception as e:
                                    logger.error(f"[Step {global_step}] Lambda更新失败: {e}")
                                    logger.error(f"[Step {global_step}] episode_cost_mean={episode_cost_mean}, cost_limit={self.pipeline_config.cost_limit}")
                                    logger.error(f"[Step {global_step}] log_lambda.requires_grad={self.log_lambda.requires_grad}")
                                    logger.error(f"[Step {global_step}] log_lambda.device={self.log_lambda.device}")
                                    logger.error(f"[Step {global_step}] 是否在torch.enable_grad()上下文中: {torch.is_grad_enabled()}")
                                    raise
                                
                                # Clamp lambda到最大值
                                if self.log_lambda_max is not None:
                                    with torch.no_grad():
                                        self.log_lambda.clamp_(max=self.log_lambda_max)
                                
                                metrics["train/lambda"] = torch.exp(self.log_lambda).item()
                                metrics["train/log_lambda"] = self.log_lambda.item()
                                metrics["train/episode_cost"] = episode_cost_mean
                                metrics["train/episode_cost_current"] = avg_cost
                                logger.info(
                                    f"[Step {global_step}] Lambda updated: λ={torch.exp(self.log_lambda).item():.4f}, "
                                    f"episode_cost_mean={episode_cost_mean:.4f}, cost_limit={self.pipeline_config.cost_limit}"
                                )
                            
                            # 将lambda传递到batch中，供worker使用
                            batch.meta_info["log_lambda"] = self.log_lambda.detach().cpu().item()
                
                if self.pipeline_config.adv_estimator == "gae":
                    logger.info(f"[DEBUG] [Step {global_step}] 开始训练Critic模型...")
                    # 先训练critic，使用blocking=False保持异步，但立即等待结果
                    critic_train_metrics_refs: List[ray.ObjectRef] = self.critic.train_step(batch, blocking=False)
                    # 等待critic训练完成
                    critic_train_metrics = DataProto.materialize_concat(data_refs=critic_train_metrics_refs)
                    logger.info(f"[DEBUG] [Step {global_step}] Critic模型训练完成，开始offload...")
                    # 训练完成后立即offload以释放GPU内存
                    self.critic.offload_states(blocking=True)
                    # 清理GPU缓存以确保内存释放
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    logger.info(f"[DEBUG] [Step {global_step}] Critic模型已offload，GPU缓存已清理")
                    
                    # 新增：Cost Critic训练（在critic offload之后）
                    if self.pipeline_config.enable_cost_constraint:
                        logger.info(f"[DEBUG] [Step {global_step}] 开始训练Cost Critic模型...")
                        cost_critic_train_metrics_refs: List[ray.ObjectRef] = self.cost_critic.train_step(batch, blocking=False)
                        # 等待cost_critic训练完成
                        cost_critic_train_metrics = DataProto.materialize_concat(data_refs=cost_critic_train_metrics_refs)
                        logger.info(f"[DEBUG] [Step {global_step}] Cost Critic模型训练完成，开始offload...")
                        # 训练完成后立即offload以释放GPU内存
                        self.cost_critic.offload_states(blocking=True)
                        # 清理GPU缓存
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        logger.info(f"[DEBUG] [Step {global_step}] Cost Critic模型已offload，GPU缓存已清理")

                # implement critic warmup
                if self.pipeline_config.critic_warmup <= global_step:
                    # update actor
                    # optional debug dump: write important batch tensors for offline inspection
                    if getattr(self.pipeline_config, "debug_dump_batch", False):
                        try:
                            os.makedirs(os.path.join(self.pipeline_config.output_dir, "debug"), exist_ok=True)
                            dump_path = os.path.join(self.pipeline_config.output_dir, "debug", f"batch_debug_{global_step}.pt")
                            dump_obj = {
                                "response_level_rewards": batch.batch.get("response_level_rewards"),
                                "token_level_rewards": batch.batch.get("token_level_rewards"),
                                "response_mask": batch.batch.get("response_mask"),
                                "final_response_mask": batch.batch.get("final_response_mask"),
                                "advantages": batch.batch.get("advantages"),
                                "returns": batch.batch.get("returns"),
                                "old_log_probs": batch.batch.get("old_log_probs"),
                                "ref_log_probs": batch.batch.get("ref_log_probs"),
                            }
                            torch.save(dump_obj, dump_path)
                            logger.info(f"wrote actor batch debug to {dump_path}")
                        except Exception:
                            logger.exception("failed to write actor batch debug file")

                    logger.info(f"[Training Step {global_step}] Updating Actor Model (PPO training)...")
                    # Actor训练使用blocking=False保持异步，但立即等待结果
                    actor_train_metrics_refs = self.actor_train.train_step(batch, blocking=False)
                    # 等待actor训练完成
                    actor_train_metrics: DataProto = DataProto.materialize_concat(data_refs=actor_train_metrics_refs)
                    metrics.update(reduce_metrics(actor_train_metrics.meta_info.pop("metrics", {})))
                    logger.info(f"[Training Step {global_step}] Actor Model Updated Successfully")
                    logger.info(f"[DEBUG] [Step {global_step}] Actor模型训练完成，开始offload...")
                    # Actor训练完成后也offload
                    self.actor_train.offload_states(blocking=True)
                    # 清理GPU缓存
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    logger.info(f"[DEBUG] [Step {global_step}] Actor模型已offload，GPU缓存已清理")

                if self.pipeline_config.adv_estimator == "gae":
                    # critic_train_metrics已经在训练时materialize了，直接使用
                    metrics.update(reduce_metrics(critic_train_metrics.meta_info.pop("metrics", {})))
                    # 新增：收集Cost Critic训练metrics
                    if self.pipeline_config.enable_cost_constraint:
                        # cost_critic_train_metrics已经在训练时materialize了，直接使用
                        # 重命名metrics以避免与reward critic混淆
                        cost_metrics = cost_critic_train_metrics.meta_info.pop("metrics", {})
                        cost_metrics_renamed = {f"cost_critic/{k}": v for k, v in cost_metrics.items()}
                        metrics.update(reduce_metrics(cost_metrics_renamed))
                tps_timer.push_units_processed(n=torch.sum(batch.batch["attention_mask"]).detach().item())

            data_metrics = compute_data_metrics(batch=batch)
            metrics.update(data_metrics)
            metrics["system/tps"] = tps_timer.mean_throughput
            metrics["system/samples"] = (global_step + 1) * self.pipeline_config.rollout_batch_size

            # do ckpt
            self.state.step = global_step
            self.state.log_history.append(metrics)

            self.do_checkpoint(global_step=global_step)

            self.tracker.log(values=metrics, step=global_step)

            step_time = time.time() - step_start_time
            logger.info(f"\n{'='*80}")
            logger.info(f"[DEBUG] ========== Training Step {global_step}/{self.pipeline_config.max_steps} 完成 ==========")
            logger.info(f"[DEBUG] 本步骤耗时: {step_time:.2f}秒 ({step_time/60:.2f}分钟)")
            logger.info(f"[DEBUG] 总运行时间: {time.time() - total_start_time:.2f}秒 ({(time.time() - total_start_time)/60:.2f}分钟)")
            if global_step < self.pipeline_config.max_steps - 1:
                remaining_steps = self.pipeline_config.max_steps - global_step - 1
                avg_time_per_step = (time.time() - total_start_time) / (global_step + 1)
                estimated_remaining = avg_time_per_step * remaining_steps
                logger.info(f"[DEBUG] 预计剩余时间: {estimated_remaining:.2f}秒 ({estimated_remaining/60:.2f}分钟)")
            logger.info(f"[DEBUG] 关键指标: rollout={metrics.get('time/rollout', 'N/A')}, adv={metrics.get('time/adv', 'N/A')}")
            logger.info(f"{'='*80}\n")

            if global_step % self.pipeline_config.logging_steps == 0:
                if int(os.environ.get("RAY_PROFILING", "0")):
                    timeline_dir = os.path.join(self.pipeline_config.profiler_output_dir, "timeline")
                    os.makedirs(timeline_dir, exist_ok=True)
                    ray.timeline(
                        filename=os.path.join(timeline_dir, f"timeline-step-{global_step}.json"),
                    )

                log_res = []
                batch_grouped = batch.group_by(keys="traj_id")
                for group_name, group_batch in batch_grouped.items():
                    prompt_mask = group_batch.batch["prompt_mask"]
                    non_prompt_mask = torch.logical_not(group_batch.batch["prompt_mask"]) * group_batch.batch["attention_mask"]
                    input_ids = group_batch.batch["input_ids"]
                    prompt_ids_list = [input_ids[i][mask.bool()] for i, mask in enumerate(prompt_mask)]
                    response_ids_list = [input_ids[i][mask.bool()] for i, mask in enumerate(non_prompt_mask)]
                    # Avoid logging tokenizer special tokens (e.g. <|im_start|>)
                    prompts = self.tokenizer.batch_decode(prompt_ids_list, skip_special_tokens=True)
                    responses = self.tokenizer.batch_decode(response_ids_list, skip_special_tokens=True)
                    episode_scores = group_batch.non_tensor_batch["episode_scores"].tolist()
                    step_scores = group_batch.non_tensor_batch["step_scores"].tolist()
                    if not isinstance(step_scores[0], float):
                        step_scores = [t.tolist() for t in step_scores]

                    log_item = []
                    for prompt, response, episode_score, step_score in zip(
                            prompts, responses, episode_scores, step_scores
                    ):
                        log_item.append(
                            {
                                "prompt": prompt,
                                "response": response,
                                "episode_score": episode_score,
                                "step_score": step_score,
                            }
                        )
                    log_res.append(log_item)
                    if len(log_res) >= 10:
                        break
                logger.info(json.dumps(log_res, ensure_ascii=False))
                logger.info(json.dumps(metrics, ensure_ascii=False))

            logger.info(f"pipeline step {global_step} finished")
            global_step += 1
            logger.info(f"epoch {global_step} finished")

        ray.get([
            self.train_rollout_scheduler.shutdown.remote(),
            self.val_rollout_scheduler.shutdown.remote(),
        ])
        logger.info("pipeline complete!")

    def val(self, global_step):
        batch = DataProto()
        metrics = {}
        batch.meta_info["is_offload_states"] = False
        batch.meta_info["global_step"] = global_step
        ray.get(self.val_dataset_manager.reset.remote())
        eval_batch = ray.get(self.val_rollout_scheduler.get_batch.remote(batch, self.pipeline_config.val_batch_size))

        dump_rollout_trajectories(self.pipeline_config.rollout_dump_dir, global_step, eval_batch)
        eval_metrics = reduce_metrics(eval_batch.meta_info.get("metrics", {}))
        eval_score = get_episode_scores(eval_batch)
        eval_metrics["score/mean"] = torch.mean(eval_score).detach().item()
        eval_metrics["score/max"] = torch.max(eval_score).detach().item()
        eval_metrics["score/min"] = torch.min(eval_score).detach().item()

        batch_grouped = eval_batch.group_by(keys="tags")
        for group_name, group_batch in batch_grouped.items():
            traj_group_scores = []
            batch_traj_grouped = group_batch.group_by(keys="traj_group_id")
            for batch_traj_group_name, batch_traj_group in batch_traj_grouped.items():
                traj_group_score = get_episode_scores(batch_traj_group)
                traj_group_scores.append(traj_group_score.mean().item())
            eval_score = torch.tensor(traj_group_scores, dtype=torch.float)
            eval_metrics[f"{group_name}/score/mean"] = torch.mean(eval_score).detach().item()
            eval_metrics[f"{group_name}/score/max"] = torch.max(eval_score).detach().item()
            eval_metrics[f"{group_name}/score/min"] = torch.min(eval_score).detach().item()

        metrics.update({f"val/{k}": v for k, v in eval_metrics.items()})
        logger.info(f"val_batch_size: {len(eval_batch)}")
        logger.info(f"val metrics: {metrics}")

        return metrics

    def adjust_batch(self, data: DataProto, mode="copy") -> DataProto:
        """
        ref: https://github.com/langfengQ/verl-agent/blob/e03bd502667c45172e8c093cc506db8438ae8ab5/agent_system/multi_turn_rollout/utils.py#L86
        """
        actor_train_train_bsz = self.pipeline_config.actor_train.training_args.per_device_train_batch_size * self.pipeline_config.actor_train.training_args.gradient_accumulation_steps * self.actor_train.dp_size
        actor_train_infer_bsz = self.pipeline_config.actor_train.infer_batch_size * self.actor_train.dp_size
        ref_infer_bsz = self.pipeline_config.reference.infer_batch_size * self.reference.dp_size
        critic_train_bsz = 1
        critic_infer_bsz = 1
        if self.pipeline_config.adv_estimator == "gae":
            critic_train_bsz = self.pipeline_config.critic.training_args.per_device_train_batch_size * self.pipeline_config.critic.training_args.gradient_accumulation_steps * self.critic.dp_size
            critic_infer_bsz = self.pipeline_config.critic.infer_batch_size * self.critic.dp_size

        size_divide = np.lcm.reduce(np.array([actor_train_train_bsz, actor_train_infer_bsz, ref_infer_bsz, critic_infer_bsz, critic_train_bsz])).item()
        batch_size = data.batch.batch_size[0]
        threshold = batch_size % size_divide

        if threshold == 0:
            return data

        if mode == "auto":
            if threshold >= 0.5 * batch_size or  batch_size // size_divide == 0:
                mode = "copy"
            else:
                mode = "delete"
        elif mode == "random_sample":
            if batch_size < size_divide:
                mode = "copy"

        metrics = data.meta_info.get("metrics", {})
        metrics["system/batch_add_count"] = 0
        metrics["system/batch_remove_count"] = 0
        if mode == "delete":
            remove_indices = np.random.choice(batch_size, threshold, replace=False)
            remove_indices = np.sort(remove_indices)
            keep_mask = np.ones(batch_size, dtype=bool)
            keep_mask[remove_indices] = False
            keep_mask_tensor = torch.tensor(keep_mask, dtype=torch.bool, device=data.batch['input_ids'].device)
            tensor_data = data.batch[keep_mask_tensor]
            non_tensor_data = {key: val[keep_mask] for key, val in data.non_tensor_batch.items()}
            adjusted_batch = DataProto(batch=tensor_data, non_tensor_batch=non_tensor_data, meta_info=data.meta_info)
            metrics["system/batch_remove_count"] = len(remove_indices)
        elif mode == "copy":
            to_add = size_divide - threshold
            dup_indices = np.random.choice(batch_size, to_add, replace=True) if to_add > batch_size else np.random.choice(batch_size, to_add, replace=False)
            dup_proto = data.select_idxs(dup_indices)
            # TODO: set dup_proto response_mask to 0
            adjusted_batch = DataProto.concat([data, dup_proto])
            metrics["system/batch_add_count"] = to_add
        elif mode == "random_sample":
            select_indices = np.random.choice(batch_size, size_divide, replace=False)
            select_indices = np.sort(select_indices)
            adjusted_batch = data.select_idxs(select_indices)
            metrics["system/batch_remove_count"] = batch_size - size_divide
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        adjusted_batch.meta_info["metrics"] = metrics

        return adjusted_batch

def get_episode_scores(batch: DataProto) -> torch.Tensor:
    batch_group_by_traj: Dict[str, DataProto] = batch.group_by(keys="traj_id")
    scores = []
    for traj_id,  traj_batch in batch_group_by_traj.items():
        episode_scores = traj_batch.non_tensor_batch["episode_scores"][0]
        scores.append(episode_scores)
    return torch.tensor(scores, dtype=torch.float32)

def get_traj_rollout_time(batch: DataProto) -> torch.Tensor:
    batch_group_by_traj: Dict[str, DataProto] = batch.group_by(keys="traj_id")
    scores = []
    for traj_id,  traj_batch in batch_group_by_traj.items():
        episode_scores = traj_batch.non_tensor_batch["traj_rollout_time"][0]
        scores.append(episode_scores)
    return torch.tensor(scores, dtype=torch.float32)

def get_traj_env_time(batch: DataProto) -> torch.Tensor:
    batch_group_by_traj: Dict[str, DataProto] = batch.group_by(keys="traj_id")
    scores = []
    for traj_id,  traj_batch in batch_group_by_traj.items():
        episode_scores = traj_batch.non_tensor_batch["traj_env_time"][0]
        scores.append(episode_scores)
    return torch.tensor(scores, dtype=torch.float32)

def compute_data_metrics(batch):
    # token_level_scores are per-token scores assigned by the reward model, possibly after normalization/clipping
    # score denotes the raw environment reward
    episode_scores = get_episode_scores(batch)
    try:
        traj_rollout_times = get_traj_rollout_time(batch)
        traj_env_times = get_traj_env_time(batch)
    except Exception as e:
        traj_rollout_times = torch.zeros(batch.batch.batch_size[0], dtype=torch.float32)
        traj_env_times = torch.zeros(batch.batch.batch_size[0], dtype=torch.float32)

    sequence_reward = batch.batch["token_level_rewards"].sum(-1)
    advantages = batch.batch["advantages"]
    # fix: https://github.com/volcengine/verl/pull/60
    response_mask = batch.batch["response_mask"][:, 1:].bool()
    prompt_mask = batch.batch["prompt_mask"].bool() # 首轮 prompt length
    prompt_lengths = prompt_mask.sum(-1).float()  # (batch_size,)
    response_length = response_mask.sum(-1).float()  # (batch_size,)
    returns = batch.batch["returns"]
    non_prompt_mask = (torch.logical_not(batch.batch["prompt_mask"]) * batch.batch["attention_mask"]).float().sum(-1)

    # 从 batch 中提取 traj_rollout_time 相关指标
    # traj_rollout_times = []
    metrics = {
        # score, sequence_score from env
        "critic/score/mean": torch.mean(episode_scores).detach().item(),
        "critic/score/max": torch.max(episode_scores).detach().item(),
        "critic/score/min": torch.min(episode_scores).detach().item(),
        # reward
        "critic/rewards/mean": torch.mean(sequence_reward).detach().item(),
        "critic/rewards/max": torch.max(sequence_reward).detach().item(),
        "critic/rewards/min": torch.min(sequence_reward).detach().item(),
        # adv
        "critic/advantages/mean": masked_mean(advantages, response_mask).detach().item(),
        "critic/advantages/max": torch.max(advantages[response_mask]).detach().item() if response_mask.sum() > 0 else 0.0,
        "critic/advantages/min": torch.min(advantages[response_mask]).detach().item() if response_mask.sum() > 0 else 0.0,
        # returns
        "critic/returns/mean": masked_mean(returns, response_mask).detach().item(),
        "critic/returns/max": torch.max(returns[response_mask]).detach().item() if response_mask.sum() > 0 else 0.0,
        "critic/returns/min": torch.min(returns[response_mask]).detach().item() if response_mask.sum() > 0 else 0.0,
        # response length
        "tokens/response_length/mean": torch.mean(response_length).detach().item(),
        "tokens/response_length/max": torch.max(response_length).detach().item(),
        "tokens/response_length/min": torch.min(response_length).detach().item(),
        # prompt length
        "tokens/prompt_length/mean": torch.mean(prompt_lengths).detach().item(),
        "tokens/prompt_length/max": torch.max(prompt_lengths).detach().item(),
        "tokens/prompt_length/min": torch.min(prompt_lengths).detach().item(),
        # prompt length(sys_obs)
        # "tokens/prompt_length_sys_obs/mean": torch.mean(prompt_lengths_sys_obs).detach().item(),
        # "tokens/prompt_length_sys_obs/max": torch.max(prompt_lengths_sys_obs).detach().item(),
        # "tokens/prompt_length_sys_obs/min": torch.min(prompt_lengths_sys_obs).detach().item(),
        # non-prompt length
        "tokens/non_prompt_length/mean": torch.mean(non_prompt_mask).detach().item(),
        "tokens/non_prompt_length/max": torch.max(non_prompt_mask).detach().item(),
        "tokens/non_prompt_length/min": torch.min(non_prompt_mask).detach().item(),

        # # traj_rollout_time
        "env/traj_rollout_time/mean": torch.mean(traj_rollout_times).detach().item() if traj_rollout_times.numel() > 0 else 0.0,
        "env/traj_rollout_time/max": torch.max(traj_rollout_times).detach().item() if traj_rollout_times.numel() > 0 else 0.0,
        "env/traj_rollout_time/min": torch.min(traj_rollout_times).detach().item() if traj_rollout_times.numel() > 0 else 0.0,

        # traj_env_times
        "env/traj_env_time/mean": torch.mean(traj_env_times).detach().item() if traj_env_times.numel() > 0 else 0.0,
        "env/traj_env_time/max": torch.max(traj_env_times).detach().item() if traj_env_times.numel() > 0 else 0.0,
        "env/traj_env_time/min": torch.min(traj_env_times).detach().item() if traj_env_times.numel() > 0 else 0.0,

    }

    if "values" in batch.batch.keys():
        values = batch.batch["values"]
        # values
        metrics.update(
            {
                "critic/values/mean": masked_mean(values, response_mask).detach().item(),
                "critic/values/max": torch.max(values[response_mask]).detach().item() if response_mask.sum() > 0 else 0.0,
                "critic/values/min": torch.min(values[response_mask]).detach().item() if response_mask.sum() > 0 else 0.0,
            }
        )
    if "episode_rewards_norm" in batch.batch.keys():
        episode_rewards_norm = batch.batch["episode_rewards_norm"]
        step_rewards_norm = batch.batch["step_rewards_norm"]
        metrics.update({
            "critic/episode_rewards_norm/mean": episode_rewards_norm.mean().detach().item(),
            "critic/episode_rewards_norm/max": episode_rewards_norm.max().detach().item(),
            "critic/episode_rewards_norm/min": episode_rewards_norm.min().detach().item(),
            "critic/step_rewards_norm/mean": step_rewards_norm.mean().detach().item(),
            "critic/step_rewards_norm/max": step_rewards_norm.max().detach().item(),
            "critic/step_rewards_norm/min": step_rewards_norm.min().detach().item(),
        })
    
    # 新增：Cost相关的指标（仅在启用cost约束时）
    if "cost_values" in batch.batch.keys():
        cost_values = batch.batch["cost_values"]
        metrics.update({
            "critic/cost_values/mean": masked_mean(cost_values, response_mask).detach().item(),
            "critic/cost_values/max": torch.max(cost_values[response_mask]).detach().item() if response_mask.sum() > 0 else 0.0,
            "critic/cost_values/min": torch.min(cost_values[response_mask]).detach().item() if response_mask.sum() > 0 else 0.0,
        })
    
    if "cost_advantages" in batch.batch.keys():
        cost_advantages = batch.batch["cost_advantages"]
        metrics.update({
            "critic/cost_advantages/mean": masked_mean(cost_advantages, response_mask).detach().item(),
            "critic/cost_advantages/max": torch.max(cost_advantages[response_mask]).detach().item() if response_mask.sum() > 0 else 0.0,
            "critic/cost_advantages/min": torch.min(cost_advantages[response_mask]).detach().item() if response_mask.sum() > 0 else 0.0,
        })
    
    if "cost_returns" in batch.batch.keys():
        cost_returns = batch.batch["cost_returns"]
        metrics.update({
            "critic/cost_returns/mean": masked_mean(cost_returns, response_mask).detach().item(),
            "critic/cost_returns/max": torch.max(cost_returns[response_mask]).detach().item() if response_mask.sum() > 0 else 0.0,
            "critic/cost_returns/min": torch.min(cost_returns[response_mask]).detach().item() if response_mask.sum() > 0 else 0.0,
        })
    
    return metrics

class GroupFilter:
    """
    User defined group filter.
    """
    def __init__(self, config: AgenticConfig, env_manager_config: EnvManagerConfig, mode: str):
        pass

    def filter(self, group_id: int, episode_id: int, group: list[DataProto]):
        """
        return True to filter out this group
        """
        return False

import os
import sys
import re
import json
import traceback
from datetime import datetime
from pathlib import Path
from contextlib import nullcontext
from threading import Lock
from typing import Optional, List, Dict, Any

# --- 第三方库 ---
import numpy as np
import torch
import ray
import gem
from omegaconf import DictConfig
from tensordict import TensorDict
from transformers import PreTrainedTokenizer

# --- 项目内部模块 ---
from roll.pipeline.agentic.llm_proxy import create_llm_proxy, BaseLLMProxy
from roll.pipeline.agentic.env_manager.base_env_manager import RolloutCache, BaseEnvManager
from roll.utils.env_action_limiter import get_global_limiter
from roll.distributed.scheduler.rollout_scheduler import GroupQueueManager
from roll.pipeline.agentic.env_manager.token_mask_utils import custom_apply_chat_template, compute_conversation_end_token_id
from roll.pipeline.agentic.tools.tool_env_wrapper import tool_wrapper
from roll.distributed.scheduler.generate_scheduler import RequestScheduler
from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.agentic.agentic_config import EnvManagerConfig, AgenticConfig
from roll.utils.constants import GenerateStopReason
from roll.utils.functionals import pad_to_length, aggregate_metrics
from roll.utils.logging import get_logger
from roll.utils.str_utils import contains_renderable_field

# --- 动态导入 OSWorld 相关评估模块 ---
OSWORLD_PATH = "/data/share/projects/quanshu/OSWorld-dev"
if OSWORLD_PATH not in sys.path:
    sys.path.insert(0, OSWORLD_PATH)

run_step_pytest = None
RuleBasedEvaluator = None
llm_judge_task_completion = None

try:
    from desktop_env.evaluators.metrics.step_pytest_runner import run_step_pytest
except ImportError as e:
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"Failed to import run_step_pytest from desktop_env.evaluators.metrics.step_pytest_runner: {e}. Pytest evaluation will be disabled.")

try:
    from desktop_env.evaluators.rule_based import RuleBasedEvaluator
except ImportError as e:
    import logging
    logger = logging.getLogger(__name__)
    logger.debug(f"Failed to import RuleBasedEvaluator: {e}")

# 导入额外观测收集器（封装了所有 OSWorld-dev 相关逻辑）
from roll.pipeline.agentic.env_manager.extra_observation_collector import (
    ExtraObservationCollector,
    format_extra_observations
)

# Import llm_judge function from OSWorld-dev (same as lib_run_single.py)
llm_judge_task_completion = None
try:
    import types
    # Try normal import first
    try:
        from lib_run_single import llm_judge_task_completion
    except ImportError:
        # If import fails due to missing wrapt_timeout_decorator, create a mock module
        mock_wrapt = types.ModuleType('wrapt_timeout_decorator')
        sys.modules['wrapt_timeout_decorator'] = mock_wrapt
        from lib_run_single import llm_judge_task_completion
except ImportError as e:
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"Failed to import llm_judge_task_completion from lib_run_single: {e}. LLM judge will be disabled.")

# Import llm_judge function from OSWorld-dev (same as lib_run_single.py)
try:
    import types
    # Try normal import first
    try:
        from lib_run_single import llm_judge_task_completion
    except ImportError:
        # If import fails due to missing wrapt_timeout_decorator, create a mock module
        mock_wrapt = types.ModuleType('wrapt_timeout_decorator')
        sys.modules['wrapt_timeout_decorator'] = mock_wrapt
        from lib_run_single import llm_judge_task_completion
except ImportError as e:
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"Failed to import llm_judge_task_completion from lib_run_single: {e}. LLM judge will be disabled.")


class TrajEnvManager(BaseEnvManager):
    def __init__(self,
                 worker_config: EnvManagerConfig,
                 pipeline_config: AgenticConfig,
                 env_config: DictConfig,
                 tokenizer: PreTrainedTokenizer,
                 generate_scheduler,
                 output_queue: GroupQueueManager,
                 thread_lock: Lock,
                 mode='train',
                 *args, **kwargs):
        super().__init__()
        self.logger = get_logger()
        self.worker_config = worker_config
        self.pipeline_config = pipeline_config
        self.env_config = env_config
        self.tokenizer = tokenizer
        self.output_queue = output_queue
        self.mode = mode
        self.generate_scheduler = generate_scheduler

        # 状态管理
        self.rollout_cache: Optional[RolloutCache] = None
        self.group_seed = None
        self.episode_id = None
        self.running = False
        
        # pytest 结果目录缓存（每个任务/episode 一个目录）
        self._pytest_task_result_dir: Optional[Path] = None
        self._pytest_task_timestamp: Optional[str] = None
        self._pytest_task_key: Optional[str] = None  # 用于检测 episode/task 变化
        
        # 线程锁与并发限制
        self.use_thread_lock = self.env_config.get("use_thread_lock", False)
        self.thread_lock = thread_lock if self.use_thread_lock else nullcontext()
        
        self.max_env_step_concurrent = self.env_config.get("max_env_step_concurrent", 0)
        self.env_step_limiter = nullcontext()
        if self.max_env_step_concurrent > 0:
            env_tag = self.env_config.get("tag", "default")
            self.env_step_limiter = get_global_limiter(tag=env_tag, max_concurrent_calls=self.max_env_step_concurrent)

        # 初始化环境
        with self.thread_lock, self.env_step_limiter:
            if "seed" in self.env_config['config']:
                self.env_config['config']["seed"] = self.env_config['group_seed']
            self.env = gem.make(env_id=self.env_config["env_type"], **self.env_config['config'])
            if "tool_wrapper" in self.env_config:
                self.env = tool_wrapper(self.env,
                                        wrapper_args=self.env_config.tool_wrapper.wrapper_args,
                                        tool_configs=self.env_config.tool_wrapper.tool_configs)

        # 加载模板
        self.cfg_template = self.pipeline_config.custom_envs[self.env_config["tag"]]
        self.agent_system_template = self.cfg_template["agent_system_template"]
        self.agent_template = self.cfg_template["agent_template"]

        # LLM Proxy
        self.llm_proxy: BaseLLMProxy = create_llm_proxy(
            generate_scheduler=self.generate_scheduler,
            llm_proxy_config=self.worker_config.llm_proxy,
            tokenizer=self.tokenizer,
            env=self.env
        )

        # 懒加载评估器
        self._rule_evaluator = None

    # =========================================================================
    #                               核心循环
    # =========================================================================

    def run_rollout_loop(self, data: DataProto):
        assert "seed" in data.meta_info
        self.running = True
        self.group_seed = data.meta_info['seed'] + self.env_config['group_seed']
        rollout_cache = self.reset()
        start_step = self.current_step

        while self.running and rollout_cache is not None:
            # 1. 生成决策
            lm_output = self.make_decision(rollout_cache)
            stop_reason = lm_output.meta_info.pop("stop_reason")

            # 如果 LLM 生成失败（例如推理引擎超时或崩溃），直接结束本 episode，
            # 否则会因为 stop_reason 既不是 FINISH 也不是 MAX_LENGTH 而陷入死循环。
            if stop_reason == GenerateStopReason.ABORT:
                self.logger.error(
                    "[run_rollout_loop] LLM generation aborted "
                    "(possibly due to timeout or inference engine failure). "
                    "Stopping current rollout to avoid hanging."
                )
                break

            # 2. 执行步骤
            if stop_reason == GenerateStopReason.FINISH:
                rollout_cache = self.step(lm_output)

            # 3. 结束处理
            if self.running and (rollout_cache.terminated or stop_reason == GenerateStopReason.MAX_LENGTH):
                rollout = self.formulate_rollouts(rollout_cache)
                
                # 设置 ID
                traj_group_id = f"{self.rollout_cache.tag}_{self.rollout_cache.group_id}_{self.episode_id}_{self.group_seed}"
                traj_id = f"{traj_group_id}_{self.rollout_cache.env_id}"
                rollout.non_tensor_batch["traj_group_id"] = np.array([traj_group_id] * rollout.batch.batch_size[0], dtype=object)
                rollout.non_tensor_batch["traj_id"] = np.array([traj_id] * rollout.batch.batch_size[0], dtype=object)
                
                # 添加调试日志和超时机制
                self.logger.info(
                    f"[formulate_rollouts] About to put rollout to queue: "
                    f"group_id={self.env_config['group_id']}, episode_id={self.episode_id}, start_step={start_step}, "
                    f"traj_id={traj_id}"
                )
                try:
                    put_ref = self.output_queue.put.remote(self.env_config['group_id'], self.episode_id, start_step, rollout)
                    result = ray.get(put_ref, timeout=120.0)  # 120秒超时
                    self.logger.info(f"[formulate_rollouts] Successfully put rollout to queue")
                except ray.exceptions.GetTimeoutError:
                    self.logger.error(
                        f"[formulate_rollouts] TIMEOUT: Failed to put rollout to queue after 120s. "
                        f"This may indicate a deadlock or queue blocking issue. "
                        f"group_id={self.env_config['group_id']}, episode_id={self.episode_id}"
                    )
                    # 继续执行，避免完全卡死，但记录错误
                    import traceback
                    self.logger.error(traceback.format_exc())
                except Exception as e:
                    self.logger.error(f"[formulate_rollouts] Error putting rollout to queue: {e}")
                    import traceback
                    self.logger.error(traceback.format_exc())
                    raise
                rollout_cache = self.reset()
                start_step = self.current_step

        ray.get(self.output_queue.put.remote(self.env_config['group_id'], self.episode_id, start_step, None))

    def reset(self) -> RolloutCache:
        self.rollout_cache = RolloutCache(env_id=self.env_config['env_id'],
                                          group_id=self.env_config['group_id'],
                                          tag=self.env_config['tag'])

        old_episode_id = self.episode_id
        self.episode_id = ray.get(self.output_queue.get_episode_id.remote(self.env_config['group_id']))
        if self.episode_id is None:
            return None
        
        # 如果 episode_id 变化，清除 pytest 任务目录缓存，让新 episode 创建新文件夹
        if old_episode_id != self.episode_id:
            self._pytest_task_result_dir = None
            self._pytest_task_timestamp = None
            self._pytest_task_key = None
        
        seed = self.group_seed + self.episode_id

        with self.thread_lock, self.env_step_limiter:
            observation, info = self.env.reset(seed=seed)
            if observation is None: return None
        
        self.rollout_cache.history.append({
            "observation": observation,
            "actions_left": self.env_config.max_steps - self.rollout_cache.step,
            "messages": None,
            **info,
        })
        return self.rollout_cache

    # =========================================================================
    #                               Step 逻辑 (含评估)
    # =========================================================================

    def step(self, llm_output: DataProto):
        # 1. 解码响应
        responses = self.tokenizer.batch_decode(llm_output.batch['responses'], skip_special_tokens=True)
        try:
            responses = [self._strip_control_tokens(r) for r in responses]
        except Exception: pass
        
        action = self.extract_action(responses[0])
        # 记录完整的 action（不截断，便于调试）
        if len(action) > 2000:
            self.logger.info(f"[Step {self.rollout_cache.step}] ACTION (truncated): {action[:2000]}...")
            self.logger.debug(f"[Step {self.rollout_cache.step}] ACTION (full): {action}")
        else:
            self.logger.info(f"[Step {self.rollout_cache.step}] ACTION: {action}")
        
        # 如果 action 为空，记录警告
        if not action or not action.strip():
            self.logger.warning(f"[Step {self.rollout_cache.step}] Empty action extracted from response: {responses[0][:500]}")

        # 2. 执行动作
        with self.thread_lock, self.env_step_limiter:
            raw_res = self.env.step(action=action)

        observation, reward, cost, terminated, truncated, info = self._normalize_env_result(raw_res)

        # 3. 更新状态
        self.rollout_cache.step += 1
        self.rollout_cache.terminated = terminated
        self.rollout_cache.truncated = truncated
        if self.rollout_cache.step >= self.env_config.max_steps:
            self.rollout_cache.terminated = True
            if not terminated: self.rollout_cache.truncated = True

        # 4. 记录历史（环境原始 reward/cost 单独保存，不参与训练信号）
        history_item = self.rollout_cache.history[-1]
        history_item.update({
            # 仅作为参考记录环境返回的原始 reward/cost，不用于最终训练信号
            'env_reward': reward,
            'env_cost': info.get('cost', cost),
            # 训练用的 reward/cost 将完全由 Linux Pytest 逻辑覆盖写入
            'reward': 0.0,
            'cost': 0.0,
            'llm_response': responses[0],
            'action': responses[0],
        })
        if info:
            try: history_item.update(info)
            except: history_item['env_info_repr'] = repr(info)

        # 5. 评估：仅使用 Linux Pytest 体系计算 reward / cost
        self._run_linux_pytest_evaluator()

        # 6. 收集额外观测（用于cost critic）
        # 注意：额外观测是在执行完 pytest 后收集的，保存到当前步骤的 history_item 中
        # 这样在下一步的 format_messages 中，可以读取当前步骤的额外观测并拼接到 prompt 中
        self._collect_extra_observations(history_item)

        # 7. 打印日志（在评估器运行后，显示更新后的reward和cost）
        self.logger.info(
            f"[Step {self.rollout_cache.step}] ENV STEP RESULT:\n"
            f"Reward: {history_item.get('reward', 0)}, Cost: {history_item.get('cost', 0)}, Done={terminated}\n"
            f"Observation preview: {str(observation)[:2000]}\n"
        )

        # 8. 准备下一步
        # 注意：额外观测已保存到当前步骤的 history_item 中（第310行）
        # 在下一步的 format_messages() 中，history[-1] 是新步骤，history[-2] 是当前步骤
        # 所以 format_messages() 需要使用 history[-2] 来读取上一步（当前步骤）的额外观测
        next_history_item = {
            "observation": observation,
            "actions_left": self.env_config.max_steps - self.rollout_cache.step,
            "messages": None
        }
        self.rollout_cache.history.append(next_history_item)
        return self.rollout_cache

    def _normalize_env_result(self, raw_res):
        res = raw_res.get('result', raw_res) if isinstance(raw_res, dict) else raw_res
        observation, reward, cost, terminated, truncated, info = None, 0, 0, False, False, {}
        
        if isinstance(res, (list, tuple)):
            if len(res) == 5:
                if isinstance(res[2], (int, float, type(None))) and isinstance(res[3], bool):
                    observation, reward, c, terminated, info = res
                    cost = c if c is not None else 0
                else:
                    observation, reward, terminated, truncated, info = res
            elif len(res) == 4:
                observation, reward, terminated, info = res
            else:
                try:
                    observation = res[0]
                    reward = res[1]
                    info = res[-1]
                except: pass
        else:
            observation = res

        if info is None: info = {}
        return observation, reward, cost, terminated, truncated, info

    # =========================================================================
    #                               评估器实现
    # =========================================================================

    def _get_task_type(self, task_id: str) -> str:
        """
        根据任务ID从配置文件中获取任务类型（general 或 harm）。
        
        Args:
            task_id: 任务ID
            
        Returns:
            任务类型字符串（'general' 或 'harm'），如果找不到则返回 'task'
        """
        try:
            # 从 env_config 中获取 task_config_path
            config = self.env_config.get('config', {})
            task_config_path = config.get('task_config_path')
            if not task_config_path:
                return 'task'
            
            # 处理相对路径（相对于 OSWorld-dev 目录）
            if not os.path.isabs(task_config_path):
                task_config_path = os.path.join(OSWORLD_PATH, task_config_path)
            
            # 加载 JSON 配置文件
            if not os.path.exists(task_config_path):
                self.logger.debug(f"Task config file not found: {task_config_path}")
                return 'task'
            
            with open(task_config_path, 'r', encoding='utf-8') as f:
                task_config = json.load(f)
            
            # 查找任务ID在哪个列表中
            if 'general' in task_config and isinstance(task_config['general'], list):
                if task_id in task_config['general']:
                    return 'general'
            
            if 'harm' in task_config and isinstance(task_config['harm'], list):
                if task_id in task_config['harm']:
                    return 'harm'
            
            # 如果找不到，返回默认值
            return 'task'
        except Exception as e:
            self.logger.debug(f"Failed to get task type for {task_id}: {e}")
            return 'task'
    
    def _get_task_id(self) -> Optional[str]:
        """
        获取当前任务的 ID，用于生成唯一的结果目录。
        
        按优先级尝试多种方法：
        1. 从 env.get_task_info() 获取
        2. 从 env.task 属性获取
        3. 从 rollout_cache.history 中获取
        4. 从环境变量获取
        
        Returns:
            任务 ID 字符串，如果找不到则返回 None
        """
        # 方法1: 从 env 的 get_task_info 方法获取
        if hasattr(self.env, 'get_task_info'):
            try:
                task_info = self.env.get_task_info()
                if isinstance(task_info, dict):
                    task_id = task_info.get('id')
                    if task_id:
                        return task_id
                # 兼容返回格式为 {'result': {...}} 的情况
                if isinstance(task_info, dict) and 'result' in task_info:
                    result = task_info['result']
                    if isinstance(result, dict):
                        task_id = result.get('id')
                        if task_id:
                            return task_id
            except Exception as e:
                self.logger.debug(f"Failed to get task info from env.get_task_info(): {e}")
        
        # 方法2: 从 env 的 task 属性获取
        if hasattr(self.env, 'task'):
            task_obj = getattr(self.env, 'task', None)
            if isinstance(task_obj, dict):
                task_id = task_obj.get('id')
            elif task_obj:
                task_id = getattr(task_obj, 'id', None)
            if task_id:
                return task_id
        
        # 方法3: 从 rollout_cache 的历史记录中获取
        if hasattr(self, 'rollout_cache') and hasattr(self.rollout_cache, 'history'):
            for entry in reversed(self.rollout_cache.history):
                if isinstance(entry, dict):
                    # 首先直接从entry中查找task_id（因为reset时info被展开到entry中）
                    task_id = entry.get('task_id')
                    if task_id:
                        return task_id
                    # 然后从entry中的task对象获取
                    task_obj = entry.get('task')
                    if isinstance(task_obj, dict):
                        task_id = task_obj.get('id')
                        if task_id:
                            return task_id
                    # 最后从entry中的info字典获取（兼容旧格式）
                    info = entry.get('info', {})
                    if isinstance(info, dict):
                        task_id = info.get('task_id')
                        if not task_id:
                            task_obj = info.get('task')
                            if isinstance(task_obj, dict):
                                task_id = task_obj.get('id')
                        if task_id:
                            return task_id
        
        # 方法4: 从环境变量获取
        task_id = os.environ.get('OSWORLD_CURRENT_TASK_ID') or os.environ.get('CURRENT_TASK_ID')
        if task_id:
            return task_id
        
        return None

    def _write_manifests_from_config(self, result_dir: Path, cfg: Dict[str, Any]):
        """
        根据任务配置中的 manifests 项写入 manifest 文件。
        
        这些 manifest 文件会被 pytest 评估器读取，用于：
        - file_compare_manifest.json: 文件内容比对任务
        - system_settings_manifest.json: 系统设置检查任务
        
        Args:
            result_dir: 结果目录路径（pytest 评估器会在此目录查找 manifest）
            cfg: step_pytest 配置字典，应包含 'manifests' 键
        """
        manifests = cfg.get('manifests', {})
        if not manifests:
            return
        
        try:
            # 写入文件比对 manifest（如果配置中存在）
            file_manifest = manifests.get('file_compare_manifest')
            if file_manifest:
                manifest_path = result_dir / "file_compare_manifest.json"
                manifest_path.write_text(
                    json.dumps(file_manifest, indent=2, ensure_ascii=False)
                )
                self.logger.debug(
                    f"Written file_compare_manifest.json with {len(file_manifest)} entries "
                    f"to {manifest_path}"
                )
            
            # 写入系统设置 manifest（如果配置中存在）
            system_manifest = manifests.get('system_settings_manifest')
            if system_manifest:
                manifest_path = result_dir / "system_settings_manifest.json"
                manifest_path.write_text(
                    json.dumps(system_manifest, indent=2, ensure_ascii=False)
                )
                self.logger.debug(
                    f"Written system_settings_manifest.json with {len(system_manifest)} entries "
                    f"to {manifest_path}"
                )
        except Exception as e:
            self.logger.warning(f"Failed to write manifests from config: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())

    def _collect_extra_observations(self, history_item: Dict[str, Any]):
        """
        收集额外观测数据，用于 cost critic。
        
        使用 ExtraObservationCollector 来封装所有 OSWorld-dev 相关的逻辑，
        保持代码简洁和模块化。
        
        Args:
            history_item: 当前步骤的 history 项，用于存储额外观测
        """
        step_num = self.rollout_cache.step
        
        # 获取 pytest 结果目录
            result_dir = self._pytest_task_result_dir
            if not result_dir or not result_dir.exists():
                history_item['extra_observations'] = {}
                return
            
        # 使用收集器收集额外观测（所有 OSWorld-dev 相关逻辑都在收集器中）
        collector = ExtraObservationCollector(result_dir=result_dir, logger=self.logger)
        extra_obs = collector.collect(step_num=step_num)
            
            # 保存到 history_item
            history_item['extra_observations'] = extra_obs
            
            if extra_obs:
                self.logger.info(
                f"[Step {step_num}] [_collect_extra_observations] ✓ 收集完成: {len(extra_obs)} 个额外观测 "
                    f"(keys: {list(extra_obs.keys())})"
                )
            else:
            self.logger.debug(f"[Step {step_num}] [_collect_extra_observations] 未找到额外观测")

    def _run_linux_pytest_evaluator(self):
        """
        使用 Linux Pytest 体系（如 linux_state_pytest.py）计算当前 step 的 reward/cost。
        """
        if run_step_pytest is None:
            self.logger.warning(
                "[_run_linux_pytest_evaluator] run_step_pytest is None, skipping pytest evaluation. "
                "Check if desktop_env.evaluators.metrics.step_pytest_runner is properly imported."
            )
            return

        step_num = int(self.rollout_cache.step)
        try:
            # 1) 合并 env 级别和 task 级别的 step_pytest 配置
            cfg_env = (
                self.env_config.get('step_pytest', {}) or 
                self.env_config.get('config', {}).get('step_pytest', {}) or 
                {}
            )
            cfg_task = {}
            if self.rollout_cache.history and isinstance(self.rollout_cache.history[0], dict):
                cfg_task = self.rollout_cache.history[0].get("step_pytest", {}) or {}

            # task 配置优先覆盖 env 配置
            cfg = {**cfg_env, **cfg_task}
            
            # 检查是否启用 step_pytest
            enabled = cfg.get('enabled', True)
            if not enabled:
                self.logger.info(f"[_run_linux_pytest_evaluator] step_pytest is disabled in config, skipping.")
                return

            # 获取 pytest_file 和 grading
            pytest_file = (
                cfg.get('per_step', {}).get(str(step_num), {}).get('pytest_file') or
                cfg.get('global_harm_detection', {}).get('pytest_file')
            )
            grading = (
                cfg.get('per_step', {}).get(str(step_num), {}).get('grading') or
                cfg.get('global_harm_detection', {}).get('grading')
            )

            if not pytest_file:
                self.logger.warning(
                    f"[_run_linux_pytest_evaluator] No pytest_file found in config "
                    f"(step={step_num}, cfg_keys={list(cfg.keys())}), skipping pytest evaluation."
                )
                return

            # 解析 pytest_file 路径
            osworld_root = "/data/share/projects/quanshu/OSWorld-dev"
            if not os.path.isabs(pytest_file):
                pytest_file = os.path.join(osworld_root, pytest_file)

            # 创建环境对象（用于 pytest runner）
            env_obj = None
            if hasattr(self.env, 'get_vm_info'):
                try:
                    info = self.env.get_vm_info()
                    self.logger.debug(f"[_run_linux_pytest_evaluator] get_vm_info returned: {info}")
                    # get_vm_info() 返回的可能是 {'result': {...}} 或直接的 {'vm_ip': ..., 'server_port': ...}
                    result = None
                    if info and isinstance(info, dict):
                        if 'result' in info:
                            # 格式1: {'result': {'vm_ip': ..., 'server_port': ...}}
                            result = info.get('result')
                        elif 'vm_ip' in info and 'server_port' in info:
                            # 格式2: 直接返回 {'vm_ip': ..., 'server_port': ...}
                            result = info
                    
                    if result and isinstance(result, dict):
                        vm_ip = result.get('vm_ip')
                        server_port = result.get('server_port')
                        self.logger.debug(f"[_run_linux_pytest_evaluator] vm_ip={vm_ip}, server_port={server_port}")
                        # 确保 vm_ip 和 server_port 都是有效的
                        if vm_ip and server_port is not None:
                            # 确保 server_port 是整数
                            try:
                                server_port = int(server_port)
                            except (ValueError, TypeError):
                                self.logger.warning(f"Invalid server_port: {server_port}, skipping vm_info")
                                vm_ip = None
                            if vm_ip:
                                class V:
                                    def __init__(self, i, p):
                                        self.vm_ip = str(i).strip()
                                        self.server_port = int(p)
                                env_obj = V(vm_ip, server_port)
                                self.logger.info(f"[_run_linux_pytest_evaluator] Created env_obj with vm_ip={vm_ip}, server_port={server_port}")
                            else:
                                self.logger.warning(f"[_run_linux_pytest_evaluator] vm_ip is empty after validation")
                        else:
                            self.logger.warning(f"[_run_linux_pytest_evaluator] vm_ip or server_port is None/invalid: vm_ip={vm_ip}, server_port={server_port}")
                    else:
                        self.logger.warning(f"[_run_linux_pytest_evaluator] get_vm_info returned invalid format or None: {info}")
                except Exception as e:
                    self.logger.warning(f"[_run_linux_pytest_evaluator] Failed to get_vm_info: {e}", exc_info=True)
            
            if env_obj is None:
                self.logger.warning(f"[_run_linux_pytest_evaluator] env_obj is None, pytest may not be able to connect to VM (OSWORLD_VM_IP/PORT not set)")

            # 获取任务 ID 用于生成唯一的 result_dir
            task_id = self._get_task_id() or "unknown"
            if task_id == "unknown":
                self.logger.warning(
                    f"Could not determine task ID, using 'unknown'. "
                    f"This may cause result directory conflicts."
                )

            # 创建结果目录 - 每个任务/episode 一个带时间戳的文件夹，所有步骤的结果保存在其中
            tag = self.env_config.get('tag', 'unknown')
            # 获取基础日志目录（从 HYDRA_RUN_DIR 或当前工作目录）
            base_log_dir = Path(os.environ.get("HYDRA_RUN_DIR", "./output/logs"))
            # 创建 pytest 子文件夹
            pytest_root = base_log_dir / "pytest"
            # 添加mode信息以区分train和val环境，避免结果互相覆盖
            mode_str = getattr(self, 'mode', 'unknown')
            
            # 检查是否是新的任务/episode（第一次调用或 episode_id/task_id 变化）
            current_episode_id = getattr(self, 'episode_id', 0)
            episode_task_key = f"{current_episode_id}_{task_id}"
            
            # 如果是第一次调用、任务/episode 变化、或目录不存在，创建新的任务级别目录
            if (self._pytest_task_result_dir is None or 
                self._pytest_task_timestamp is None or
                self._pytest_task_key != episode_task_key or
                not self._pytest_task_result_dir.exists()):
                # 获取任务类型（general 或 harm）
                task_type = self._get_task_type(task_id)
                
                # 创建任务级别的目录（带时间戳）
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                task_result_dir = pytest_root / (
                    f"linux_pytest_{tag}_{mode_str}_env{self.env_config.get('env_id', 0)}_"
                    f"ep{current_episode_id}_{task_type}_{task_id}_{timestamp}"
                )
                task_result_dir.mkdir(parents=True, exist_ok=True)
                self._pytest_task_result_dir = task_result_dir
                self._pytest_task_timestamp = timestamp
                self._pytest_task_key = episode_task_key
                self.logger.info(
                    f"Created new pytest task directory: {task_result_dir} (episode={current_episode_id}, task={task_id})"
                )
            
            # 所有步骤的结果直接保存在任务目录中（与Windows格式一致，不创建step子文件夹）
            # 设置环境变量为任务级别的目录（pytest runner 会使用这个）
            os.environ["OSWORLD_RESULT_DIR"] = str(self._pytest_task_result_dir)
            
            # 记录任务级别的目录（用于日志）
            result_dir = self._pytest_task_result_dir

            # 从配置中写入 manifests（不再硬编码任务特定逻辑）
            self._write_manifests_from_config(result_dir, cfg)
            
            # Save trajectory to traj.jsonl (like OSWorld-dev lib_run_single.py does)
            # This is needed for llm_judge to work
            if self.rollout_cache.history:
                traj_file = result_dir / "traj.jsonl"
                try:
                    # Get current step data
                    current_step_data = self.rollout_cache.history[-1]
                    traj_entry = {
                        "step_num": step_num,
                        "action": current_step_data.get('action', ''),
                        "reward": current_step_data.get('reward', 0),
                        "cost": current_step_data.get('cost', 0),
                        "done": self.rollout_cache.terminated,
                        "truncated": self.rollout_cache.truncated,
                        "info": current_step_data.get('info', {}),
                        "observation": current_step_data.get('observation', '')
                    }
                    # Append to traj.jsonl (like OSWorld-dev does)
                    with open(traj_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(traj_entry, ensure_ascii=False) + '\n')
                except Exception as e:
                    self.logger.warning(f"Failed to save trajectory to traj.jsonl: {e}")

            self.logger.info(
                f"Step {step_num}: Running Linux Pytest... "
                f"(pytest_file={pytest_file}, result_dir={result_dir})"
            )

            # 运行 pytest 评估
            grading = grading or {}
            summary = run_step_pytest(step_num, str(result_dir), pytest_file, grading, env_obj)

            if not summary:
                self.logger.warning("Linux Pytest returned empty summary.")
                return

            # 解析评估结果
            reward_val = 0.0  # 默认值
            cost_val = 0.0    # 默认值
            if 'breakdown' in summary:
                res = summary['breakdown'].get('summary', {}) or {}
                reward_val = res.get('reward_total', 0.0)
                cost_val = res.get('cost_total', 0.0)
                self.logger.info(f"Linux Pytest Result (with grading): {res}")
            else:
                # 兼容 fallback 计数格式
                reward_info = summary.get('reward', {})
                cost_info = summary.get('cost', {})
                if reward_info and reward_info.get('total', 0) > 0:
                    reward_val = float(reward_info.get('passed', 0))
                if cost_info and cost_info.get('total', 0) > 0:
                    cost_val = float(cost_info.get('failed', 0))
                self.logger.info(f"Linux Pytest Result (fallback): reward={reward_val}, cost={cost_val}")

            # 写入结果到 history（确保总是写入，即使值为0.0）
            hist = self.rollout_cache.history[-1]
            hist['reward'] = float(reward_val) if reward_val is not None else 0.0
            hist['cost'] = float(cost_val) if cost_val is not None else 0.0

        except Exception as e:
            self.logger.warning(f"Linux Pytest evaluator failed: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())

    # =========================================================================
    #                               Prompt 构造
    # =========================================================================

    def make_decision(self, rollout_cache: RolloutCache):
        lm_input = self.format_messages(rollout_cache)
        input_ids = lm_input.batch["input_ids"]

        if input_ids.shape[1] >= self.pipeline_config.sequence_length:
            return DataProto(meta_info={"stop_reason": GenerateStopReason.MAX_LENGTH})

        gen_cfg = self.worker_config.generating_args.to_dict()
        gen_cfg["max_new_tokens"] = min(self.env_config["max_tokens_per_step"], 
                                        self.pipeline_config.sequence_length - input_ids.shape[1])
        for k in ["eos_token_id", "pad_token_id"]:
            if k not in gen_cfg: gen_cfg[k] = getattr(self.tokenizer, k, None)

        lm_input.meta_info["src_rank"] = self.env_config["env_id"]

        input_messages = [m for h in self.rollout_cache.history for m in h["messages"]]
        self.logger.info(f"[Step {self.rollout_cache.step}] Generating (Text Only)...")
        
        lm_output = self.llm_proxy.generate(messages=input_messages, lm_input=lm_input, generation_config=gen_cfg)
        
        if lm_output is None:
            return DataProto(meta_info={"stop_reason": GenerateStopReason.ABORT})

        resp_ids = lm_output.batch['responses'][0].tolist()
        text = self.tokenizer.decode(resp_ids, skip_special_tokens=True)
        text = self._strip_control_tokens(text)
        
        self.rollout_cache.history[-1]["response_ids"] = resp_ids
        self.rollout_cache.history[-1]["messages"].append({"role": "assistant", "content": text})
        
        lm_output.meta_info["stop_reason"] = GenerateStopReason.FINISH
        return lm_output

    def format_messages(self, history: RolloutCache) -> DataProto:
        """纯文本 Prompt 构造"""
        content = self.rollout_cache.history[-1]
        messages = []
        user_content = ""

        if self.rollout_cache.step == 0:
            messages.append({"role": "system", "content": self.agent_system_template})
            if "env_instruction" in history.history[0]:
                user_content = f"{history.history[0]['env_instruction']}\n"

        if len(self.rollout_cache.history) > 1 and self.rollout_cache.history[-2].get("use_tool"):
            tool_out = str(content.get("observation", ""))
            messages.append({"role": "tool", "content": tool_out})
        else:
            obs_val = content.get("observation")
            if isinstance(obs_val, dict):
                obs_str = obs_val.get('accessibility_tree') or obs_val.get('terminal') or str(obs_val)
            else:
                obs_str = str(obs_val) if obs_val is not None else ""

            obs_str = self._sanitize_obs_str(obs_str)
            if len(obs_str) > 20000: obs_str = f"[truncated] {obs_str[-20000:]}"

            render_dict = {
                "observation": obs_str,
                "suffix": content.get("suffix", ""),
                "actions_left": content.get("actions_left", 0),
                "max_response_length": self.env_config.get("max_tokens_per_step", 128)
            }
            if contains_renderable_field(self.agent_template, "turn_idx"):
                render_dict["turn_idx"] = self.rollout_cache.step + 1
            
            user_content += self.agent_template.format(**render_dict)
            
            # 注意：额外观测不再在 format_messages 中添加，仅在 formulate_rollouts 中为 cost critic 构建时使用
            # 这样可以避免额外观测影响 actor 的响应生成
            
            messages.append({"role": "user", "content": user_content})

            if len(self.rollout_cache.history) > 1:
                prev = self.rollout_cache.history[-2]
                if prev.get('action'):
                    messages.append({"role": "assistant", "content": f"previous_action:{prev['action']}"})

        prompt_ids = custom_apply_chat_template(messages=messages, tokenizer=self.tokenizer, add_generation_prompt=True)
        
        history_ids = []
        for item in self.rollout_cache.history[:-1]:
            history_ids.extend(self._coerce_ids(item.get("prompt_ids")))
            history_ids.extend(self._coerce_ids(item.get("response_ids")))
        
        if history_ids:
            prompt_ids = compute_conversation_end_token_id(self.tokenizer) + prompt_ids
            
        input_ids = torch.tensor(history_ids + prompt_ids, dtype=torch.long).unsqueeze(0)
        mask = torch.ones_like(input_ids)
        
        lm_input = DataProto()
        lm_input.batch = TensorDict({
            "input_ids": input_ids,
            "attention_mask": mask,
            "position_ids": mask.cumsum(dim=-1),
        }, batch_size=1)
        
        content["prompt_ids"] = prompt_ids
        content["messages"] = messages
        return lm_input

    # =========================================================================
    #                               Helpers
    # =========================================================================

    def extract_action(self, response: str) -> str:
        """
        提取 action 代码块
        支持多种格式：
        1. ```python\n...\n```
        2. ```\n...\n``` (无语言标识)
        3. 如果找不到代码块，返回原始响应（让 OSWorld 的 parse_code_from_string 处理）
        """
        txt = str(response)
        
        # 尝试匹配 ```python\n...\n```
        m = re.search(r"```\s*python\s*\n([\s\S]*?)\n```", txt, re.IGNORECASE)
        if m:
            code = m.group(1).strip()
            if code:  # 确保代码不为空
                return f"```python\n{code}\n```"
        
        # 尝试匹配 ```\n...\n``` (无语言标识)
        m = re.search(r"```\s*\n([\s\S]*?)\n```", txt, re.IGNORECASE)
        if m:
            code = m.group(1).strip()
            if code:  # 确保代码不为空
                return f"```python\n{code}\n```"
        
        # 如果找不到代码块，返回原始响应（OSWorld 的 parse_code_from_string 可能会处理）
        # 但先检查是否包含代码特征
        if any(keyword in txt.lower() for keyword in ['pyautogui', 'time.sleep', 'import', 'def ', 'print(']):
            # 看起来像代码，尝试包装成代码块
            return f"```python\n{txt.strip()}\n```"
        
        # 如果都不匹配，返回空字符串（让环境处理）
        self.logger.warning(f"Could not extract action from response: {txt[:500]}")
        return ""

    def formulate_rollouts(self, rollout_cache: RolloutCache):
        """[关键修复] 补全所有缺失的 non_tensor_batch 字段，防止 KeyError"""
        history = list(rollout_cache.history)
        if history and 'llm_response' not in history[-1]: history.pop() 
        
        # 修复：正确处理reward和cost的提取，避免将有效0值误判
        # 使用get方法，如果key不存在返回0，如果值是None也返回0，但保留有效的0值
        scores = []
        costs = []
        for h in history:
            r = h.get('reward')
            c = h.get('cost')
            scores.append(float(r) if r is not None else 0.0)
            costs.append(float(c) if c is not None else 0.0)
        ep_score = sum(scores)
        ep_cost = sum(costs)

        # LLM judge evaluation (trajectory-level reward) - same as OSWorld-dev lib_run_single.py
        # Check if llm_judge_result is already in the last step, otherwise try to call llm_judge
        llm_judge_result = None
        if history:
            last_step = history[-1]
            llm_judge_result = last_step.get('llm_judge_result')
            
            # If not already evaluated, try to call llm_judge (like OSWorld-dev does after the loop)
            if llm_judge_result is None and llm_judge_task_completion is not None:
                # Get pytest result directory (where traj.jsonl should be)
                pytest_result_dir = self._pytest_task_result_dir
                self.logger.info(f"LLM judge check: pytest_result_dir={pytest_result_dir}, exists={pytest_result_dir.exists() if pytest_result_dir else False}")
                if pytest_result_dir and pytest_result_dir.exists():
                    traj_file = pytest_result_dir / "traj.jsonl"
                    instruction = last_step.get('env_instruction') or (history[0].get('env_instruction') if history else None)
                    self.logger.info(f"LLM judge check: traj_file={traj_file}, exists={traj_file.exists()}, instruction={bool(instruction)}")
                    
                    if traj_file.exists() and instruction:
                        try:
                            judge_model = os.getenv("LLM_JUDGE_MODEL", "gpt-4")
                            llm_judge_detail = llm_judge_task_completion(
                                traj_file=str(traj_file),
                                instruction=instruction,
                                result_dir=str(pytest_result_dir),
                                model=judge_model
                            )
                            llm_judge_result = llm_judge_detail.get("result", 0)
                            self.logger.info(f"LLM judge result: {llm_judge_result} (1=completed, 0=not completed)")
                            
                            # Explicitly save llm_judge_result.json to pytest result directory
                            # This ensures the file is saved even if TaskCompletionJudge.judge() fails internally
                            llm_judge_result_file = pytest_result_dir / "llm_judge_result.json"
                            try:
                                import json
                                with open(llm_judge_result_file, 'w', encoding='utf-8') as f:
                                    json.dump(llm_judge_detail, f, indent=2, ensure_ascii=False)
                                self.logger.info(f"LLM judge result saved to: {llm_judge_result_file}")
                            except Exception as save_e:
                                self.logger.warning(f"Failed to save LLM judge result to {llm_judge_result_file}: {save_e}")
                            
                            # Save to history's last step (so it can be used later)
                            last_step['llm_judge_result'] = llm_judge_result
                            last_step['llm_judge_detail'] = {
                                "completed": llm_judge_detail.get("completed", False),
                                "reason": llm_judge_detail.get("reason", ""),
                                "error": llm_judge_detail.get("error"),
                                "prompt": llm_judge_detail.get("prompt"),
                                "response": llm_judge_detail.get("response"),
                                "model": llm_judge_detail.get("model"),
                                "api_base": llm_judge_detail.get("api_base")
                            }
                        except Exception as e:
                            self.logger.warning(f"LLM judge evaluation failed: {e}")
                            self.logger.debug(traceback.format_exc())
                            # Save error detail to file even when exception occurs
                            try:
                                import json
                                error_detail = {
                                    "result": 0,
                                    "completed": False,
                                    "reason": "",
                                    "prompt": None,
                                    "response": None,
                                    "error": f"Exception during LLM judge evaluation: {str(e)}"
                                }
                                llm_judge_result_file = pytest_result_dir / "llm_judge_result.json"
                                with open(llm_judge_result_file, 'w', encoding='utf-8') as f:
                                    json.dump(error_detail, f, indent=2, ensure_ascii=False)
                                self.logger.info(f"LLM judge error detail saved to: {llm_judge_result_file}")
                            except Exception as save_e:
                                self.logger.warning(f"Failed to save LLM judge error detail: {save_e}")
            
            # Use llm_judge_result as episode reward if available
            if llm_judge_result is not None:
                try:
                    llm_judge_result = float(llm_judge_result)
                    self.logger.info(f"Using LLM judge result as episode reward: {llm_judge_result} (pytest reward: {ep_score})")
                    ep_score = llm_judge_result
                except (ValueError, TypeError):
                    self.logger.warning(f"Invalid llm_judge_result value: {llm_judge_result}, using pytest reward instead")

        if ep_score == 0 and hasattr(self.env, 'evaluate'):
            try:
                eval_score = self.env.evaluate()
                if isinstance(eval_score, (int, float)): ep_score = float(eval_score)
            except Exception: pass

        # 统计额外观测的使用情况
        extra_obs_stats = {
            "total_steps": len(history),
            "steps_with_extra_obs": 0,
            "steps_without_extra_obs": 0,
            "extra_obs_keys": set()
        }
        for idx, item in enumerate(history):
            extra_obs = item.get("extra_observations", {})
            if extra_obs:
                extra_obs_stats["steps_with_extra_obs"] += 1
                extra_obs_stats["extra_obs_keys"].update(extra_obs.keys())
            else:
                extra_obs_stats["steps_without_extra_obs"] += 1
        
        self.logger.info(
            f"[formulate_rollouts] 额外观测统计: "
            f"总步数={extra_obs_stats['total_steps']}, "
            f"有额外观测={extra_obs_stats['steps_with_extra_obs']}步, "
            f"无额外观测={extra_obs_stats['steps_without_extra_obs']}步, "
            f"观测类型={list(extra_obs_stats['extra_obs_keys'])}"
        )
        
        all_ids = []
        prompt_masks = []
        response_masks = []
        step_response_lengths = []  # 记录每个step的response长度，用于step-level cost分配
        
        for item in history:
            p = self._coerce_ids(item.get("prompt_ids"))
            r = self._coerce_ids(item.get("response_ids"))
            all_ids.extend(p + r)
            prompt_masks.extend([1] * len(p) + [0] * len(r))
            response_masks.extend([0] * len(p) + [1] * len(r))
            step_response_lengths.append(len(r))  # 记录当前step的response长度
        
        self.logger.info(
            f"[formulate_rollouts] Prompt 构造完成: "
            f"总token数={len(all_ids)}, "
            f"prompt tokens={sum(prompt_masks)}, "
            f"response tokens={sum(response_masks)}"
        )
        
        seq_len = self.pipeline_config.sequence_length
        input_tensor = pad_to_length(torch.tensor(all_ids).unsqueeze(0), seq_len, self.tokenizer.pad_token_id)
        
        # 补全 Mask 和 Score Tensor
        att_mask = pad_to_length(torch.ones(1, len(all_ids)), seq_len, 0)
        pos_ids = pad_to_length(att_mask.cumsum(dim=-1), seq_len, 0)
        resp_mask = pad_to_length(torch.tensor(response_masks).unsqueeze(0), seq_len, 0)
        prm_mask = pad_to_length(torch.tensor(prompt_masks).unsqueeze(0), seq_len, 0)
        
        score_tensor = torch.zeros_like(input_tensor, dtype=torch.float)
        if len(all_ids) > 0: score_tensor[0, len(all_ids)-1] = ep_score

        lm_input = DataProto()
        lm_input.batch = TensorDict({
            "input_ids": input_tensor,
            "attention_mask": att_mask,
            "position_ids": pos_ids,
            "response_mask": resp_mask,
            "prompt_mask": prm_mask,
            "scores": score_tensor,
        }, batch_size=1)
        
        # 计算 Response Level Reward (RL 训练必需)
        # Formula: R = reward - cost_coef * cost
        # Default cost_coef is 0.3, can be configured via env_config.reward_cost_coef
        cost_coef = float(self.env_config.get('reward_cost_coef', 0.3))
        response_level = float(ep_score) - cost_coef * float(ep_cost)
        lm_input.batch["response_level_rewards"] = torch.tensor([response_level], dtype=torch.float)
        
        # 记录 response_level 计算详情（用于调试和监控）
        self.logger.info(
            f"[formulate_rollouts] Response Level Reward Calculation: "
            f"ep_score={ep_score:.4f}, ep_cost={ep_cost:.4f}, cost_coef={cost_coef:.4f}, "
            f"response_level={response_level:.4f} (R = {ep_score:.4f} - {cost_coef:.4f} * {ep_cost:.4f})"
        )
        
        # 为 cost critic 构建包含额外观测的 input_ids（仅在启用 cost_constraint 时）
        if getattr(self.pipeline_config, 'enable_cost_constraint', False):
            try:
                cost_critic_all_ids = self._build_cost_critic_prompt_ids_with_extra_obs(history, rollout_cache)
                cost_input_tensor = pad_to_length(torch.tensor(cost_critic_all_ids).unsqueeze(0), seq_len, self.tokenizer.pad_token_id)
                cost_att_mask = pad_to_length(torch.ones(1, len(cost_critic_all_ids)), seq_len, 0)
                cost_pos_ids = pad_to_length(cost_att_mask.cumsum(dim=-1), seq_len, 0)
                
                # 计算 cost critic 的 prompt_mask 和 response_mask
                # 使用与 _build_cost_critic_prompt_ids_with_extra_obs 相同的逻辑来计算每个 step 的长度
                cost_prompt_masks = []
                cost_response_masks = []
                
                for idx, item in enumerate(history):
                    item_messages = item.get("messages", [])
                    
                    if not item_messages:
                        # 降级方案：使用原始的 prompt_ids 长度
                        p_len = len(self._coerce_ids(item.get("prompt_ids")))
                    else:
                        # 获取上一步的额外观测
                        extra_obs = {}
                        if idx > 0:
                            prev_item = history[idx - 1]
                            extra_obs = prev_item.get("extra_observations", {})
                        
                        # 使用辅助方法构建包含额外观测的 messages（与 _build_cost_critic_prompt_ids_with_extra_obs 保持一致）
                        messages_with_extra = self._build_messages_with_extra_obs(item_messages, extra_obs, idx)
                        
                        prompt_ids_with_extra = custom_apply_chat_template(
                            messages=messages_with_extra, 
                            tokenizer=self.tokenizer, 
                            add_generation_prompt=True
                        )
                        
                        # 如果有历史，需要添加 conversation end token
                        if idx > 0:
                            prompt_ids_with_extra = compute_conversation_end_token_id(self.tokenizer) + prompt_ids_with_extra
                        
                        p_len = len(prompt_ids_with_extra)
                    
                    r_len = len(self._coerce_ids(item.get("response_ids")))
                    cost_prompt_masks.extend([1] * p_len + [0] * r_len)
                    cost_response_masks.extend([0] * p_len + [1] * r_len)
                
                cost_resp_mask = pad_to_length(torch.tensor(cost_response_masks).unsqueeze(0), seq_len, 0)
                cost_prm_mask = pad_to_length(torch.tensor(cost_prompt_masks).unsqueeze(0), seq_len, 0)
                
                # 将 cost critic 的 input_ids 添加到 batch 中
                lm_input.batch["cost_input_ids"] = cost_input_tensor
                lm_input.batch["cost_attention_mask"] = cost_att_mask
                lm_input.batch["cost_position_ids"] = cost_pos_ids
                lm_input.batch["cost_response_mask"] = cost_resp_mask
                lm_input.batch["cost_prompt_mask"] = cost_prm_mask
                
                self.logger.info(
                    f"[formulate_rollouts] 为 cost critic 构建了包含额外观测的 input_ids: "
                    f"总长度={len(cost_critic_all_ids)} tokens"
                )
            except Exception as e:
                self.logger.warning(
                    f"[formulate_rollouts] 构建 cost critic 额外观测 input_ids 失败: {e}, "
                    f"cost critic 将使用原始 input_ids"
                )
                import traceback
                self.logger.debug(traceback.format_exc())

        # 补全 non_tensor_batch (解决 KeyError: step_scores)
        lm_input.non_tensor_batch.update({
            "env_ids": np.array([rollout_cache.env_id], dtype=object),
            "group_ids": np.array([rollout_cache.group_id], dtype=object),
            "tags": np.array([rollout_cache.tag], dtype=object),
            "episode_scores": np.array([ep_score], dtype=object),
            "episode_costs": np.array([ep_cost], dtype=object),
            "step_scores": np.array([scores], dtype=object),
            "step_costs": np.array([costs], dtype=object),
            "step_response_lengths": np.array([step_response_lengths], dtype=object),  # 保存每个step的response长度，用于step-level cost分配
            "frames": np.array([rollout_cache.frames], dtype=object),
        })

        metrics_agg_mode = history[-1].get('metrics_agg_mode', {}) if history else {}
        history_metrics = [item.get("metrics", {}) for item in history]
        env_metric = aggregate_metrics(history_metrics=history_metrics, metrics_agg_mode=metrics_agg_mode)
        env_metric["num_actions"] = rollout_cache.step
        env_metric["episode_return"] = ep_score
        
        env_metric = {f"env/{rollout_cache.tag}/{k}": v for k, v in env_metric.items()}
        # 补全 response_length metric
        if resp_mask.numel() > 0:
            env_metric["env/response_length"] = resp_mask.sum(dim=-1).float().mean().item()

        save_data = {
            "history": history, 
            "episode_score": float(ep_score),
            "episode_cost": float(ep_cost),
            "env_metric": env_metric,
            "env_id": int(rollout_cache.env_id),
            "group_id": int(rollout_cache.group_id),
            "tag": rollout_cache.tag,
            "mode": self.mode
        }
        lm_input.non_tensor_batch["save_content"] = np.array([json.dumps(save_data, ensure_ascii=False)], dtype=object)
        
        colummns_config = [
            ["traj_id", "string"],
            ["env_ids", "string"],
            ["episode_scores", "double"],
            ["save_content", "string"],
        ]

        lm_input.meta_info = {
            "metrics": env_metric, 
            "COLUMMNS_CONFIG": colummns_config
        }
        
        return lm_input

    def _coerce_ids(self, x):
        if isinstance(x, list): return x
        return []

    def _strip_control_tokens(self, text: str) -> str:
        text = re.sub(r"<\|[^>|]+\|>", "", str(text))
        return text.strip()

    def _sanitize_obs_str(self, s: str) -> str:
        return re.sub(r"[\x00-\x08\x0B-\x1F\x7F]", "", str(s))
    
    def _build_messages_with_extra_obs(self, original_messages: List[Dict], extra_obs: Dict[str, Any], step_idx: int) -> List[Dict]:
        """
        为给定的 messages 添加额外观测信息。
        
        Args:
            original_messages: 原始的 messages 列表
            extra_obs: 额外观测字典
            step_idx: 步骤索引（用于日志记录）
            
        Returns:
            包含额外观测的 messages 列表
        """
        if not extra_obs:
            return original_messages
        
        # 使用独立的格式化函数（不依赖类状态）
        extra_obs_str = format_extra_observations(extra_obs)
        if not extra_obs_str:
            return original_messages
        
        messages_with_extra = []
        for msg in original_messages:
            if msg.get("role") == "user":
                original_content = msg.get("content", "")
                content_with_extra = original_content + "\n\n[额外观测信息（用于安全评估）]\n" + extra_obs_str
                messages_with_extra.append({"role": "user", "content": content_with_extra})
                self.logger.debug(
                    f"[formulate_rollouts] Step {step_idx}: 为 cost critic 添加额外观测, "
                    f"长度: {len(extra_obs_str)} 字符"
                )
            else:
                messages_with_extra.append(msg)
        
        return messages_with_extra
    
    def _build_cost_critic_prompt_ids_with_extra_obs(self, history: List[Dict], rollout_cache: RolloutCache) -> List[int]:
        """
        为 cost critic 构建包含额外观测的 prompt_ids。
        
        Args:
            history: rollout history，每个 item 包含 prompt_ids, response_ids, extra_observations 等
            rollout_cache: rollout cache，用于获取 step 信息
            
        Returns:
            包含额外观测的完整 input_ids 列表
        """
        all_ids = []
        
        for idx, item in enumerate(history):
            # 获取原始的 messages（保存在 history 中）
            original_messages = item.get("messages", [])
            
            # 如果没有 messages，使用原始的 prompt_ids（降级方案）
            if not original_messages:
                prompt_ids = self._coerce_ids(item.get("prompt_ids"))
                all_ids.extend(prompt_ids)
                response_ids = self._coerce_ids(item.get("response_ids"))
                all_ids.extend(response_ids)
                continue
            
            # 获取上一步的额外观测（用于当前 step）
            extra_obs = {}
            if idx > 0:
                prev_item = history[idx - 1]
                extra_obs = prev_item.get("extra_observations", {})
            
            # 使用辅助方法构建包含额外观测的 messages
            messages_with_extra_obs = self._build_messages_with_extra_obs(original_messages, extra_obs, idx)
            
            # 使用包含额外观测的 messages 构建 prompt_ids
            prompt_ids = custom_apply_chat_template(
                messages=messages_with_extra_obs, 
                tokenizer=self.tokenizer, 
                add_generation_prompt=True
            )
            
            # 如果有历史，需要添加 conversation end token
            if idx > 0 and all_ids:
                prompt_ids = compute_conversation_end_token_id(self.tokenizer) + prompt_ids
            
            all_ids.extend(prompt_ids)
            
            # 添加 response_ids
            response_ids = self._coerce_ids(item.get("response_ids"))
            all_ids.extend(response_ids)
        
        return all_ids
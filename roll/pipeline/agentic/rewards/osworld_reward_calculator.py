"""
OSWorld Reward Calculator
集中管理 OSWorld 环境的 reward 和 cost 计算逻辑
"""
import os
import tempfile
from typing import Optional, Dict, Any, Tuple
from roll.utils.logging import get_logger

logger = get_logger()

# 动态导入 OSWorld 相关评估模块
OSWORLD_PATH = "/data/share/projects/quanshu/OSWorld-dev"
if OSWORLD_PATH not in __import__('sys').path:
    __import__('sys').path.insert(0, OSWORLD_PATH)

run_step_pytest = None
RuleBasedEvaluator = None

try:
    from desktop_env.evaluators.metrics.step_pytest_runner import run_step_pytest
except ImportError:
    pass

try:
    from desktop_env.evaluators.rule_based import RuleBasedEvaluator
except ImportError:
    pass


class OSWorldRewardCalculator:
    """
    OSWorld 环境的 Reward 和 Cost 计算器
    集中管理所有评估器的调用逻辑
    """
    
    def __init__(self, env_config: Dict[str, Any]):
        self.env_config = env_config
        self._rule_evaluator = None
        
        # Pytest 配置
        # 支持两种路径: env_config.step_pytest 或 env_config.config.step_pytest
        self.pytest_cfg = (
            env_config.get('step_pytest') or 
            env_config.get('config', {}).get('step_pytest', {})
        )
    
    def get_rule_evaluator(self):
        """懒加载 RuleBasedEvaluator"""
        if RuleBasedEvaluator is None:
            return None
        if self._rule_evaluator is None:
            try:
                self._rule_evaluator = RuleBasedEvaluator()
            except Exception as e:
                logger.warning(f"Failed to initialize RuleBasedEvaluator: {e}")
                return None
        return self._rule_evaluator
    
    def _load_grading_from_file(self, global_cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        尝试从外部文件加载 grading 配置
        
        支持以下方式：
        1. grading_file 路径（相对于 OSWorld-dev 或绝对路径）
        2. 默认路径：与 pytest_file 同目录下的 system_harm_grading.yaml
        """
        import yaml
        from pathlib import Path
        
        # 方式1: 从配置中读取 grading_file
        grading_file = global_cfg.get('grading_file')
        if grading_file:
            if not os.path.isabs(grading_file):
                grading_file = os.path.join(OSWORLD_PATH, grading_file)
            if os.path.exists(grading_file):
                try:
                    with open(grading_file, 'r', encoding='utf-8') as f:
                        data = yaml.safe_load(f)
                        grading = data.get('grading', {})
                        if grading:
                            logger.info(f"Loaded grading config from: {grading_file}")
                            return grading
                except Exception as e:
                    logger.warning(f"Failed to load grading from {grading_file}: {e}")
        
        # 方式2: 尝试从 pytest_file 同目录下的 system_harm_grading.yaml
        pytest_file = global_cfg.get('pytest_file')
        if pytest_file:
            if not os.path.isabs(pytest_file):
                pytest_file = os.path.join(OSWORLD_PATH, pytest_file)
            pytest_path = Path(pytest_file)
            default_grading_file = pytest_path.parent / "system_harm_grading.yaml"
            if default_grading_file.exists():
                try:
                    with open(default_grading_file, 'r', encoding='utf-8') as f:
                        data = yaml.safe_load(f)
                        grading = data.get('grading', {})
                        if grading:
                            logger.info(f"Loaded grading config from default location: {default_grading_file}")
                            return grading
                except Exception as e:
                    logger.warning(f"Failed to load grading from {default_grading_file}: {e}")
        
        return None
    
    def compute_rule_evaluator_reward_cost(self, action: str) -> Tuple[Optional[float], Optional[float]]:
        """
        计算规则评估器的 reward 和 cost
        
        Returns:
            (reward, cost): reward 和 cost 值，如果为 None 表示不更新
        """
        evaluator = self.get_rule_evaluator()
        if evaluator is None:
            return None, None
        
        try:
            cmd = action.get('command', str(action)) if isinstance(action, dict) else str(action)
            r, c, _ = evaluator.score_step({'command': cmd})
            return r, c
        except Exception as e:
            logger.warning(f"Rule eval failed: {e}")
            return None, None
    
    def compute_pytest_reward_cost(self, step_num: int, env) -> Tuple[Optional[float], Optional[float]]:
        """
        计算 Pytest 评估器的 reward 和 cost
        
        Returns:
            (reward, cost): reward 和 cost 值，如果为 None 表示不更新
        """
        if run_step_pytest is None:
            return None, None
        
        # 检查是否启用 step_pytest
        if not self.pytest_cfg.get('enabled', False):
            return None, None
        
        # 检查频率（每 N 步运行一次）
        frequency = self.pytest_cfg.get('frequency', 1)
        if step_num % frequency != 0:
            return None, None
        
        try:
            # 优先从 per_step 配置读取，否则从 global_harm_detection 读取
            per_step_cfg = self.pytest_cfg.get('per_step', {}).get(str(step_num), {})
            global_cfg = self.pytest_cfg.get('global_harm_detection', {})
            
            # 检查 global_harm_detection 是否启用
            if not global_cfg.get('enabled', True):
                return None, None
            
            pytest_file = per_step_cfg.get('pytest_file') or global_cfg.get('pytest_file')
            grading = per_step_cfg.get('grading') or global_cfg.get('grading')
            
            # 如果配置中没有 grading，尝试从外部文件加载
            if not grading:
                grading = self._load_grading_from_file(global_cfg)
            
            # 如果还是没有，使用空字典（fallback 逻辑）
            if not grading:
                grading = {}
            
            if not pytest_file:
                return None, None
            
            if not os.path.isabs(pytest_file):
                pytest_file = os.path.join(OSWORLD_PATH, pytest_file)
            
            env_obj = None
            if hasattr(env, 'get_vm_info'):
                info = env.get_vm_info()
                if info:
                    class V:
                        def __init__(self, i, p):
                            self.vm_ip, self.server_port = i, p
                    env_obj = V(info.get('vm_ip'), info.get('server_port'))
            
            logger.info(f"Step {step_num}: Running Pytest with grading config: {bool(grading)}")
            summary = run_step_pytest(step_num, tempfile.mkdtemp(), pytest_file, grading, env_obj)
            
            if not summary or not summary.get('ok'):
                logger.warning(f"Pytest execution failed or returned no results: {summary}")
                return None, None
            
            # 如果有 breakdown（使用 grading 配置时）
            if 'breakdown' in summary:
                res = summary['breakdown'].get('summary', {})
                reward = res.get('reward_total')
                cost = res.get('cost_total')
                logger.info(f"Pytest Result (with grading): reward={reward}, cost={cost}")
                return reward, cost
            
            # 如果没有 breakdown，使用 fallback 逻辑的结果（简单计数）
            # Fallback 返回格式: {"reward": {"passed": X, "total": Y}, "cost": {"failed": X, "total": Y}}
            if 'reward' in summary and 'cost' in summary:
                reward_info = summary.get('reward', {})
                cost_info = summary.get('cost', {})
                # 简单转换：passed/total -> reward_total, failed/total -> cost_total
                reward = float(reward_info.get('passed', 0)) if reward_info.get('total', 0) > 0 else None
                cost = float(cost_info.get('failed', 0)) if cost_info.get('total', 0) > 0 else None
                logger.info(f"Pytest Result (fallback): reward={reward}, cost={cost}")
                return reward, cost
            
            logger.warning(f"Pytest summary format not recognized: {summary.keys()}")
            return None, None
        except Exception as e:
            logger.warning(f"Pytest failed: {e}")
            return None, None
    
    def compute_reward_cost(
        self, 
        step_num: int,
        action: str,
        env,
        initial_reward: float = 0.0,
        initial_cost: float = 0.0
    ) -> Tuple[float, float]:
        """
        综合计算 reward 和 cost
        
        Args:
            step_num: 当前步骤编号
            action: 动作字符串
            env: 环境对象
            initial_reward: 环境返回的初始 reward
            initial_cost: 环境返回的初始 cost
        
        Returns:
            (final_reward, final_cost): 最终的 reward 和 cost
        """
        reward = initial_reward
        cost = initial_cost
        
        # 1. 规则评估器
        rule_reward, rule_cost = self.compute_rule_evaluator_reward_cost(action)
        if rule_reward is not None:
            reward = rule_reward
        if rule_cost is not None:
            cost = max(cost, float(rule_cost))
        if rule_reward is not None or rule_cost is not None:
            logger.info(f"Rule Eval: R={rule_reward}, C={rule_cost}")
        
        # 2. Pytest 评估器
        pytest_reward, pytest_cost = self.compute_pytest_reward_cost(step_num, env)
        if pytest_reward is not None:
            reward = pytest_reward
        if pytest_cost is not None:
            cost = max(cost, pytest_cost)
        
        return reward, cost












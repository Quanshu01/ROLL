"""
额外观测收集器模块
"""

import os
import sys
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
from contextlib import contextmanager


# 动态导入 OSWorld-dev 相关模块
OSWORLD_PATH = "/data/share/projects/quanshu/OSWorld-dev"
if OSWORLD_PATH not in sys.path:
    sys.path.insert(0, OSWORLD_PATH)

load_monitoring_state = None
try:
    from desktop_env.evaluators.linux_security_cost_pytest import load_monitoring_state
except ImportError:
    pass


class ExtraObservationCollector:
    """
    额外观测收集器
    """
    
    def __init__(self, result_dir: Optional[Path] = None, logger=None):
        """
        初始化收集器
        
        Args:
            result_dir: OSWorld-dev 的结果目录路径
            logger: 日志记录器
        """
        self.result_dir = result_dir
        self.logger = logger or self._get_default_logger()
    
    @staticmethod
    def _get_default_logger():
        """获取默认日志记录器"""
        import logging
        return logging.getLogger(__name__)
    
    @contextmanager
    def _set_result_dir_env(self, result_dir: Path):
        """临时设置 OSWORLD_RESULT_DIR 环境变量"""
        old_value = os.environ.get("OSWORLD_RESULT_DIR")
        os.environ["OSWORLD_RESULT_DIR"] = str(result_dir)
        try:
            yield
        finally:
            if old_value is not None:
                os.environ["OSWORLD_RESULT_DIR"] = old_value
            elif "OSWORLD_RESULT_DIR" in os.environ:
                del os.environ["OSWORLD_RESULT_DIR"]
    
    def collect(self, step_num: int = 0) -> Dict[str, Any]:
        """
        收集所有可用的额外观测数据
        
        Args:
            step_num: 当前步骤编号（用于日志和文件副本命名）
            
        Returns:
            包含额外观测数据的字典，格式为：
            {
                'linux_security_cost_state': {...},
                'linux_state_monitoring': {...}
            }
        """
        if not self.result_dir or not self.result_dir.exists():
            self.logger.debug(f"[ExtraObservationCollector] Result dir not available: {self.result_dir}")
            return {}
        
        extra_obs = {}
        
        # 设置环境变量以便 OSWorld-dev 接口能找到结果目录
        with self._set_result_dir_env(self.result_dir):
            # 收集安全状态监控数据
            security_state = self._collect_security_state(step_num)
            if security_state:
                extra_obs['linux_security_cost_state'] = security_state
            
            # 收集系统状态监控数据
            state_monitoring = self._collect_state_monitoring(step_num)
            if state_monitoring:
                extra_obs['linux_state_monitoring'] = state_monitoring
        
        return extra_obs
    
    def _collect_security_state(self, step_num: int) -> Optional[Dict[str, Any]]:
        """收集安全状态监控数据"""
        # 优先使用 OSWorld-dev 的接口
        if load_monitoring_state is not None:
            try:
                state = load_monitoring_state()
                if state:
                    self.logger.debug(
                        f"[ExtraObservationCollector] Loaded security state via OSWorld-dev interface "
                        f"(step={step_num}, keys={len(state)})"
                    )
                    self._save_step_copy("linux_security_cost_state.json", step_num)
                    return state
            except Exception as e:
                self.logger.warning(
                    f"[ExtraObservationCollector] Failed to load via OSWorld-dev interface: {e}, "
                    f"falling back to manual read"
                )
        
        # 降级方案：手动读取文件
        return self._read_json_file("linux_security_cost_state.json", step_num)
    
    def _collect_state_monitoring(self, step_num: int) -> Optional[Dict[str, Any]]:
        """收集系统状态监控数据（OSWorld-dev 没有公开接口，手动读取）"""
        return self._read_json_file("linux_state_monitoring.json", step_num, required=False)
    
    def _read_json_file(self, filename: str, step_num: int, required: bool = True) -> Optional[Dict[str, Any]]:
        """读取 JSON 文件"""
        file_path = self.result_dir / filename
        if not file_path.exists():
            if required:
                self.logger.debug(f"[ExtraObservationCollector] File not found: {file_path}")
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.logger.debug(
                    f"[ExtraObservationCollector] Loaded {filename} (step={step_num}, "
                    f"size={len(json.dumps(data, ensure_ascii=False))} chars)"
                )
                self._save_step_copy(filename, step_num)
                return data
        except Exception as e:
            self.logger.warning(f"[ExtraObservationCollector] Failed to read {filename}: {e}")
            return None
    
    def _save_step_copy(self, filename: str, step_num: int):
        """保存步骤副本，避免覆盖式写入导致无法观察每步变化"""
        source_file = self.result_dir / filename
        if not source_file.exists():
            return
        
        step_copy_name = f"{filename.replace('.json', '')}_step_{step_num}.json"
        step_copy_path = self.result_dir / step_copy_name
        
        try:
            shutil.copy2(source_file, step_copy_path)
            self.logger.debug(f"[ExtraObservationCollector] Saved step copy: {step_copy_name}")
        except Exception as e:
            self.logger.warning(f"[ExtraObservationCollector] Failed to save step copy: {e}")


def format_extra_observations(extra_obs: Dict[str, Any], max_length: int = 20000) -> str:
    """
    将额外观测数据格式化为字符串，用于拼接到 prompt 中。
    
    Args:
        extra_obs: 额外观测字典
        max_length: 最大长度限制
        
    Returns:
        格式化后的字符串
    """
    if not extra_obs:
        return ""
    
    parts = []
    
    # 格式化 linux_security_cost_state
    if 'linux_security_cost_state' in extra_obs:
        state = extra_obs['linux_security_cost_state']
        parts.append("=== Linux 安全状态监控 ===")
        if isinstance(state, dict):
            # 提取关键信息摘要
            if 'sudo_attempts' in state:
                parts.append(f"Sudo 尝试次数: {state['sudo_attempts']}")
            if 'failed_operations_count' in state:
                parts.append(f"失败操作计数: {state['failed_operations_count']}")
            if 'crash_logs_count' in state:
                parts.append(f"崩溃日志计数: {state['crash_logs_count']}")
            
            # 包含完整 JSON（但限制长度）
            state_str = json.dumps(state, ensure_ascii=False, indent=2)
            max_json_length = 15000
            if len(state_str) <= max_json_length:
                parts.append("\n完整状态（JSON）:\n" + state_str)
            else:
                truncated = state_str[:max_json_length] + f"\n[内容已截断，原始长度: {len(state_str)} 字符]"
                parts.append("\n完整状态（JSON，已截断）:\n" + truncated)
    
    # 格式化 linux_state_monitoring
    if 'linux_state_monitoring' in extra_obs:
        monitoring = extra_obs['linux_state_monitoring']
        if isinstance(monitoring, dict):
            parts.append("\n=== Linux 系统状态监控 ===")
            monitoring_str = json.dumps(monitoring, ensure_ascii=False, indent=2)
            max_monitoring_length = 10000
            if len(monitoring_str) <= max_monitoring_length:
                parts.append(monitoring_str)
            else:
                truncated = monitoring_str[:max_monitoring_length] + f"\n[内容已截断，原始长度: {len(monitoring_str)} 字符]"
                parts.append(truncated)
    
    result = "\n".join(parts)
    
    # 限制总长度
    if len(result) > max_length:
        result = result[:max_length] + f"\n[内容已截断，原始长度: {len(result)} 字符]"
    
    return result


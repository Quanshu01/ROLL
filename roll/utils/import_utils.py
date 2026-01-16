import importlib
from importlib.util import find_spec
from typing import Any, Optional
import traceback

from roll.utils.logging import get_logger


logger = get_logger()


def is_vllm_available() -> bool:
    return find_spec("vllm") is not None


def can_import_class(class_path: str) -> bool:
    try:
        module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        getattr(module, class_name)
        return True
    except Exception as e:
        logger.error(f"Failed to import class {class_path}: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return False


def safe_import_class(class_path: str) -> Optional[Any]:
    """
    安全导入类，即使模块级别的代码有异常（如环境重复注册），
    只要类本身能获取到就返回类对象。
    """
    try:
        module_path, class_name = class_path.rsplit(".", 1)
        # 先尝试导入模块，即使模块级别有警告/错误也继续
        try:
            module = importlib.import_module(module_path)
        except Exception as e:
            # 如果导入失败，记录错误但继续尝试
            logger.warning(f"Module import warning for {module_path}: {e}")
            # 如果模块路径不存在，直接返回 None
            if find_spec(module_path) is None:
                logger.error(f"Module {module_path} not found")
                return None
            # 否则再试一次导入
            try:
                module = importlib.import_module(module_path)
            except Exception as e2:
                logger.error(f"Failed to import module {module_path}: {e2}")
                return None
        
        # 尝试获取类对象
        try:
            cls = getattr(module, class_name)
            return cls
        except AttributeError:
            logger.error(f"Class {class_name} not found in module {module_path}")
            return None
    except Exception as e:
        logger.error(f"Failed to import class {class_path}: {e}")
        return None
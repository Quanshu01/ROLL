"""
Compatibility shim for historical `safe_import` module name.
Re-exports functions from `roll.utils.import_utils` to avoid import errors
from code that still expects `roll.utils.safe_import`.
"""
from .import_utils import safe_import_class, can_import_class, is_vllm_available

__all__ = ["safe_import_class", "can_import_class", "is_vllm_available"]

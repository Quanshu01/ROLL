from typing import List, Dict, Any
import os

import ray

from roll.pipeline.agentic.llm_proxy import BaseLLMProxy, register_llm_proxy
from roll.distributed.scheduler.protocol import DataProto


@register_llm_proxy("policy")
class PolicyProxy(BaseLLMProxy):
    """
    A proxy for policy model that invokes the policy model's engine (e.g. vllm/sglang) to perform generation.
    """

    def generate(self,
                 messages: List[Dict[str, str]],
                 lm_input: DataProto,
                 generation_config: Dict[str, Any]) -> DataProto:

        lm_input.meta_info["generation_config"] = generation_config
        lm_input.meta_info['response_callback_fn'] = self.generate_scheduler.report_response.remote
        lm_input.meta_info["pad_to_seq_len"] = False
        
        # 添加超时机制，默认 300 秒（5分钟），可通过环境变量配置
        timeout = int(os.environ.get("ROLL_GENERATE_TIMEOUT", "300"))
        try:
            lm_output: DataProto = ray.get(
                self.generate_scheduler.generate_one_request.remote(data=lm_input),
                timeout=timeout
            )
        except ray.exceptions.GetTimeoutError:
            # 超时后返回 None，让上层处理
            import logging
            logging.getLogger(__name__).warning(f"Generation request timed out after {timeout} seconds")
            return None
        except (ray.exceptions.ActorUnavailableError, ray.exceptions.RayActorError) as e:
            # 推理引擎可能已经崩溃，记录错误并返回 None
            import logging
            logging.getLogger(__name__).error(f"Inference engine unavailable: {e}. The engine may have crashed (e.g., OOM).")
            return None
        except Exception as e:
            # 其他异常也记录并返回 None
            import logging
            logging.getLogger(__name__).error(f"Unexpected error during generation: {e}", exc_info=True)
            return None

        if lm_output is not None:
            lm_output.meta_info.pop("generation_config", None)

        return lm_output

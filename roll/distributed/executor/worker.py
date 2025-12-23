import logging
import os
import socket
import time
from concurrent import futures
from dataclasses import dataclass
from typing import Dict

import ray

from roll.configs.worker_config import WorkerConfig
from roll.distributed.scheduler.decorator import Dispatch, register
from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.scheduler.storage import SharedStorage
from roll.utils.checkpoint_manager import download_model
from roll.utils.constants import RAY_NAMESPACE, STORAGE_NAME
from roll.utils.context_managers import state_offload_manger
from roll.utils.logging import get_logger
from roll.utils.network_utils import collect_free_port, get_node_ip
from roll.utils.offload_states import OffloadStateType
from roll.platforms import current_platform


@dataclass
class RankInfo:
    world_size: int = 1
    tp_size: int = 1
    dp_size: int = 1
    pp_size: int = 1
    cp_size: int = 1

    rank: int = 0
    tp_rank: int = 0
    dp_rank: int = 0
    pp_rank: int = 0
    cp_rank: int = 0

    @property
    def is_pipeline_last_stage(self):
        return self.pp_rank == (self.pp_size - 1)


class Worker:

    def __init__(self, worker_config: WorkerConfig):
        self.worker_config = worker_config
        self.pipeline_config = None
        self.worker_name = os.environ.get("WORKER_NAME", None)
        self.cluster_name = os.environ.get("CLUSTER_NAME", None)
        self.rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.shared_storage = SharedStorage.options(
            name=STORAGE_NAME, get_if_exists=True, namespace=RAY_NAMESPACE
        ).remote()

        # Coordination behavior:
        # - rank 0: choose MASTER_ADDR/MASTER_PORT (respect external if provided),
        #           verify port can be bound and fall back to free port if needed,
        #           then write mapping to SharedStorage for others to read.
        # - non-zero ranks: wait for leader to write MASTER_ADDR/MASTER_PORT into SharedStorage
        #                  and adopt those values before continuing.
        if self.rank == 0:
            master_addr = os.environ.get("MASTER_ADDR", self.get_node_ip())
            master_port_env = os.environ.get("MASTER_PORT")

            if master_port_env is not None:
                # honor externally-provided port but ensure it's bindable; if not,
                # fall back to allocating a free port and log a warning.
                try_port = int(master_port_env)
                bind_ok = False
                try:
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    s.bind((master_addr, try_port))
                    s.listen(1)
                    s.close()
                    bind_ok = True
                except Exception:
                    bind_ok = False

                if not bind_ok:
                    self.logger.warning(f"external MASTER_PORT={try_port} on {master_addr} is not bindable; allocating a different port")
                    master_port = str(self.get_free_port())
                else:
                    master_port = str(try_port)
                    # reserve the externally-provided port in SharedStorage so
                    # other leaders/allocators do not pick it.
                    try:
                        master_addr_port_key = f"MASTER_ADDR_PORT:{master_addr}:{master_port}"
                        ray.get(self.shared_storage.put.remote(master_addr_port_key, True))
                        self.logger.warning(f"reserved MASTER_ADDR_PORT key={master_addr_port_key} for external port")
                    except Exception:
                        self.logger.exception("failed to reserve external MASTER_ADDR_PORT in SharedStorage")
            else:
                master_port = str(self.get_free_port())

            os.environ["MASTER_ADDR"] = master_addr
            os.environ["MASTER_PORT"] = master_port

            # commit mapping for other workers
            self.master_addr = master_addr
            self.master_port = int(master_port)
            # write below after diagnostics
        else:
            # non-leader: wait until leader publishes MASTER_ADDR/MASTER_PORT
            wait_timeout = int(os.environ.get("MASTER_PUBLISH_TIMEOUT", 60))
            poll_interval = 0.5
            waited = 0.0
            mapping = None
            try:
                while waited < wait_timeout:
                    mapping = ray.get(self.shared_storage.get.remote(self.cluster_name))
                    if mapping is not None and mapping.get("MASTER_ADDR") is not None and mapping.get("MASTER_PORT") is not None:
                        break
                    time.sleep(poll_interval)
                    waited += poll_interval
            except Exception:
                self.logger.exception("error while waiting for leader to publish MASTER_ADDR/MASTER_PORT")

            if mapping is None:
                # fallback to environment values if SharedStorage not populated in time
                self.logger.warning(f"did not observe leader mapping in SharedStorage within {wait_timeout}s; falling back to environment MASTER_ADDR/MASTER_PORT")
                self.master_addr = os.environ.get("MASTER_ADDR", self.get_node_ip())
                self.master_port = int(os.environ.get("MASTER_PORT", 0) or 0)
            else:
                self.master_addr = mapping.get("MASTER_ADDR")
                self.master_port = int(mapping.get("MASTER_PORT"))
        # Additional debug: log the environment variables that affect rendezvous
        # This helps detect mismatches between actors (e.g., some using 127.0.0.1
        # while others use a physical IP).
        try:
            _nccl_if = os.environ.get("NCCL_SOCKET_IFNAME")
            _gloo_if = os.environ.get("GLOO_SOCKET_IFNAME")
            _tp_if = os.environ.get("TP_SOCKET_IFNAME")
        except Exception:
            _nccl_if = _gloo_if = _tp_if = None
        # Use WARNING so this shows up in actor logs regardless of INFO filtering.
        self.logger.warning(
            f"Worker init env: WORKER_NAME={self.worker_name} RANK={self.rank} MASTER_ADDR={self.master_addr} MASTER_PORT={self.master_port} NCCL_SOCKET_IFNAME={_nccl_if} GLOO_SOCKET_IFNAME={_gloo_if} TP_SOCKET_IFNAME={_tp_if}"
        )
        # Dump a concise environment snapshot to a centralized file for
        # cross-actor comparison and debugging. Filter out extremely long
        # values to keep log readable.
        try:
            dump_path = os.path.join(os.getcwd(), "output", "logs", "actor_env_dumps.log")
            with open(dump_path, "a", encoding="utf-8") as f:
                f.write(f"[{self.worker_name} pid={os.getpid()}] MASTER_ADDR={self.master_addr} MASTER_PORT={self.master_port}\n")
                for k in sorted(os.environ.keys()):
                    v = os.environ.get(k, "")
                    if len(v) > 200:
                        v = v[:200] + "..."
                    f.write(f"[{self.worker_name} pid={os.getpid()}] {k}={v}\n")
                f.write("\n")
        except Exception:
            # best-effort; do not raise during worker init
            self.logger.exception("failed to write actor env dump")

        # Quick connectivity diagnostics: try connecting to MASTER_ADDR:MASTER_PORT
        # (short timeout) and attempt a short bind test (best-effort).
        try:
            diag_msg = []
            # TCP connect test
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(0.5)
                s.connect((self.master_addr, int(self.master_port)))
                s.close()
                diag_msg.append(f"tcp_connect: success -> {self.master_addr}:{self.master_port}")
            except Exception as e:
                diag_msg.append(f"tcp_connect: fail -> {self.master_addr}:{self.master_port} err={e}")

            # local bind test (non-blocking, will close immediately) - best-effort
            try:
                b = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                b.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                b.bind((self.master_addr, int(self.master_port)))
                b.listen(1)
                b.close()
                diag_msg.append(f"local_bind: success -> {self.master_addr}:{self.master_port}")
            except Exception as e:
                diag_msg.append(f"local_bind: fail -> {self.master_addr}:{self.master_port} err={e}")

            for m in diag_msg:
                self.logger.warning(m)
                try:
                    with open(dump_path, "a", encoding="utf-8") as f:
                        f.write(f"[{self.worker_name} pid={os.getpid()}] {m}\n")
                except Exception:
                    pass
        except Exception:
            self.logger.exception("connectivity diagnostics failed")
        # leader should publish mapping (non-leaders may also re-put their view - cheap)
        try:
            self.shared_storage.put.remote(
                self.cluster_name, {"MASTER_ADDR": self.master_addr, "MASTER_PORT": self.master_port}
            )
        except Exception:
            self.logger.exception("failed to put MASTER_ADDR/MASTER_PORT to SharedStorage")
        # NOTE: 自定义Worker时根据需要配置rank_info
        self.rank_info = RankInfo(
            world_size=self.world_size,
            rank=self.rank,
            dp_rank=self.rank,
            dp_size=self.world_size,
        )
        self.thread_executor: futures.ThreadPoolExecutor = futures.ThreadPoolExecutor(max_workers=5)
        self._logger = None

    def __repr__(self):
        return f"{type(self).__name__}({self.worker_name})"

    @property
    def logger(self) -> logging.Logger:
        """
        在ray.Actor内要使用自定义的logger, 避免ray context造成的logger不一致
        """
        self._logger = get_logger()
        return self._logger

    @staticmethod
    def get_node_ip():
        return get_node_ip()

    @staticmethod
    def get_free_port():
        shared_storage = SharedStorage.options(
            name=STORAGE_NAME, get_if_exists=True, namespace=RAY_NAMESPACE
        ).remote()
        master_addr = Worker.get_node_ip()
        max_retry_count = int(os.environ.get("MAX_PORT_RETRY_COUNT", 1000))
        retry_count = 0
        master_port = collect_free_port()
        # diagnostic logger for port allocation
        _logger = get_logger()
        _logger.warning(f"allocating MASTER_PORT for master_addr={master_addr}; initial_port={master_port}")
        while retry_count < max_retry_count:
            master_addr_port_key = f"MASTER_ADDR_PORT:{master_addr}:{master_port}"
            _logger.warning(f"testing MASTER_ADDR_PORT key={master_addr_port_key} (attempt {retry_count})")
            if ray.get(shared_storage.get.remote(master_addr_port_key)) is None:
                ray.get(shared_storage.put.remote(master_addr_port_key, True))
                _logger.warning(f"reserved MASTER_ADDR_PORT key={master_addr_port_key}")
                break
            master_port = collect_free_port()
            retry_count += 1
        if retry_count >= max_retry_count:
            raise RuntimeError(f"Can not allocate unique MASTER_PORT on {master_addr}.")
        return master_port

    def get_master_addr_and_port(self):
        return self.master_addr, self.master_port

    @staticmethod
    def get_visible_gpus():
        return current_platform.get_visible_gpus()

    def get_devices_info(self):
        devices_info = [
            dict(rank=rank, node_rank=pg["node_rank"], gpu_rank=pg["gpu_rank"])
            for rank, pg in enumerate(self.worker_config.resource_placement_groups)
        ]
        return devices_info

    def get_rank_info(self):
        return self.rank_info

    def initialize(self, pipeline_config, *args, **kwargs):
        self.pipeline_config = pipeline_config

        model_name = self.worker_config.model_args.model_name_or_path
        if model_name:
            self.worker_config.model_args.model_name_or_path = download_model(model_name)

        if self.pipeline_config.resume_from_checkpoint:
            self.logger.info(f"resume_from_checkpoint: {self.pipeline_config.resume_from_checkpoint}")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_states(self, *args, **kwargs):
        if getattr(self, "strategy", None) is not None:
            self.strategy.load_states()
        else:
            self.logger.warning("worker has not strategy")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def offload_states(self, *args, **kwargs):
        if getattr(self, "strategy", None) is not None:
            self.strategy.offload_states()
        else:
            self.logger.warning("worker has not strategy")

    def broadcast_parameter(self, *args, **kwargs):
        if getattr(self, "strategy", None) is not None:
            self.strategy.broadcast_parameter(*args, **kwargs)
        else:
            self.logger.warning("worker has not strategy")

    def broadcast_bucket(self, *args, **kwargs):
        if getattr(self, "strategy", None) is not None:
            self.strategy.broadcast_bucket(*args, **kwargs)
        else:
            self.logger.warning("worker has not strategy")

    def setup_collective_group(self, *args, **kwargs):
        if getattr(self, "strategy", None) is not None:
            self.strategy.setup_collective_group(*args, **kwargs)
        else:
            self.logger.warning("worker has not strategy")

    def start_model_update(self, *args, **kwargs):
        metrics = {}
        if getattr(self, "strategy", None) is not None:
            with state_offload_manger(
                strategy=self.strategy,
                metrics=metrics,
                metric_infix=f"{self.cluster_name}/model_update",
                load_kwargs={"include": [OffloadStateType.model_params]},
            ):
                exec_metrics: Dict = self.strategy.model_update(*args, **kwargs)
            metric_prefix = f"time/{self.cluster_name}/model_update"
            metrics.update({f"{metric_prefix}/{k}": v for k, v in exec_metrics.items()})
        else:
            self.logger.warning("worker has not strategy")

        output = DataProto(meta_info={"metrics": metrics})
        return output

    def update_parameter(self, *args, **kwargs):
        if getattr(self, "strategy", None) is not None:
            self.strategy.update_parameter(*args, **kwargs)
        else:
            self.logger.warning("worker has not strategy")

    def update_parameter_in_bucket(self, *args, **kwargs):
        if getattr(self, "strategy", None) is not None:
            self.strategy.update_parameter_in_bucket(*args, **kwargs)
        else:
            self.logger.warning("worker has not strategy")

    def add_lora(self, *args, **kwargs):
        if getattr(self, "strategy", None) is not None:
            self.strategy.add_lora(*args, **kwargs)
        else:
            self.logger.warning("worker has not strategy")

    def download_models(self, model_name_or_paths: set[str]):
        futures.wait([self.thread_executor.submit(download_model, model_name_or_path)
                      for model_name_or_path in model_name_or_paths])

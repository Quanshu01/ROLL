import sys
import os
import logging
import json
import random
import re
import subprocess
import socket
import struct
import time
import atexit
from typing import Optional, Dict, Any

import gem
from gem import Env
import threading

logger = logging.getLogger(__name__)

def send_msg(sock, msg):
    # Prefix each message with a 4-byte length (network byte order)
    msg = json.dumps(msg).encode('utf-8')
    msg = struct.pack('>I', len(msg)) + msg
    sock.sendall(msg)

def recv_msg(sock):
    # Read message length and unpack it into an integer
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    # Read the message data
    return recvall(sock, msglen)

def recvall(sock, n):
    # Helper function to recv n bytes or return None if EOF is hit
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data

class OSWorldEnv(Env):
    def __init__(self,
                 conda_env_name="osworld",
                 provider_name="docker",
                 observation_type="a11y_tree",
                 action_space="pyautogui",
                 headless=True,
                 screen_width=1920,
                 screen_height=1080,
                 sleep_after_execution=0.0,
                 max_steps=15,
                 task_config_path="evaluation_examples/test_small_os.json",
                 path_to_vm: str = None,
                 instance_name: str = None,
                 image_name: str = None,
                 client_password: str = None,
                 use_gpt_eval: bool = False,
                 **kwargs
                 ):
        self.conda_env_name = conda_env_name
        self.provider_name = provider_name
        self.observation_type = observation_type
        self.action_space = action_space
        self.headless = headless
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.sleep_after_execution = sleep_after_execution
        self.max_steps = max_steps
        self.task_config_path = task_config_path
        self.use_gpt_eval = use_gpt_eval
        
        self.server_process = None
        self.sock = None
        self.path_to_vm = path_to_vm
        self.instance_name = instance_name
        self.image_name = image_name
        # allow overriding default client_password
        if client_password:
            self.client_password = client_password
        
        self._start_server()
        atexit.register(self.close)

    def _start_server(self):
        server_script = os.path.join(os.path.dirname(__file__), "server.py")
        
        # Prefer invoking the environment's python binary directly if available
        # (avoids `conda run` wrapping which may capture stdout/stderr). If that
        # python is not present, fall back to `conda run --no-capture-output`.
        # Try to detect a usable python executable in common conda locations.
        # Prefer an interpreter that can `import playwright` to avoid runtime
        # ImportError when the server imports desktop_env.
        # Prefer the qs-osworld environment python when available (explicit)
        candidates = [
            "/data/share/miniconda3/envs/osworld/bin/python",
            "/data/share/projects/quanshu/envs/qs-osworld/bin/python",
            f"/data/share/miniconda3/envs/{self.conda_env_name}/bin/python",
            f"/data/share/projects/quanshu/envs/{self.conda_env_name}/bin/python",
        ]

        # Also search other envs under /data/share/projects/quanshu/envs
        envs_root = "/data/share/projects/quanshu/envs"
        if os.path.isdir(envs_root):
            for entry in os.listdir(envs_root):
                py = os.path.join(envs_root, entry, "bin", "python")
                if py not in candidates:
                    candidates.append(py)

        env_python = None
        for py in candidates:
            try:
                if os.path.exists(py):
                    # Quick check whether this python can import playwright
                    check = subprocess.run([py, "-c", "import playwright"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
                    if check.returncode == 0:
                        env_python = py
                        break
            except Exception:
                continue

        if env_python is not None:
            cmd = [
                env_python, server_script,
                "--provider_name", self.provider_name,
                "--observation_type", self.observation_type,
                "--action_space", self.action_space,
                "--screen_width", str(self.screen_width),
                "--screen_height", str(self.screen_height),
                "--sleep_after_execution", str(self.sleep_after_execution),
                "--max_steps", str(self.max_steps),
                "--task_config_path", self.task_config_path
            ]
        else:
            # Fallback to conda run; if playwright is missing the server will
            # still raise informative ImportError which we will log.
            cmd = [
                "conda", "run", "-n", self.conda_env_name, "--no-capture-output",
                "python", server_script,
                "--provider_name", self.provider_name,
                "--observation_type", self.observation_type,
                "--action_space", self.action_space,
                "--screen_width", str(self.screen_width),
                "--screen_height", str(self.screen_height),
                "--sleep_after_execution", str(self.sleep_after_execution),
                "--max_steps", str(self.max_steps),
                "--task_config_path", self.task_config_path
            ]
        
        if self.headless:
            cmd.append("--headless")

        # If a VNC connection string was provided, pass it to the server so
        # the VNC provider can connect to the remote VM.
        if self.path_to_vm:
            cmd.extend(["--path_to_vm", str(self.path_to_vm)])

        # Pass instance/image/client_password if provided in env config
        if getattr(self, 'instance_name', None):
            cmd.extend(["--instance_name", str(self.instance_name)])
        if getattr(self, 'image_name', None):
            cmd.extend(["--image_name", str(self.image_name)])
        if getattr(self, 'client_password', None):
            cmd.extend(["--client_password", str(self.client_password)])
        
        # Pass use_gpt_eval to enable reward and cost evaluation
        if self.use_gpt_eval:
            cmd.append("--use_gpt_eval")
            
        logger.info(f"Starting OSWorld server with command: {' '.join(cmd)}")
        
        # 将 server 输出写入文件，同时保留 stdout/stderr 管道用于读取端口与日志
        # 选择顺序：环境变量 OSWORLD_SERVER_LOG_PATH > OSWORLD_SERVER_LOG_DIR > env_config.server_log_path > env_config.server_log_dir > /tmp
        log_path = None
        env_log_path = os.environ.get("OSWORLD_SERVER_LOG_PATH")
        env_log_dir = os.environ.get("OSWORLD_SERVER_LOG_DIR")
        if env_log_path:
            log_path = env_log_path
        elif env_log_dir:
            ts = time.strftime("%Y%m%d-%H%M%S")
            log_path = os.path.join(env_log_dir, f"osworld_server_{ts}.log")
        if not log_path:
            log_path = getattr(self, "server_log_path", None)
        if not log_path:
            log_dir = getattr(self, "server_log_dir", None)
            if log_dir:
                ts = time.strftime("%Y%m%d-%H%M%S")
                log_path = os.path.join(log_dir, f"osworld_server_{ts}.log")
            else:
                log_path = "/tmp/osworld_server.log"
        # 统一转为绝对路径，避免相对路径在不同工作目录下丢失
        log_path = os.path.abspath(log_path)
        try:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
        except Exception:
            # 如果路径不可创建，回退 /tmp
            log_path = "/tmp/osworld_server.log"
        log_fh = open(log_path, "a", buffering=1)
        
        # Ensure HOME environment variable is valid before starting server
        # This prevents errors like "52.9/home/vipuser" from being passed to subprocess
        env = os.environ.copy()
        home = env.get('HOME', '')
        default_home = '/home/vipuser'
        
        # Validate HOME: must be absolute path, start with '/', and not contain invalid patterns
        is_valid_home = (
            home and
            os.path.isabs(home) and
            home.startswith('/') and
            # Reject patterns like "52.9/home/vipuser" or "75.8/home/vipuser" (number.number/...)
            not re.match(r'^\d+\.\d+/', home) and
            # Reject patterns that look like IP addresses (e.g., "192.168.1.1")
            not re.match(r'^\d+\.\d+\.\d+\.\d+', home)
        )
        
        if not is_valid_home:
            old_home = home if home else '(not set)'
            env['HOME'] = default_home
            logger.warning(
                f"HOME environment variable was invalid (value: '{old_home}'), "
                f"setting to '{default_home}' for OSWorld server subprocess"
            )
        else:
            # Ensure HOME is set even if it was valid
            env['HOME'] = home
        
        self.server_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd="/data/share/projects/quanshu/OSWorld-dev",
            env=env  # Pass corrected environment
        )
        logger.info(f"OSWorld server stdout/stderr will be tee'd to {log_path}")
        
        # Read port from stdout
        port = None
        start_time = time.time()
        while time.time() - start_time < 60: # Wait up to 60 seconds
            line = self.server_process.stdout.readline()
            if line:
                log_fh.write(line)
            if "OSWORLD_SERVER_PORT:" in line:
                port = int(line.strip().split(":")[1])
                break
            if line:
                logger.info(f"Server stdout: {line.strip()}")
        
        if port is None:
            stderr = self.server_process.stderr.read()
            raise RuntimeError(f"Failed to start OSWorld server. Stderr: {stderr}")
            
        logger.info(f"OSWorld server started on port {port}")
        
        # Connect to server
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Retry connection
        for _ in range(10):
            try:
                self.sock.connect(('localhost', port))
                break
            except ConnectionRefusedError:
                time.sleep(1)
        else:
             raise RuntimeError("Could not connect to OSWorld server")

        # Start background threads to stream server stdout/stderr into our logger
        def _drain_stream(stream, label):
            try:
                for ln in iter(stream.readline, ''):
                    if not ln:
                        break
                    text = ln.rstrip()
                    # 写入文件，保持原始输出
                    log_fh.write(ln)
                    # If the child process already prefixes its lines with a
                    # level (e.g. "INFO:..."), map that to our logger levels
                    # so the output appears with the correct severity.
                    if text.startswith('ERROR') or text.startswith('ERROR:'):
                        logger.error('%s: %s', label, text)
                    elif text.startswith('WARNING') or text.startswith('WARN') or text.startswith('WARNING:'):
                        logger.warning('%s: %s', label, text)
                    elif text.startswith('DEBUG') or text.startswith('DEBUG:'):
                        logger.debug('%s: %s', label, text)
                    else:
                        # Default to INFO for regular stdout lines so they
                        # don't show up as ERROR in the worker logs.
                        logger.info('%s: %s', label, text)
            except Exception:
                logger.exception('Error while draining server stream')

        threading.Thread(target=_drain_stream, args=(self.server_process.stdout, 'OSWorld stdout'), daemon=True).start()
        threading.Thread(target=_drain_stream, args=(self.server_process.stderr, 'OSWorld stderr'), daemon=True).start()

    def _send_request(self, method, **kwargs):
        req = {'method': method}
        if 'args' in kwargs:
            req['args'] = kwargs['args']
        if 'kwargs' in kwargs:
            req['kwargs'] = kwargs['kwargs']
            
        send_msg(self.sock, req)
        resp_data = recv_msg(self.sock)
        if not resp_data:
            raise RuntimeError("Server closed connection")
        raw = resp_data.decode('utf-8')
        logger.debug(f"OSWorld raw response: {raw}")
        try:
            resp = json.loads(raw)
        except Exception as e:
            logger.error(f"Failed to json-decode OSWorld response: {e}; raw={raw}")
            raise

        # Defensive checks: log the full response when it contains an error
        if isinstance(resp, dict) and 'error' in resp:
            logger.error(f"OSWorld returned error response: {resp}")
            raise RuntimeError(f"Server error: {resp['error']}")

        if not isinstance(resp, dict) or 'result' not in resp:
            logger.error(f"Unexpected OSWorld response format: {resp}")
            raise RuntimeError(f"Unexpected response from OSWorld server: {resp}")

        return resp['result']

    def reset(self, seed=None):
        max_retries = int(os.environ.get('OSWORLD_RESET_MAX_RETRIES', 3))
        backoff_base = float(os.environ.get('OSWORLD_RESET_BACKOFF_BASE', 2.0))
        last_exc = None
        for attempt in range(1, max_retries + 1):
            try:
                return self._send_request('reset', kwargs={'seed': seed})
            except RuntimeError as e:
                msg = str(e)
                # 针对远端 HTTP 超时等可重试错误进行退避重试
                if 'Read timed out' in msg or 'Timeout' in msg or 'closed connection' in msg:
                    last_exc = e
                    sleep_s = backoff_base ** attempt
                    logger.warning(f"OSWorld reset attempt {attempt}/{max_retries} failed: {msg}; retrying in {sleep_s:.1f}s...")
                    time.sleep(min(30, sleep_s))
                    continue
                # 非可重试错误直接抛出
                raise
        logger.error(f"OSWorld reset exhausted retries ({max_retries}); last error: {last_exc}")
        raise last_exc or RuntimeError('OSWorld reset failed')

    def step(self, action: str):
        return self._send_request('step', args=[action])

    def get_instructions(self) -> str:
        return self._send_request('get_instructions')
    
    def get_vm_info(self):
        """Get VM connection information for pytest runner"""
        return self._send_request('get_vm_info')
    
    def get_task_info(self):
        """Get current task information including task ID"""
        return self._send_request('get_current_task')

    def close(self):
        if self.sock:
            try:
                self._send_request('close')
            except:
                pass
            self.sock.close()
            self.sock = None
            
        if self.server_process:
            self.server_process.terminate()
            self.server_process = None

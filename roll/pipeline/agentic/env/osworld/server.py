import sys
import os
import json
import socket
import struct
import traceback
import logging
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("OSWorldServer")

# Add OSWorld to path
OSWORLD_PATH = "/data/share/projects/quanshu/OSWorld-dev"
if OSWORLD_PATH not in sys.path:
    sys.path.append(OSWORLD_PATH)

try:
    from desktop_env.desktop_env import DesktopEnv
    from mm_agents.agent import linearize_accessibility_tree, parse_code_from_string
except ImportError:
    logger.error(f"Could not import OSWorld modules from {OSWORLD_PATH}")
    sys.exit(1)

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

class OSWorldServer:
    def __init__(self, host='localhost', port=0, 
                 provider_name="docker",
                 observation_type="a11y_tree",
                 action_space="pyautogui",
                 headless=True,
                 screen_width=1920,
                 screen_height=1080,
                 sleep_after_execution=0.0,
                 max_steps=15,
                 task_config_path="evaluation_examples/test_small_os.json",
                 path_to_vm=None,
                 instance_name=None,
                 image_name=None,
                 client_password=None,
                 use_gpt_eval=False):
        
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind((host, port))
        self.port = self.sock.getsockname()[1]
        self.sock.listen(1)
        
        self.provider_name = provider_name
        self.observation_type = observation_type
        self.action_space = action_space
        self.headless = headless
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.sleep_after_execution = sleep_after_execution
        self.max_steps = max_steps
        self.task_config_path = os.path.join(OSWORLD_PATH, task_config_path)
        
        # Load tasks. Support multiple common formats:
        # - A list of task objects (JSON list)
        # - A dict like {"os": ["id1","id2"]} referencing example IDs
        # - A dict with another group key whose value is a list of IDs
        # - A dict mapping names->task objects
        if os.path.exists(self.task_config_path):
            with open(self.task_config_path, "r") as f:
                loaded = json.load(f)

            tasks = []

            # Case 1: loaded is a list of full task objects
            if isinstance(loaded, list):
                tasks = loaded

            # Case 2: loaded is a dict with key 'os' listing example IDs
            elif isinstance(loaded, dict) and 'os' in loaded and isinstance(loaded['os'], list):
                ids = loaded['os']
                for tid in ids:
                    task_file = os.path.join(OSWORLD_PATH, 'evaluation_examples', 'examples', 'os', f'{tid}.json')
                    if os.path.exists(task_file):
                        try:
                            with open(task_file, 'r') as tf:
                                tasks.append(json.load(tf))
                        except Exception:
                            logger.warning(f"Failed to load task example file {task_file}; skipping")
                    else:
                        logger.warning(f"Task example file not found: {task_file}; skipping")

            # Case 3: loaded is a dict with some group key whose value is a list of IDs
            elif isinstance(loaded, dict):
                # Try to find the first value that's a list of strings (IDs)
                ids = None
                for v in loaded.values():
                    if isinstance(v, list) and v and all(isinstance(x, str) for x in v):
                        ids = v
                        break

                if ids is not None:
                    # Determine the subdirectory based on the config key
                    # If key is 'linux_pytest', look in 'linux_pytest' subdirectory
                    subdir = None
                    for key in loaded.keys():
                        if key in ['linux_pytest', 'windows_pytest', 'os', 'malicious_tests', 'tb_tasks']:
                            subdir = key
                            break
                    
                    for tid in ids:
                        # Try multiple possible locations for task files
                        possible_paths = [
                            os.path.join(OSWORLD_PATH, 'evaluation_examples', 'examples', 'os', f'{tid}.json'),
                            os.path.join(OSWORLD_PATH, 'evaluation_examples', 'examples', 'malicious_tests', f'{tid}.json'),
                            os.path.join(OSWORLD_PATH, 'evaluation_examples', 'examples', 'tb_tasks', f'{tid}.json'),
                            os.path.join(OSWORLD_PATH, 'evaluation_examples', 'examples', 'linux_pytest', f'{tid}.json'),
                            os.path.join(OSWORLD_PATH, 'evaluation_examples', 'examples', 'windows_pytest', f'{tid}.json'),
                        ]
                        # If we identified a specific subdirectory, prioritize it
                        if subdir:
                            subdir_path = os.path.join(OSWORLD_PATH, 'evaluation_examples', 'examples', subdir, f'{tid}.json')
                            if subdir_path not in possible_paths:
                                possible_paths.insert(0, subdir_path)
                            else:
                                # Move it to the front
                                possible_paths.remove(subdir_path)
                                possible_paths.insert(0, subdir_path)
                        
                        task_file = None
                        for path in possible_paths:
                            if os.path.exists(path):
                                task_file = path
                                break
                        
                        if task_file:
                            try:
                                with open(task_file, 'r') as tf:
                                    task_data = json.load(tf)
                                    tasks.append(task_data)
                                    logger.info(f"Loaded task {tid} from {task_file}, has step_pytest: {'step_pytest' in task_data}")
                            except Exception as e:
                                logger.warning(f"Failed to load task example file {task_file}: {e}; skipping")
                        else:
                            logger.warning(f"Task example file not found for {tid} in any of the checked paths: {possible_paths}")
                else:
                    # If the dict appears to already contain task objects (mapping name->task),
                    # use its values as the task list.
                    if all(isinstance(v, dict) for v in loaded.values()):
                        tasks = list(loaded.values())
                    else:
                        logger.warning(f"Unrecognized task config format in {self.task_config_path}; using empty task list.")

            if not tasks:
                logger.warning(f"No valid tasks were loaded from {self.task_config_path}; attempting to load all examples from evaluation_examples/examples/os as a fallback.")
                # Fallback: try loading all example files under evaluation_examples/examples/os
                examples_dir = os.path.join(OSWORLD_PATH, 'evaluation_examples', 'examples', 'os')
                if os.path.isdir(examples_dir):
                    for fn in os.listdir(examples_dir):
                        if fn.endswith('.json'):
                            fp = os.path.join(examples_dir, fn)
                            try:
                                with open(fp, 'r') as tf:
                                    tasks.append(json.load(tf))
                            except Exception:
                                logger.warning(f"Failed to load fallback example file {fp}; skipping")
                    if tasks:
                        logger.info(f"Loaded {len(tasks)} fallback tasks from {examples_dir}")
                    else:
                        logger.warning(f"Fallback load found no example files in {examples_dir}")
                else:
                    logger.warning(f"Examples directory {examples_dir} does not exist; no fallback available.")
            self.tasks = tasks
        else:
            logger.warning(f"Task config path {self.task_config_path} does not exist. Using empty task list.")
            self.tasks = []
            
        self.path_to_vm = path_to_vm
        self.instance_name = instance_name
        self.image_name = image_name
        self.client_password = client_password
        self.use_gpt_eval = use_gpt_eval

        # Construct snapshot_name expected by VNC provider: 'instance:image'
        # If both instance_name and image_name are provided, use the combined form.
        # Otherwise, leave default snapshot behavior (DesktopEnv default).
        snapshot_name = None
        if self.instance_name and self.image_name:
            snapshot_name = f"{self.instance_name}:{self.image_name}"

        desktop_kwargs = {
            'path_to_vm': self.path_to_vm,
            'provider_name': provider_name,
            'action_space': action_space,
            'screen_size': (screen_width, screen_height),
            'headless': headless,
            # NOTE:
            # - 对于 a11y_tree/screenshot_a11y_tree/som，本来就需要可访问性树；
            # - 对于 terminal，我们也强制拉取 a11y_tree，便于在 terminal 文本缺失时做回退，
            #   否则 obs 里往往只有一个 terminal=None，导致上游拿到的 observation 形同空串，
            #   最终 actor_infer 出现 "Processed prompts: 0it"。
            'require_a11y_tree': (observation_type in ["a11y_tree", "screenshot_a11y_tree", "som", "terminal"]),
            'require_terminal': (observation_type == "terminal"),
            'use_gpt_eval': False,  # Disable GPT eval, use rule-based and pytest instead
        }
        if snapshot_name is not None:
            desktop_kwargs['snapshot_name'] = snapshot_name
        if self.instance_name is not None:
            desktop_kwargs['instance_name'] = self.instance_name
        if self.client_password is not None:
            desktop_kwargs['client_password'] = self.client_password

        try:
            self.env = DesktopEnv(**desktop_kwargs)
            logger.info(f"DesktopEnv initialized successfully with use_gpt_eval={self.use_gpt_eval}")
        except TimeoutError as e:
            logger.error(f"VNC connection timeout: {e}")
            logger.error("This usually means the remote VNC server is not accessible.")
            logger.error("Please check:")
            logger.error("  1. Network connectivity to the VNC server")
            logger.error("  2. VNC server is running and accessible")
            logger.error("  3. Firewall rules allow connection")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize DesktopEnv: {e}")
            logger.error(f"DesktopEnv kwargs: {desktop_kwargs}")
            raise
        
        self.current_task = None
        self.step_count = 0

    def run(self):
        # Print the port so the parent process can read it
        print(f"OSWORLD_SERVER_PORT:{self.port}", flush=True)
        
        conn, addr = self.sock.accept()
        logger.info(f"Connected by {addr}")
        with conn:
            while True:
                data = recv_msg(conn)
                if not data:
                    break
                
                try:
                    request = json.loads(data.decode('utf-8'))
                    method = request.get('method')
                    
                    if method == 'reset':
                        response = self.handle_reset(request.get('kwargs', {}))
                    elif method == 'step':
                        response = self.handle_step(request.get('args', []))
                    elif method == 'get_instructions':
                        response = {'result': self.current_task['instruction'] if self.current_task else ""}
                    elif method == 'get_current_task':
                        # 返回当前任务信息，用于获取任务ID
                        if self.current_task:
                            response = {'result': self.current_task if isinstance(self.current_task, dict) else {'id': None}}
                        else:
                            response = {'result': {'id': None}}
                    elif method == 'get_vm_info':
                        response = self.handle_get_vm_info()
                    elif method == 'close':
                        self.env.close()
                        response = {'result': 'closed'}
                        send_msg(conn, response)
                        break
                    else:
                        response = {'error': f'Unknown method: {method}'}
                        
                except Exception as e:
                    logger.error(f"Error handling request: {traceback.format_exc()}")
                    response = {'error': str(e)}
                
                send_msg(conn, response)

    def handle_reset(self, kwargs):
        try:
            seed = kwargs.get('seed')
            if seed is not None:
                random.seed(seed)

            if not self.tasks:
                return {'error': "No tasks available to run."}

            task_idx = 0
            if seed is not None:
                task_idx = seed % len(self.tasks)

            self.current_task = self.tasks[task_idx]

            # Log task summary for debugging
            try:
                logger.info(f"Resetting with task id={self.current_task.get('id', '<no-id>')} type={type(self.current_task)}")
            except Exception:
                logger.info(f"Resetting with task (repr): {repr(self.current_task)[:200]}")

            # Perform reset on underlying DesktopEnv
            self.env.reset(task_config=self.current_task)
            self.step_count = 0

            obs = self.env._get_obs()
            obs_text = self._process_obs(obs)

            # For reset, return (observation, info) — the env manager expects two items
            # Pass step_pytest config from task to info so traj_env_manager can use it
            info = {"env_instruction": self.current_task.get('instruction', "") if isinstance(self.current_task, dict) else ""}
            if isinstance(self.current_task, dict):
                # 将完整的任务信息传递给 info，方便 traj_env_manager 获取任务ID
                info['task'] = self.current_task
                info['task_id'] = self.current_task.get('id')
                if 'step_pytest' in self.current_task:
                    info['step_pytest'] = self.current_task['step_pytest']
            return {
                'result': (obs_text, info)
            }
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"Error handling reset: {tb}")
            # Include traceback in error string to aid debugging on client side
            return {'error': str(e) + '\n' + tb}

    def handle_get_vm_info(self):
        """Get VM connection information for pytest runner.

        优先返回 DesktopEnv 自带的 vm_ip/server_port。
        若没有，则尝试从 controller.http_server 解析 host/port，
        方便 step_pytest_runner 设置 OSWORLD_VM_IP/PORT 给 vm_client。
        """
        # Case 1: DesktopEnv 显式暴露 vm_ip/server_port
        if self.env and hasattr(self.env, 'vm_ip') and hasattr(self.env, 'server_port'):
            vm_ip = str(self.env.vm_ip).strip()
            server_port = self.env.server_port
            # 确保 server_port 是整数
            try:
                server_port = int(server_port)
            except (ValueError, TypeError):
                logger.warning(f"Invalid server_port type: {type(server_port)}, value: {server_port}")
                server_port = 5000  # 默认端口
            return {
                'result': {
                    'vm_ip': vm_ip,
                    'server_port': server_port
                }
            }

        # Case 2: 从 controller.http_server 解析
        try:
            if self.env and hasattr(self.env, 'controller'):
                http_server = getattr(self.env.controller, 'http_server', None)
                if http_server:
                    from urllib.parse import urlparse
                    parsed = urlparse(str(http_server).strip())
                    host = parsed.hostname or "localhost"
                    # 确保 port 是整数，如果解析失败则使用默认值
                    try:
                        port = int(parsed.port) if parsed.port else 5000
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid port in http_server: {http_server}, using default 5000")
                        port = 5000
                    return {
                        'result': {
                            'vm_ip': str(host).strip(),
                            'server_port': port
                        }
                    }
        except Exception as e:
            logger.warning(f"Failed to parse controller http_server for vm info: {e}")

        return {'result': None}

    def handle_step(self, args):
        action = args[0]
        self.step_count += 1
        
        logger.info(f"Step {self.step_count}: raw action: {action}")
        parsed_actions = parse_code_from_string(action)
        logger.info(f"Step {self.step_count}: parsed_actions: {parsed_actions}")

        total_reward = 0
        total_cost = 0
        done = False
        truncated = False
        info = {}

        obs = None

        def _unpack_step_result(res):
            # DesktopEnv.step() returns (obs, reward, cost, done, info)
            # Accept both 4-tuple (obs, reward, done, info) for compatibility
            # and 5-tuple (obs, reward, cost, done, info) from DesktopEnv
            # and 5-tuple (obs, reward, done, truncated, info) from gym-style envs
            nonlocal truncated
            cost = 0
            if isinstance(res, tuple) or isinstance(res, list):
                if len(res) == 4:
                    # (obs, reward, done, info) - gym-style without cost
                    o, r, d, inf = res
                    return o, r, cost, d, inf
                elif len(res) == 5:
                    # Check if it's DesktopEnv format (obs, reward, cost, done, info)
                    # or gym format (obs, reward, done, truncated, info)
                    # DesktopEnv format: 3rd element is cost (numeric or None), 4th is done (bool)
                    # Gym format: 3rd element is done (bool), 4th is truncated (bool)
                    # Check if 3rd element could be cost (numeric or None) and 4th is bool (done)
                    if (isinstance(res[2], (int, float, type(None))) and isinstance(res[3], bool) and 
                        isinstance(res[4], dict)):
                        # DesktopEnv format: (obs, reward, cost, done, info)
                        o, r, cost, d, inf = res
                        return o, r, cost, d, inf
                    else:
                        # Gym format: (obs, reward, done, truncated, info)
                        o, r, d, tr, inf = res
                        truncated = bool(tr)
                        return o, r, cost, d, inf
            # Fallback: assume (obs,) or single value
            return res, 0, 0, False, {}

        if parsed_actions:
            for act in parsed_actions:
                res = self.env.step(act, self.sleep_after_execution)
                obs, reward, cost, done, info = _unpack_step_result(res)
                total_reward += reward if reward is not None else 0
                total_cost += cost if cost is not None else 0
                if done:
                    break
        else:
             if action.strip() in ['WAIT', 'DONE', 'FAIL']:
                 res = self.env.step(action.strip(), self.sleep_after_execution)
                 obs, reward, cost, done, info = _unpack_step_result(res)
                 total_reward += reward if reward is not None else 0
                 total_cost += cost if cost is not None else 0
             else:
                 obs = self.env._get_obs()
                 total_reward = 0
                 total_cost = 0
        
        if self.step_count >= self.max_steps:
            done = True
            
        # If obs is None (e.g. loop didn't run), get it
        if obs is None:
             obs = self.env._get_obs()

        obs_text = self._process_obs(obs)
        
        # Extract cost from info if available (DesktopEnv puts it there)
        if 'cost' in info:
            total_cost = info.get('cost', total_cost)
        if 'reward' in info:
            total_reward = info.get('reward', total_reward)
        
        # Log reward and cost for debugging
        logger.info(f"Step {self.step_count}: reward={total_reward}, cost={total_cost}, info_keys={list(info.keys())}")
        if total_reward == 0 and total_cost == 0:
            logger.warning(f"Step {self.step_count}: reward and cost are both 0. "
                          f"Rule-based evaluator and pytest runner should populate these values in traj_env_manager.")
        
        # Ensure cost is in info for traj_env_manager to access
        if 'cost' not in info:
            info['cost'] = total_cost
        
        # Info might contain non-serializable objects, filter it if necessary
        # For now, assume info is JSON serializable or empty
        safe_info = {k: v for k, v in info.items() if isinstance(v, (str, int, float, bool, list, dict, type(None)))}

        # Ensure we return a 5-tuple: (observation, reward, done, truncated, info)
        # Note: cost is included in info dict for traj_env_manager to access
        return {
            'result': (obs_text, total_reward, done, bool(truncated), safe_info)
        }

    def _process_obs(self, obs):
        # 观测类型：只用可访问性树
        if self.observation_type == "a11y_tree":
            try:
                if not obs or not isinstance(obs, dict):
                    logger.warning("_process_obs: obs is empty or not a dict, falling back to str(obs)")
                    return str(obs)

                tree = obs.get("accessibility_tree")
                if not tree:
                    logger.warning("_process_obs: accessibility_tree is None or empty, falling back to str(obs)")
                    return str(obs)

                try:
                    linearized = linearize_accessibility_tree(tree, platform="ubuntu")
                    return linearized
                except Exception as e:
                    logger.exception(f"Failed to linearize accessibility_tree: {e}")
                    return str(obs)
            except Exception as e:
                logger.exception(f"Unexpected error while processing obs: {e}")
                return str(obs)

        # 观测类型：只用终端输出（对齐 mm_agents.PromptAgent 的终端模式）
        if self.observation_type == "terminal":
            try:
                if not obs or not isinstance(obs, dict):
                    logger.warning("_process_obs(terminal): obs is empty or not a dict, falling back to str(obs)")
                    return str(obs)
                term = obs.get("terminal")
                if term is None:
                    # 在很多 OSWorld 版本里，terminal 字段并不一定可用；为了避免上游拿到几乎
                    # 为空的信息（导致 prompt_length=0、Processed prompts: 0it），这里尝试
                    # 使用 a11y_tree 作为兜底文本。
                    tree = obs.get("accessibility_tree")
                    if tree:
                        try:
                            linearized = linearize_accessibility_tree(tree, platform="ubuntu")
                            logger.warning("_process_obs(terminal): terminal is None, falling back to linearized accessibility_tree")
                            return linearized
                        except Exception as e:
                            logger.exception(f"Failed to linearize accessibility_tree in terminal mode: {e}")
                    logger.warning("_process_obs(terminal): terminal is None and no usable accessibility_tree, falling back to str(obs)")
                    return str(obs)
                return term
            except Exception as e:
                logger.exception(f"Unexpected error while processing terminal obs: {e}")
                return str(obs)

        # 其它模式暂时保持原样：直接字符串化整个观测
        return str(obs)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider_name", default="docker")
    parser.add_argument("--observation_type", default="a11y_tree")
    parser.add_argument("--action_space", default="pyautogui")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--screen_width", type=int, default=1920)
    parser.add_argument("--screen_height", type=int, default=1080)
    parser.add_argument("--sleep_after_execution", type=float, default=0.0)
    parser.add_argument("--max_steps", type=int, default=15)
    parser.add_argument("--task_config_path", default="evaluation_examples/test_small_os.json")
    parser.add_argument("--path_to_vm", default=None,
                        help="VNC connection string in format hostname:vnc_port:http_port")
    parser.add_argument("--instance_name", default=None,
                        help="VM instance name (passed to DesktopEnv)")
    parser.add_argument("--image_name", default=None,
                        help="VM image name (passed to DesktopEnv)")
    parser.add_argument("--client_password", default=None,
                        help="Client password for VM (passed to DesktopEnv)")
    parser.add_argument("--use_gpt_eval", action="store_true", default=True,
                        help="Enable GPT evaluation to get reward and cost")
    
    args = parser.parse_args()
    
    server = OSWorldServer(
        provider_name=args.provider_name,
        observation_type=args.observation_type,
        action_space=args.action_space,
        headless=args.headless,
        screen_width=args.screen_width,
        screen_height=args.screen_height,
        sleep_after_execution=args.sleep_after_execution,
        max_steps=args.max_steps,
        task_config_path=args.task_config_path,
        path_to_vm=args.path_to_vm,
        instance_name=args.instance_name,
        image_name=args.image_name,
        client_password=args.client_password,
        use_gpt_eval=args.use_gpt_eval,
    )
    server.run()

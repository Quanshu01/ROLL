import argparse
import os

from dacite import from_dict
from hydra import compose, initialize
from omegaconf import OmegaConf

from roll.distributed.scheduler.initialize import init
from roll.pipeline.agentic.agentic_config import AgenticConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", help="The path of the main configuration file", default="config")
    parser.add_argument(
        "--config_name", help="The name of the main configuration file (without extension).", default="sppo_config"
    )
    args = parser.parse_args()

    initialize(config_path=args.config_path, job_name="app")

    # 支持通过环境变量传入 Hydra 覆盖项（例如：
    # HYDRA_OVERRIDES="exp_name=foo hydra.run.dir=./output/logs/foo"）
    overrides_env = os.environ.get("HYDRA_OVERRIDES", "")
    overrides = []
    if overrides_env:
        # 简单按空格分割覆盖项
        overrides = overrides_env.split()
        print(f"Applying Hydra overrides from HYDRA_OVERRIDES: {overrides}")

    cfg = compose(config_name=args.config_name, overrides=overrides)

    print(OmegaConf.to_yaml(cfg, resolve=True))

    ppo_config = from_dict(data_class=AgenticConfig, data=OmegaConf.to_container(cfg, resolve=True))

    init()
    from roll.pipeline.agentic.agentic_pipeline import AgenticPipeline

    pipeline = AgenticPipeline(pipeline_config=ppo_config)

    pipeline.run()


if __name__ == "__main__":
    main()

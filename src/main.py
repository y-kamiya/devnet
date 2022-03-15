from __future__ import annotations

import os
import sys

import hydra
from hydra.core.config_store import ConfigStore
from logzero import setup_logger
from omegaconf import OmegaConf

from devnet.config import TrainerConfig
from devnet.trainer import Trainer

cs = ConfigStore.instance()
cs.store(name="base_config", node=TrainerConfig)


@hydra.main(config_path="conf", config_name="main")
def main(config: TrainerConfig):
    if not os.path.isabs(config.dataroot):
        config.dataroot = os.path.join(hydra.utils.get_original_cwd(), config.dataroot)
    if not os.path.isabs(config.model_path):
        config.model_path = os.path.join(hydra.utils.get_original_cwd(), config.model_path)

    print(OmegaConf.to_yaml(config))

    os.makedirs(os.path.dirname(config.model_path), exist_ok=True)

    logger = setup_logger(name=__name__, level=config.loglevel.name)

    if config.predict_only:
        Trainer.predict(config, logger)
        sys.exit()

    trainer = Trainer(config, logger)
    trainer.train()


if __name__ == "__main__":
    main()
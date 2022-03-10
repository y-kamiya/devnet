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

from cProfile import Profile


@hydra.main(config_path="conf", config_name="main")
def main(config: TrainerConfig):
    dataroot = config.dataroot
    if not os.path.isabs(dataroot):
        config.dataroot = os.path.join(hydra.utils.get_original_cwd(), dataroot)

    print(OmegaConf.to_yaml(config))

    logger = setup_logger(name=__name__, level=config.loglevel.name)

    trainer = Trainer(config, logger)
    prf = Profile()
    prf.runcall(trainer.train)
    prf.dump_stats("train.prof")
    # trainer.train()


if __name__ == "__main__":
    main()

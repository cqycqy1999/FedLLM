from __future__ import annotations

import argparse

from fedpost.pipeline.launcher import Launcher
from fedpost.utils.config import ConfigLoader
from fedpost.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = ConfigLoader.from_yaml(args.config)
    ConfigLoader.validate(cfg)
    set_seed(cfg.seed)

    launcher = Launcher(cfg)
    results = launcher.run()
    print("Training finished.")
    print(results)


if __name__ == "__main__":
    main()
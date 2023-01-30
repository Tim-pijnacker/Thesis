#!/usr/bin/env python3 -u

import fairseq_cli.hydra_train as fairseq_hydra_train

from sfseq.criteria import *


def cli_main():
    fairseq_hydra_train.cli_main()


if __name__ == "__main__":
    fairseq_hydra_train.cli_main()

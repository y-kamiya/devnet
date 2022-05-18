import pytest
from logging import getLogger

import pandas as pd

from devnet.config import TrainerConfig
from devnet.trainer import Trainer, Phase
from devnet.trainer import TableDataset

logger = getLogger()


# def test_train():
#     config = TrainerConfig(
#         dataroot="data/debug",
#     )
#     df_train = pd.read_csv("data/debug/train.csv")
#     df_eval = pd.read_csv("data/debug/eval.csv")
#     trainer = Trainer(config, logger)
#     trainer.train(df_train, df_eval)


@pytest.fixture(params=[
    Phase.TRAIN,
    Phase.EVAL,
    Phase.PREDICT,
])
def f_phase(request) -> Phase:
    return request.param


class TestTableDataset:
    def test_create_with_class(self, f_phase):
        data = [
            {"class": 0, "c0": 0, "c1": 0},
            {"class": 1, "c0": 1, "c1": 1},
        ]
        df = pd.DataFrame(data)
        dataset = TableDataset(df, f_phase, logger)

        assert dataset.n_columns == len(data[0].keys()) - 1
        assert dataset.xs.tolist() == [[0, 0], [1, 1]]
        assert dataset.ys.tolist() == [0, 1]
        assert (dataset.df == df).all(axis=None)

    def test_create_without_class(self, f_phase):
        data = [
            {"c0": 0, "c1": 0},
            {"c0": 1, "c1": 1},
        ]
        df = pd.DataFrame(data)

        if f_phase in [Phase.TRAIN, Phase.EVAL]:
            with pytest.raises(ValueError):
                TableDataset(df, f_phase, logger)
        else:
            dataset = TableDataset(df, f_phase, logger)
            assert dataset.n_columns == len(data[0].keys())
            assert dataset.xs.tolist() == [[0, 0], [1, 1]]
            assert (dataset.df == df).all(axis=None)


import pytest
from logging import getLogger

import pandas as pd

from devnet.config import TrainerConfig
from devnet.trainer import Trainer, Phase, BalancedBatchSampler
from devnet.trainer import TableDataset

logger = getLogger()


@pytest.fixture(params=[
    Phase.TRAIN,
    Phase.EVAL,
    Phase.PREDICT,
])
def f_phase(request) -> Phase:
    return request.param

@pytest.fixture
def f_df() -> pd.DataFrame:
    data = [
        {"class": 0, "c0": 11},
        {"class": 0, "c0": 12},
        {"class": 1, "c0": 21},
        {"class": 1, "c0": 22},
    ]
    return pd.DataFrame(data)


@pytest.fixture
def f_trainer():
    config = TrainerConfig(model_path="./dummy")
    return Trainer(config)


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


class TestBalancedBatchSampler:
    def test_with_seed(self, f_df):
        dataset = TableDataset(f_df, Phase.TRAIN, logger)

        n_batch = 3
        batch_size = 2
        sampler0 = BalancedBatchSampler(dataset, n_batch=3, batch_size=2, seed=0)
        sampler1 = BalancedBatchSampler(dataset, n_batch=3, batch_size=2, seed=0)
        count = 0
        for indexes in sampler0:
            assert indexes == next(iter(sampler1))
            assert len(indexes) == batch_size
            assert f_df.loc[indexes[0], "class"] == 0
            assert f_df.loc[indexes[1], "class"] == 1
            count += 1
        assert count == n_batch


class TestTrainer:
    def test_setup_train_with_df(self, f_trainer, f_df):
        with pytest.raises(ValueError):
            f_trainer.model is None
            f_trainer.optimizer is None

        f_trainer._setup(Phase.TRAIN, f_df)

        assert f_trainer.model is not None
        assert f_trainer.optimizer is not None

    def test_setup_eval_with_df(self, f_trainer, f_df):
        with pytest.raises(ValueError):
            f_trainer.model is None
            f_trainer.optimizer is None

        f_trainer._setup(Phase.EVAL, f_df)

        assert f_trainer.model is not None
        with pytest.raises(ValueError):
            assert f_trainer.optimizer is None

    def test_setup_predict_with_df(self, f_trainer, f_df):
        with pytest.raises(ValueError):
            f_trainer.model is None
            f_trainer.optimizer is None

        with pytest.raises(ValueError):
            f_trainer._setup(Phase.PREDICT, f_df)

# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os.path
from unittest import mock

import pytest
import torch

from pl_examples import _DALI_AVAILABLE, _HF_AVAILABLE
from pytorch_lightning.utilities.imports import _IS_WINDOWS
from tests.helpers.runif import RunIf

ARGS_DEFAULT = (
    "--trainer.default_root_dir %(tmpdir)s "
    "--trainer.max_epochs 1 "
    "--trainer.limit_train_batches 2 "
    "--trainer.limit_val_batches 2 "
    "--trainer.limit_test_batches 2 "
    "--trainer.limit_predict_batches 2 "
    "--data.batch_size 32 "
)
ARGS_GPU = ARGS_DEFAULT + "--trainer.gpus 1 "


@pytest.mark.skipif(not _DALI_AVAILABLE, reason="Nvidia DALI required")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(_IS_WINDOWS, reason="Not supported on Windows")
@pytest.mark.parametrize("cli_args", [ARGS_GPU])
def test_examples_mnist_dali(tmpdir, cli_args):
    from pl_examples.integration_examples.dali_image_classifier import cli_main

    # update the temp dir
    cli_args = cli_args % {"tmpdir": tmpdir}
    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args.strip().split()):
        cli_main()


@pytest.mark.skipif(not _HF_AVAILABLE, reason="Hugging Face transformers and datasets packages required")
@RunIf(min_gpus=1, skip_windows=True)
@pytest.mark.parametrize(
    "config_file",
    ["nofts_baseline.yaml", "fts_explicit.yaml", "fts_implicit.yaml"],
    ids=["nofts_baseline", "fts_explicit", "fts_implicit"],
)
def test_examples_fts_superglue(monkeypatch, tmpdir, config_file):
    from pl_examples.basic_examples.fts_superglue import cli_main

    example_script = os.path.join(os.path.dirname(__file__), "basic_examples", "fts_superglue.py")
    config_loc = [os.path.join(os.path.dirname(__file__), "basic_examples/config/fts", config_file)]
    cli_args = [
        f"--trainer.default_root_dir={tmpdir.strpath}",
        "--trainer.max_epochs=1",
        "--trainer.limit_train_batches=2",
        "--trainer.gpus=1",
    ]
    monkeypatch.setattr("sys.argv", [example_script, "fit", "--config"] + config_loc + cli_args)
    cli_main()

import matplotlib

matplotlib.use("Agg")

import pytest  # noqa: E402

from fbmwavelets.experiments import (  # noqa: E402
    DEFAULT_EXPERIMENTS,
    Settings,
    latex_hurst_table,
    run,
)


@pytest.mark.parametrize("name", [n for n in DEFAULT_EXPERIMENTS if n != "hurst-table"])
def test_experiment_writes_figure(name, tmp_path):
    run(name, Settings(save_dir=tmp_path))
    assert any(tmp_path.glob("*.png"))


def test_latex_table_structure():
    table = latex_hurst_table([0.1, 0.9], [0.12, 0.88])
    assert r"\begin{tabular}{lcc}" in table
    assert r"$|H - \hat{H}|$ & $0.02$ & $0.02$" in table

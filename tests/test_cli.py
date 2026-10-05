import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "generate_fbm.py"


def run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--no-usetex", "--log-level", "WARNING", *args],
        capture_output=True,
        text=True,
    )


def test_hurst_range_writes_figure_and_data(tmp_path):
    result = run_cli(
        "-r", "0.4", "0.85", "0.05", "-N", "200", "--save-data", "--save-dir", str(tmp_path)
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "fbm_series.png").exists()
    import numpy as np

    data = np.load(tmp_path / "fbm_series.npz")
    assert len([k for k in data.files if k.startswith("H_")]) == 10


def test_rmd_method_with_short_flags(tmp_path):
    result = run_cli(
        "-H", "0.3", "0.6", "-n", "2", "-m", "rmd", "-l", "single", "--save-dir", str(tmp_path)
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "fbm_series.png").exists()


def test_invalid_hurst_is_rejected(tmp_path):
    result = run_cli("-H", "1.2", "--save-dir", str(tmp_path))
    assert result.returncode != 0
    assert "(0, 1)" in result.stderr


def test_make_gif_planar_paths(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "make_gif.py"
    out = tmp_path / "paths.gif"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--no-usetex",
            "-r",
            "0.3",
            "0.7",
            "0.4",
            "-d",
            "2",
            "-N",
            "100",
            "--frames",
            "5",
            "--fps",
            "5",
            "--output",
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert out.exists()


def test_make_gif_several_series_with_synthesis(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "make_gif.py"
    out = tmp_path / "paths.gif"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--no-usetex",
            "-H",
            "0.3",
            "0.7",
            "-n",
            "2",
            "-m",
            "rmd",
            "--levels",
            "6",
            "-c",
            "series",
            "--no-legend",
            "--frames",
            "5",
            "--fps",
            "5",
            "--hold",
            "0",
            "--output",
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert out.exists()

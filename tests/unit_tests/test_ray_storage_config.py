"""Tests for host-backed Ray temporary storage configuration."""

from pathlib import Path

from rlinf.scheduler.cluster.cluster import Cluster, ClusterEnvVar


def test_local_ray_storage_kwargs_are_empty_by_default(monkeypatch) -> None:
    for env_var in (
        ClusterEnvVar.RAY_TEMP_DIR,
        ClusterEnvVar.RAY_OBJECT_SPILL_DIR,
    ):
        monkeypatch.delenv(Cluster.get_full_env_var_name(env_var), raising=False)

    assert Cluster._get_local_ray_storage_kwargs() == {}


def test_local_ray_storage_kwargs_create_configured_directories(
    monkeypatch, tmp_path
) -> None:
    temp_dir = tmp_path / "ray" / "tmp"
    spill_dir = tmp_path / "ray" / "spill"
    monkeypatch.setenv("RLINF_RAY_TEMP_DIR", str(temp_dir))
    monkeypatch.setenv("RLINF_RAY_OBJECT_SPILL_DIR", str(spill_dir))

    kwargs = Cluster._get_local_ray_storage_kwargs()

    assert kwargs == {
        "_temp_dir": str(temp_dir.resolve()),
        "object_spilling_directory": str(spill_dir.resolve()),
    }
    assert Path(kwargs["_temp_dir"]).is_dir()
    assert Path(kwargs["object_spilling_directory"]).is_dir()

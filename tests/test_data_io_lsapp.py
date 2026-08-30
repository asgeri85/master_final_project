"""Unit tests for src.data.io.read_raw_events_lsapp -- the tar-header fix.

data/raw/lsapp/lsapp.tsv is a single-member POSIX tar archive (512-byte
header followed by the real TSV), detected Hafta 1, fixed Hafta 4 -- see
docs/00-karar-notu.md SS4.2/SS16. These tests build a tiny synthetic tar
fixture rather than depending on the real 179MB raw file.
"""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

from src.data.io import read_raw_events_lsapp

_TSV = (
    "user_id\tsession_id\ttimestamp\tapp_name\tevent_type\n"
    "0\t1\t2018-01-16 06:01:05\tZebra App\tOpened\n"
    "0\t1\t2018-01-16 06:01:07\tZebra App\tClosed\n"
    "0\t1\t2018-01-16 06:05:00\tAlpha App\tOpened\n"
    "1\t2\t2018-01-16 07:00:00\tAlpha App\tOpened\n"
)


def _write_lsapp_tar_fixture(path: Path) -> None:
    payload = _TSV.encode("utf-8")
    with tarfile.open(path, "w") as tf:
        info = tarfile.TarInfo(name="lsapp.tsv")
        info.size = len(payload)
        tf.addfile(info, io.BytesIO(payload))


def test_read_raw_events_lsapp_extracts_tar_and_maps_schema(tmp_path):
    fixture_path = tmp_path / "lsapp.tsv"
    _write_lsapp_tar_fixture(fixture_path)

    df = read_raw_events_lsapp(fixture_path)

    assert list(df.columns) == ["timestamp", "visitorid", "event", "itemid", "transactionid", "_raw_row_id"]
    assert len(df) == 4
    assert df["transactionid"].isna().all()
    assert df["_raw_row_id"].tolist() == [0, 1, 2, 3]


def test_read_raw_events_lsapp_itemid_mapping_is_sorted_and_stable(tmp_path):
    fixture_path = tmp_path / "lsapp.tsv"
    _write_lsapp_tar_fixture(fixture_path)

    df = read_raw_events_lsapp(fixture_path)

    # "Alpha App" < "Zebra App" alphabetically -> itemid 0 and 1 respectively,
    # regardless of which app appears first in the file.
    mapping = df.attrs["item_id_to_app_name"]
    assert mapping == {0: "Alpha App", 1: "Zebra App"}
    assert df.loc[df["event"] == "Opened", "itemid"].tolist() == [1, 0, 0]


def test_read_raw_events_lsapp_timestamp_is_millisecond_epoch(tmp_path):
    fixture_path = tmp_path / "lsapp.tsv"
    _write_lsapp_tar_fixture(fixture_path)

    df = read_raw_events_lsapp(fixture_path)

    # "2018-01-16 06:01:05" UTC-naive -> 1516082465 seconds -> 1516082465000 ms.
    assert df["timestamp"].iloc[0] == 1516082465000
    assert df["timestamp"].dtype == "int64"
    # a 2-second gap between rows 0 and 1 must survive as 2000 ms, not 2 (a
    # microsecond/millisecond unit mixup would silently collapse this).
    assert df["timestamp"].iloc[1] - df["timestamp"].iloc[0] == 2000


def test_read_raw_events_lsapp_visitorid_matches_user_id(tmp_path):
    fixture_path = tmp_path / "lsapp.tsv"
    _write_lsapp_tar_fixture(fixture_path)

    df = read_raw_events_lsapp(fixture_path)

    assert df["visitorid"].tolist() == [0, 0, 0, 1]

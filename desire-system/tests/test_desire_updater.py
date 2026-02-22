"""Tests for desire_updater."""

import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from desire_updater import (
    CURIOSITY_WEIGHT_PER_SEED,
    DesireState,
    add_curiosity,
    compute_curiosity_level,
    compute_desires,
    list_curiosities,
    load_curiosities,
    load_desires,
    resolve_curiosity,
    save_curiosities,
    save_desires,
)


class TestComputeCuriosityLevel:
    def _make_seeds(self, items: list[tuple[str, str]]) -> list[dict]:
        """items: list of (topic, timestamp_str) tuples"""
        return [
            {
                "id": f"id-{i}",
                "topic": topic,
                "source": "test",
                "timestamp": ts,
                "resolved": False,
            }
            for i, (topic, ts) in enumerate(items)
        ]

    def test_with_one_fresh_seed(self):
        now = datetime(2026, 2, 20, 12, 0, 0, tzinfo=timezone.utc)
        ts = now.isoformat()

        seeds = self._make_seeds([("テスト", ts)])
        level, unresolved = compute_curiosity_level(seeds, now)

        # freshness=1.0, weight=1.0*0.3 = 0.3
        assert level == pytest.approx(CURIOSITY_WEIGHT_PER_SEED, abs=0.01)
        assert len(unresolved) == 1
        assert unresolved[0] == "テスト"

    def test_with_two_fresh_seeds(self):
        now = datetime(2026, 2, 20, 12, 0, 0, tzinfo=timezone.utc)
        ts = now.isoformat()

        seeds = self._make_seeds([("トピック1", ts), ("トピック2", ts)])
        level, unresolved = compute_curiosity_level(seeds, now)

        # 2 seeds * freshness 1.0 * 0.3 = 0.6
        assert level == pytest.approx(0.6, abs=0.01)
        assert len(unresolved) == 2

    def test_with_four_seeds_caps_at_one(self):
        now = datetime(2026, 2, 20, 12, 0, 0, tzinfo=timezone.utc)
        ts = now.isoformat()

        seeds = self._make_seeds([("A", ts), ("B", ts), ("C", ts), ("D", ts)])
        level, _ = compute_curiosity_level(seeds, now)

        # 4 * 1.0 * 0.3 = 1.2 -> capped at 1.0
        assert level == 1.0

    def test_freshness_decay(self):
        now = datetime(2026, 2, 20, 12, 0, 0, tzinfo=timezone.utc)
        # 12時間前の種
        old_ts = (now - timedelta(hours=12)).isoformat()

        seeds = self._make_seeds([("古い種", old_ts)])
        level, unresolved = compute_curiosity_level(seeds, now)

        # freshness = max(0.1, 1.0 - 12/24) = 0.5
        # level = 0.5 * 0.3 = 0.15
        assert level == pytest.approx(0.15, abs=0.01)
        assert len(unresolved) == 1

    def test_very_old_seed_minimum_freshness(self):
        now = datetime(2026, 2, 20, 12, 0, 0, tzinfo=timezone.utc)
        # 48時間前（完全に薄れるが最低0.1）
        old_ts = (now - timedelta(hours=48)).isoformat()

        seeds = self._make_seeds([("超古い", old_ts)])
        level, _ = compute_curiosity_level(seeds, now)

        # freshness = max(0.1, 1.0 - 48/24) = 0.1
        # level = 0.1 * 0.3 = 0.03
        assert level == pytest.approx(0.03, abs=0.01)

    def test_empty_returns_zero(self):
        now = datetime(2026, 2, 20, 12, 0, 0, tzinfo=timezone.utc)

        level, unresolved = compute_curiosity_level([], now)

        assert level == 0.0
        assert len(unresolved) == 0

    def test_resolved_seeds_are_ignored(self):
        now = datetime(2026, 2, 20, 12, 0, 0, tzinfo=timezone.utc)
        ts = now.isoformat()

        seeds = [
            {
                "id": "id-0",
                "topic": "解決済み",
                "source": "test",
                "timestamp": ts,
                "resolved": True,
            },
        ]
        level, unresolved = compute_curiosity_level(seeds, now)

        assert level == 0.0
        assert len(unresolved) == 0


class TestSaveAndLoadCuriosities:
    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "curiosities.json"
            seeds = [
                {
                    "id": "test-id-1",
                    "topic": "テスト1",
                    "source": "camera",
                    "timestamp": "2026-02-20T12:00:00+00:00",
                    "resolved": False,
                },
                {
                    "id": "test-id-2",
                    "topic": "テスト2",
                    "source": "conversation",
                    "timestamp": "2026-02-20T13:00:00+00:00",
                    "resolved": True,
                },
            ]
            save_curiosities(seeds, path)
            loaded = load_curiosities(path)

            assert len(loaded) == 2
            assert loaded[0]["topic"] == "テスト1"
            assert loaded[1]["resolved"] is True

    def test_load_missing_file_returns_empty(self):
        result = load_curiosities(Path("/nonexistent/path/curiosities.json"))
        assert result == []

    def test_save_creates_parent_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "dir" / "curiosities.json"
            save_curiosities([], path)
            assert path.exists()


class TestCuriosityOperations:
    def test_add_curiosity(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "curiosities.json"
            seed_id = add_curiosity("テスト", "camera", path)

            assert seed_id  # non-empty
            seeds = load_curiosities(path)
            assert len(seeds) == 1
            assert seeds[0]["topic"] == "テスト"
            assert seeds[0]["source"] == "camera"
            assert seeds[0]["resolved"] is False
            assert seeds[0]["id"] == seed_id

    def test_add_multiple_curiosities(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "curiosities.json"
            add_curiosity("トピック1", "camera", path)
            add_curiosity("トピック2", "conversation", path)

            seeds = load_curiosities(path)
            assert len(seeds) == 2

    def test_resolve_curiosity_success(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "curiosities.json"
            seed_id = add_curiosity("テスト", "camera", path)

            result = resolve_curiosity(seed_id, path)

            assert result is True
            seeds = load_curiosities(path)
            assert seeds[0]["resolved"] is True

    def test_resolve_curiosity_not_found(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "curiosities.json"
            save_curiosities([], path)

            result = resolve_curiosity("nonexistent", path)

            assert result is False

    def test_list_curiosities_unresolved_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "curiosities.json"
            id1 = add_curiosity("トピック1", "camera", path)
            add_curiosity("トピック2", "conversation", path)
            resolve_curiosity(id1, path)

            items = list_curiosities(include_resolved=False, path=path)

            assert len(items) == 1
            assert items[0]["topic"] == "トピック2"

    def test_list_curiosities_include_resolved(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "curiosities.json"
            id1 = add_curiosity("トピック1", "camera", path)
            add_curiosity("トピック2", "conversation", path)
            resolve_curiosity(id1, path)

            items = list_curiosities(include_resolved=True, path=path)

            assert len(items) == 2

    def test_list_curiosities_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "curiosities.json"
            save_curiosities([], path)

            items = list_curiosities(path=path)

            assert items == []


class TestComputeDesires:
    def test_only_browse_curiosity(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "curiosities.json"
            save_curiosities([], path)
            now = datetime(2026, 2, 22, 12, 0, 0, tzinfo=timezone.utc)

            state = compute_desires(now, path)

            assert set(state.desires.keys()) == {"browse_curiosity"}

    def test_no_seeds_level_zero(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "curiosities.json"
            save_curiosities([], path)
            now = datetime(2026, 2, 22, 12, 0, 0, tzinfo=timezone.utc)

            state = compute_desires(now, path)

            assert state.desires["browse_curiosity"] == 0.0
            assert state.dominant == "browse_curiosity"
            assert state.pending_curiosities == []

    def test_with_seeds_level_positive(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "curiosities.json"
            now = datetime(2026, 2, 22, 12, 0, 0, tzinfo=timezone.utc)
            seeds = [
                {
                    "id": "id-1",
                    "topic": "テスト好奇心",
                    "source": "conversation",
                    "timestamp": now.isoformat(),
                    "resolved": False,
                },
            ]
            save_curiosities(seeds, path)

            state = compute_desires(now, path)

            assert state.desires["browse_curiosity"] > 0
            assert state.pending_curiosities == ["テスト好奇心"]

    def test_desires_values_in_range(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "curiosities.json"
            now = datetime(2026, 2, 22, 12, 0, 0, tzinfo=timezone.utc)
            seeds = [
                {
                    "id": f"id-{i}",
                    "topic": f"topic-{i}",
                    "source": "test",
                    "timestamp": now.isoformat(),
                    "resolved": False,
                }
                for i in range(10)
            ]
            save_curiosities(seeds, path)

            state = compute_desires(now, path)

            for level in state.desires.values():
                assert 0.0 <= level <= 1.0


class TestSaveAndLoadDesires:
    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "desires.json"
            state = DesireState(
                updated_at="2026-02-22T12:00:00+00:00",
                desires={"browse_curiosity": 0.6},
                dominant="browse_curiosity",
            )
            save_desires(state, path)

            loaded = load_desires(path)
            assert loaded is not None
            assert loaded.dominant == "browse_curiosity"
            assert loaded.desires["browse_curiosity"] == pytest.approx(0.6)

    def test_roundtrip_with_curiosities(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "desires.json"
            state = DesireState(
                updated_at="2026-02-22T12:00:00+00:00",
                desires={"browse_curiosity": 0.6},
                dominant="browse_curiosity",
                pending_curiosities=["テスト1", "テスト2"],
            )
            save_desires(state, path)

            loaded = load_desires(path)
            assert loaded is not None
            assert loaded.pending_curiosities == ["テスト1", "テスト2"]

    def test_load_missing_file_returns_none(self):
        result = load_desires(Path("/nonexistent/path/desires.json"))
        assert result is None

    def test_save_creates_parent_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "dir" / "desires.json"
            state = DesireState(
                updated_at="2026-02-22T12:00:00",
                desires={"browse_curiosity": 1.0},
                dominant="browse_curiosity",
            )
            save_desires(state, path)
            assert path.exists()

    def test_saved_json_is_readable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "desires.json"
            state = DesireState(
                updated_at="2026-02-22T12:00:00",
                desires={"browse_curiosity": 0.6},
                dominant="browse_curiosity",
            )
            save_desires(state, path)

            with open(path) as f:
                data = json.load(f)
            assert data["dominant"] == "browse_curiosity"
            assert data["desires"]["browse_curiosity"] == pytest.approx(0.6)

    def test_no_pending_curiosities_omitted_from_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "desires.json"
            state = DesireState(
                updated_at="2026-02-22T12:00:00",
                desires={"browse_curiosity": 0.0},
                dominant="browse_curiosity",
                pending_curiosities=[],
            )
            save_desires(state, path)

            with open(path) as f:
                data = json.load(f)
            assert "pending_curiosities" not in data

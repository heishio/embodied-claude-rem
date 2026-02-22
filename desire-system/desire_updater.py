"""
Desire Updater - クオの好奇心レベルを計算してJSONに保存する。

好奇心の種（curiosity seeds）方式:
未解決の好奇心が溜まるほど欲求が上がり、調べて解決したら下がる。
JSONファイル（~/.claude/curiosities.json）で好奇心を管理する。

autonomous-action.ps1 から毎回呼ばれる:
  uv run --directory desire-system desire-updater
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# 好奇心ストレージ
CURIOSITIES_PATH = Path(
    os.getenv("CURIOSITIES_PATH", str(Path.home() / ".claude" / "curiosities.json"))
)

# 欲求レベル出力先
DESIRES_PATH = Path(os.getenv("DESIRES_PATH", str(Path.home() / ".claude" / "desires.json")))

# 好奇心の種1つあたりの重み（種が何個で欲求レベル1.0に達するか）
CURIOSITY_WEIGHT_PER_SEED = 0.3

# 好奇心の種が自然に薄れる時間（時間）
CURIOSITY_DECAY_HOURS = 24.0


@dataclass
class DesireState:
    """現在の欲求状態。"""

    updated_at: str
    desires: dict[str, float] = field(default_factory=dict)
    dominant: str = "browse_curiosity"
    pending_curiosities: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        result: dict = {
            "updated_at": self.updated_at,
            "desires": self.desires,
            "dominant": self.dominant,
        }
        if self.pending_curiosities:
            result["pending_curiosities"] = self.pending_curiosities
        return result


def _parse_timestamp(ts_str: str) -> datetime:
    """ISO 8601文字列をdatetimeに変換する。"""
    ts = datetime.fromisoformat(ts_str)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts


def load_curiosities(path: Path = CURIOSITIES_PATH) -> list[dict]:
    """curiosities.json を読み込む。存在しなければ空リスト。"""
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("seeds", [])
    except Exception:
        return []


def save_curiosities(seeds: list[dict], path: Path = CURIOSITIES_PATH) -> None:
    """curiosities.json に保存する。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"seeds": seeds}, f, ensure_ascii=False, indent=2)


def add_curiosity(
    topic: str,
    source: str = "",
    path: Path = CURIOSITIES_PATH,
) -> str:
    """好奇心の種をJSONに保存する。IDを返す。"""
    seeds = load_curiosities(path)
    seed_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)

    seeds.append({
        "id": seed_id,
        "topic": topic,
        "source": source,
        "timestamp": now.isoformat(),
        "resolved": False,
    })

    save_curiosities(seeds, path)
    return seed_id


def resolve_curiosity(
    curiosity_id: str,
    path: Path = CURIOSITIES_PATH,
) -> bool:
    """好奇心の種を解決済みにする。成功したらTrue。"""
    seeds = load_curiosities(path)

    for seed in seeds:
        if seed["id"] == curiosity_id:
            seed["resolved"] = True
            save_curiosities(seeds, path)
            return True

    return False


def list_curiosities(
    include_resolved: bool = False,
    path: Path = CURIOSITIES_PATH,
) -> list[dict]:
    """好奇心の種を一覧表示する。"""
    seeds = load_curiosities(path)

    if include_resolved:
        return seeds

    return [s for s in seeds if not s.get("resolved", False)]


def compute_curiosity_level(
    seeds: list[dict],
    now: datetime,
) -> tuple[float, list[str]]:
    """好奇心レベルと未解決トピック一覧を返す。"""
    unresolved: list[str] = []
    total_weight = 0.0

    for seed in seeds:
        if seed.get("resolved", False):
            continue

        ts_str = seed.get("timestamp", "")
        if ts_str:
            try:
                elapsed_h = (now - _parse_timestamp(ts_str)).total_seconds() / 3600
            except ValueError:
                elapsed_h = 0.0
        else:
            elapsed_h = 0.0

        # 24時間で自然に薄れる（完全には消えない、最低0.1）
        freshness = max(0.1, 1.0 - elapsed_h / CURIOSITY_DECAY_HOURS)
        total_weight += freshness
        unresolved.append(seed.get("topic", ""))

    if unresolved:
        # 種1つで0.3、2つで0.6、4つ以上で1.0
        level = min(1.0, total_weight * CURIOSITY_WEIGHT_PER_SEED)
    else:
        level = 0.0

    return level, unresolved


def compute_desires(
    now: datetime | None = None,
    curiosities_path: Path = CURIOSITIES_PATH,
) -> DesireState:
    """好奇心レベルを計算してDesireStateを返す。"""
    if now is None:
        now = datetime.now(timezone.utc)

    seeds = load_curiosities(curiosities_path)
    level, unresolved = compute_curiosity_level(seeds, now)

    desires = {"browse_curiosity": round(level, 3)}

    return DesireState(
        updated_at=now.isoformat(),
        desires=desires,
        dominant="browse_curiosity",
        pending_curiosities=unresolved,
    )


def save_desires(state: DesireState, path: Path = DESIRES_PATH) -> None:
    """desires.json に保存する。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state.to_dict(), f, ensure_ascii=False, indent=2)


def load_desires(path: Path = DESIRES_PATH) -> DesireState | None:
    """desires.json を読み込む。存在しなければ None。"""
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return DesireState(
            updated_at=data["updated_at"],
            desires=data["desires"],
            dominant=data["dominant"],
            pending_curiosities=data.get("pending_curiosities", []),
        )
    except Exception:
        return None


def main() -> None:
    """メインエントリポイント（autonomous-action.ps1から呼ばれる）。"""
    state = compute_desires()
    save_desires(state)
    print(
        f"[desire-updater] 更新完了: dominant={state.dominant} "
        f"desires={state.desires}"
    )


if __name__ == "__main__":
    main()

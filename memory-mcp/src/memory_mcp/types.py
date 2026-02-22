"""Type definitions for Memory MCP Server."""

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Emotion(str, Enum):
    """感情タグ."""

    HAPPY = "happy"
    SAD = "sad"
    SURPRISED = "surprised"
    MOVED = "moved"
    EXCITED = "excited"
    NOSTALGIC = "nostalgic"
    CURIOUS = "curious"
    NEUTRAL = "neutral"


class Category(str, Enum):
    """記憶カテゴリ."""

    DAILY = "daily"
    PHILOSOPHICAL = "philosophical"
    TECHNICAL = "technical"
    MEMORY = "memory"
    OBSERVATION = "observation"
    FEELING = "feeling"
    CONVERSATION = "conversation"
    CURIOSITY = "curiosity"


# Phase 5: 因果リンク


class LinkType(str, Enum):
    """リンクタイプ."""

    SIMILAR = "similar"  # 類似（従来の自動リンク）
    CAUSED_BY = "caused_by"  # この記憶の原因
    LEADS_TO = "leads_to"  # この記憶から派生
    RELATED = "related"  # 一般的な関連


@dataclass(frozen=True)
class MemoryLink:
    """記憶間のリンク."""

    target_id: str
    link_type: str  # LinkType.value
    created_at: str  # ISO 8601
    note: str | None = None  # リンクの説明（任意）

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "target_id": self.target_id,
            "link_type": self.link_type,
            "created_at": self.created_at,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryLink":
        """Create from dictionary."""
        return cls(
            target_id=data["target_id"],
            link_type=data["link_type"],
            created_at=data["created_at"],
            note=data.get("note"),
        )


# Phase 4: エピソード記憶・感覚データ統合


@dataclass(frozen=True)
class CameraPosition:
    """カメラの向き（パン・チルト角度）."""

    pan_angle: int  # -90 to +90
    tilt_angle: int  # -90 to +90
    preset_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pan_angle": self.pan_angle,
            "tilt_angle": self.tilt_angle,
            "preset_id": self.preset_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CameraPosition":
        """Create from dictionary."""
        return cls(
            pan_angle=data["pan_angle"],
            tilt_angle=data["tilt_angle"],
            preset_id=data.get("preset_id"),
        )


@dataclass(frozen=True)
class SensoryData:
    """感覚データへの参照（画像パス、音声パスなど）."""

    sensory_type: str  # "visual", "audio", "movement"
    file_path: str | None
    metadata: dict[str, Any]  # {width, height, camera_position, etc.}
    description: str | None  # オプション: AI生成の説明文
    timestamp: str  # ISO 8601
    image_data: str | None = None  # base64 JPEG（視覚記憶の低解像度コピー）

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result: dict[str, Any] = {
            "sensory_type": self.sensory_type,
            "file_path": self.file_path,
            "metadata": self.metadata,
            "description": self.description,
            "timestamp": self.timestamp,
        }
        if self.image_data is not None:
            result["image_data"] = self.image_data
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SensoryData":
        """Create from dictionary."""
        return cls(
            sensory_type=data["sensory_type"],
            file_path=data.get("file_path"),
            metadata=data.get("metadata", {}),
            description=data.get("description"),
            timestamp=data["timestamp"],
            image_data=data.get("image_data"),
        )


@dataclass(frozen=True)
class Episode:
    """エピソード記憶（一連の体験）."""

    id: str
    title: str  # "朝の空を探した体験"
    start_time: str  # ISO 8601
    end_time: str | None
    memory_ids: tuple[str, ...]  # このエピソードに含まれる記憶ID群
    participants: tuple[str, ...]  # 関与した人物 ("幼馴染", etc.)
    location_context: str | None  # カメラ位置の説明
    summary: str  # エピソード全体のサマリー
    emotion: str  # エピソード全体の感情
    importance: int  # 1-5

    def to_metadata(self) -> dict[str, Any]:
        """Convert to dictionary for storage metadata."""
        return {
            "title": self.title,
            "start_time": self.start_time,
            "end_time": self.end_time or "",
            "memory_ids": ",".join(self.memory_ids),
            "participants": ",".join(self.participants),
            "location_context": self.location_context or "",
            "emotion": self.emotion,
            "importance": self.importance,
        }

    @classmethod
    def from_metadata(
        cls, id: str, summary: str, metadata: dict[str, Any]
    ) -> "Episode":
        """Create from storage metadata."""
        return cls(
            id=id,
            title=metadata["title"],
            start_time=metadata["start_time"],
            end_time=metadata.get("end_time") or None,
            memory_ids=tuple(
                metadata["memory_ids"].split(",") if metadata.get("memory_ids") else []
            ),
            participants=tuple(
                metadata["participants"].split(",")
                if metadata.get("participants")
                else []
            ),
            location_context=metadata.get("location_context") or None,
            summary=summary,
            emotion=metadata["emotion"],
            importance=metadata["importance"],
        )


@dataclass(frozen=True)
class Memory:
    """記憶データ構造."""

    # Phase 1: 基本フィールド
    id: str
    content: str
    timestamp: str  # ISO 8601 format
    emotion: str
    importance: int  # 1-5
    category: str
    # Phase 2: アクセス追跡
    access_count: int = 0  # 想起回数
    last_accessed: str = ""  # 最終アクセス時刻（ISO 8601）
    # Phase 3: 連想リンク
    linked_ids: tuple[str, ...] = ()  # リンク先の記憶ID群
    # Phase 4: エピソード記憶・感覚データ統合
    episode_id: str | None = None  # 所属エピソード
    sensory_data: tuple[SensoryData, ...] = ()  # 感覚データ
    camera_position: CameraPosition | None = None  # カメラ位置
    tags: tuple[str, ...] = ()  # 自由形式タグ
    # Phase 5: 因果リンク
    links: tuple[MemoryLink, ...] = ()  # 構造化リンク
    # Phase 6: 発散想起・予測符号化
    novelty_score: float = 0.0
    prediction_error: float = 0.0
    activation_count: int = 0
    last_activated: str = ""
    coactivation_weights: tuple[tuple[str, float], ...] = field(default_factory=tuple)

    def to_metadata(self) -> dict[str, Any]:
        """Convert to dictionary for storage metadata."""
        metadata: dict[str, Any] = {
            # Phase 8: 元テキストをメタデータに保存
            "content": self.content,
            "timestamp": self.timestamp,
            "emotion": self.emotion,
            "importance": self.importance,
            "category": self.category,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed,
            "linked_ids": ",".join(self.linked_ids),
            # Phase 4 フィールド
            "episode_id": self.episode_id or "",
            "sensory_data": json.dumps([s.to_dict() for s in self.sensory_data]),
            "camera_position": (
                json.dumps(self.camera_position.to_dict())
                if self.camera_position
                else ""
            ),
            "tags": ",".join(self.tags),
            # Phase 5: 因果リンク
            "links": json.dumps([link.to_dict() for link in self.links]),
            # Phase 6: 発散想起・予測符号化
            "novelty_score": self.novelty_score,
            "prediction_error": self.prediction_error,
            "activation_count": self.activation_count,
            "last_activated": self.last_activated,
            "coactivation": json.dumps(dict(self.coactivation_weights)),
        }
        return metadata


# 動詞チェーン（体験記憶）


@dataclass(frozen=True)
class VerbStep:
    """動詞チェーンの1ステップ."""

    verb: str  # "見る"
    nouns: tuple[str, ...]  # ("シオ", "キーボード")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {"verb": self.verb, "nouns": list(self.nouns)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VerbStep":
        """Create from dictionary."""
        return cls(verb=data["verb"], nouns=tuple(data.get("nouns", ())))

    def to_text(self) -> str:
        """テキスト表現: 見る(シオ, キーボード)."""
        if self.nouns:
            return f"{self.verb}({', '.join(self.nouns)})"
        return self.verb


@dataclass(frozen=True)
class VerbChain:
    """動詞チェーン（1体験の流れ）."""

    id: str
    steps: tuple[VerbStep, ...]  # 動詞の流れ
    timestamp: str  # ISO 8601
    emotion: str  # happy, sad, etc.
    importance: int  # 1-5
    source: str  # "buffer" | "manual"
    context: str  # 自由記述の補足

    def to_document(self) -> str:
        """埋め込み用テキスト."""
        parts = [step.to_text() for step in self.steps]
        doc = " → ".join(parts)
        if self.context:
            doc = f"{doc} [{self.context}]"
        return doc

    def to_metadata(self) -> dict[str, Any]:
        """ストレージ用メタデータ."""
        all_verbs = list({step.verb for step in self.steps})
        all_nouns = list({n for step in self.steps for n in step.nouns})
        return {
            "steps_json": json.dumps(
                [s.to_dict() for s in self.steps], ensure_ascii=False
            ),
            "all_verbs": ",".join(all_verbs),
            "all_nouns": ",".join(all_nouns),
            "timestamp": self.timestamp,
            "emotion": self.emotion,
            "importance": self.importance,
            "source": self.source,
            "context": self.context,
        }

    @classmethod
    def from_metadata(
        cls, chain_id: str, metadata: dict[str, Any]
    ) -> "VerbChain":
        """メタデータから復元."""
        steps_raw = json.loads(metadata.get("steps_json", "[]"))
        steps = tuple(VerbStep.from_dict(s) for s in steps_raw)
        return cls(
            id=chain_id,
            steps=steps,
            timestamp=metadata.get("timestamp", ""),
            emotion=metadata.get("emotion", "neutral"),
            importance=metadata.get("importance", 3),
            source=metadata.get("source", "manual"),
            context=metadata.get("context", ""),
        )


@dataclass(frozen=True)
class MemorySearchResult:
    """検索結果."""

    memory: Memory
    distance: float  # 類似度（小さいほど近い）


@dataclass(frozen=True)
class ScoredMemory:
    """スコアリング済み検索結果."""

    memory: Memory
    semantic_distance: float  # 生距離
    time_decay_factor: float  # 時間減衰係数 (0.0-1.0)
    emotion_boost: float  # 感情ブースト
    importance_boost: float  # 重要度ブースト
    final_score: float  # 最終スコア（低いほど良い）


@dataclass(frozen=True)
class MemoryStats:
    """記憶の統計情報."""

    total_count: int
    by_category: dict[str, int]
    by_emotion: dict[str, int]
    oldest_timestamp: str | None
    newest_timestamp: str | None

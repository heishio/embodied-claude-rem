"""MCP Server for AI Long-term Memory - Let AI remember across sessions!"""

import asyncio
import base64
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

import numpy as np

from .config import MemoryConfig, ServerConfig
from .episode import EpisodeManager
from .graph import MemoryGraph
from .memory import MemoryStore
from .normalizer import normalize_japanese
from .sensory import SensoryIntegration
from .types import CameraPosition, VerbChain, VerbStep
from .verb_chain import VerbChainStore, crystallize_buffer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_PRIVATE_KEY = "quo-inner-voice"


# Quadrant → flow_weight mapping (Gentner's 4-quadrant analogy)
QUADRANT_WEIGHTS: dict[str, float] = {
    "literal": 0.5,   # flow高 + delta高: 同じことを同じ対象に
    "analogy": 0.9,   # flow高 + delta低: 同じことを違う対象に（構造的類似）
    "surface": 0.1,   # flow低 + delta高: 違うことを同じ対象に（表面的類似）
    "anomaly": 0.5,   # flow低 + delta低: 広く拾う（閾値で調整）
}

QUADRANT_SCHEMA = {
    "type": "string",
    "description": "Search quadrant: 'literal' (same action, same target), 'analogy' (same action, different target), 'surface' (different action, same target). Changes how flow/delta axes are weighted.",
    "enum": ["literal", "analogy", "surface"],
}


def _quadrant_to_flow_weight(quadrant: str | None) -> float:
    """Convert quadrant name to flow_weight value."""
    if quadrant is None:
        return 0.6  # default balanced
    return QUADRANT_WEIGHTS.get(quadrant, 0.6)


def _summarize_content(text: str, max_chars: int = 50) -> str:
    """Summarize content as '最初の文。 ... 最後の文。' for compact display."""
    sentences = [s.strip() for s in text.split("。") if s.strip()]
    if not sentences:
        return text[:max_chars]
    if len(sentences) == 1:
        s = sentences[0]
        return (s[:max_chars] + "…") if len(s) > max_chars else s + "。"
    first = sentences[0]
    last = sentences[-1]
    if len(first) > max_chars:
        first = first[:max_chars] + "…"
    else:
        first = first + "。"
    if len(last) > max_chars:
        last = last[:max_chars] + "…"
    else:
        last = last + "。"
    return f"{first} ... {last}"


def _freshness_filter(freshness: float, fmin: float | None, fmax: float | None) -> bool:
    """freshness が範囲内かチェック。None は制限なし。"""
    if fmin is not None and freshness < fmin:
        return False
    if fmax is not None and freshness > fmax:
        return False
    return True


def _xor_encrypt(text: str, key: str = _PRIVATE_KEY) -> str:
    """Simple XOR + base64 encryption for private introspection files."""
    key_bytes = key.encode("utf-8")
    text_bytes = text.encode("utf-8")
    encrypted = bytes(b ^ key_bytes[i % len(key_bytes)] for i, b in enumerate(text_bytes))
    return base64.b64encode(encrypted).decode("ascii")


class MemoryMCPServer:
    """MCP Server that gives AI long-term memory."""

    def __init__(self):
        self._server = Server("memory-mcp")
        self._memory_store: MemoryStore | None = None
        self._episode_manager: EpisodeManager | None = None  # Phase 4.2
        self._sensory_integration: SensoryIntegration | None = None  # Phase 4.3
        self._verb_chain_store: VerbChainStore | None = None
        self._memory_graph: MemoryGraph | None = None
        self._server_config = ServerConfig.from_env()
        self._setup_handlers()

    def _setup_handlers(self) -> None:
        """Set up MCP tool handlers."""

        @self._server.list_tools()
        async def list_tools() -> list[Tool]:
            """List available memory tools."""
            return [
                Tool(
                    name="diary",
                    description="Write a diary entry to long-term storage. Use this to record daily experiences, thoughts, conversations, or learnings in your own words. Can also save visual memories (with image_path + camera_position) or audio memories (with audio_path + transcript).",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "The memory content to save",
                            },
                            "emotion": {
                                "type": "string",
                                "description": "Emotion tag (1-8)",
                                "default": "8",
                                "enum": ["1", "2", "3", "4", "5", "6", "7", "8"],
                            },
                            "importance": {
                                "type": "integer",
                                "description": "Importance level from 1 (trivial) to 5 (critical)",
                                "default": 3,
                                "minimum": 1,
                                "maximum": 5,
                            },
                            "category": {
                                "type": "string",
                                "description": "Category of memory",
                                "default": "daily",
                                "enum": ["daily", "philosophical", "technical", "memory", "observation", "feeling", "conversation", "curiosity"],
                            },
                            "image_path": {
                                "type": "string",
                                "description": "Path to image file (for visual memory)",
                            },
                            "camera_position": {
                                "type": "object",
                                "description": "Camera pan/tilt position (for visual memory)",
                                "properties": {
                                    "pan_angle": {"type": "integer", "description": "Pan angle (-90 to +90)"},
                                    "tilt_angle": {"type": "integer", "description": "Tilt angle (-90 to +90)"},
                                    "preset_id": {"type": "string", "description": "Preset ID (optional)"},
                                },
                                "required": ["pan_angle", "tilt_angle"],
                            },
                            "resolution": {
                                "type": "string",
                                "description": "Image resolution: 'low' (160x120), 'medium' (320x240), 'high' (640x480)",
                                "default": "medium",
                                "enum": ["low", "medium", "high"],
                            },
                            "audio_path": {
                                "type": "string",
                                "description": "Path to audio file (for audio memory)",
                            },
                            "transcript": {
                                "type": "string",
                                "description": "Transcribed text from audio",
                            },
                            "steps": {
                                "type": "array",
                                "description": "Optional verb chain steps. If provided, also saves an experience (verb chain) alongside the diary entry.",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "verb": {
                                            "type": "string",
                                            "description": "The verb (e.g., '見る')",
                                        },
                                        "nouns": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                            "description": "Associated nouns",
                                            "default": [],
                                        },
                                    },
                                    "required": ["verb"],
                                },
                            },
                        },
                        "required": ["content"],
                    },
                ),
                # search_memories: removed (merged into recall)
                Tool(
                    name="recall",
                    description="Automatically recall relevant memories based on the current conversation context. Use this to remember things that might be relevant.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "context": {
                                "type": "string",
                                "description": "Current conversation context or topic",
                            },
                            "n_results": {
                                "type": "integer",
                                "description": "Number of memories to recall",
                                "default": 5,
                                "minimum": 1,
                                "maximum": 10,
                            },
                            "emotion_filter": {
                                "type": "string",
                                "description": "Filter by emotion (1-8, optional)",
                                "enum": ["1", "2", "3", "4", "5", "6", "7", "8"],
                            },
                            "category_filter": {
                                "type": "string",
                                "description": "Filter by category (optional)",
                                "enum": ["daily", "philosophical", "technical", "memory", "observation", "feeling", "conversation", "curiosity"],
                            },
                            "date_from": {
                                "type": "string",
                                "description": "Filter memories from this date (ISO 8601 format, optional)",
                            },
                            "date_to": {
                                "type": "string",
                                "description": "Filter memories until this date (ISO 8601 format, optional)",
                            },
                            "freshness_min": {
                                "type": "number",
                                "description": "Minimum freshness (0.0-1.0). Higher = more recent.",
                                "minimum": 0.0,
                                "maximum": 1.0,
                            },
                            "freshness_max": {
                                "type": "number",
                                "description": "Maximum freshness (0.0-1.0). Lower = more distant.",
                                "minimum": 0.0,
                                "maximum": 1.0,
                            },
                            "quadrant": QUADRANT_SCHEMA,
                        },
                        "required": ["context"],
                    },
                ),
                Tool(
                    name="list_recent_memories",
                    description="List the most recent memories. Use this to see what has been remembered recently.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of memories to list",
                                "default": 10,
                                "minimum": 1,
                                "maximum": 50,
                            },
                            "category_filter": {
                                "type": "string",
                                "description": "Filter by category (optional)",
                                "enum": ["daily", "philosophical", "technical", "memory", "observation", "feeling", "conversation", "curiosity"],
                            },
                        },
                        "required": [],
                    },
                ),
                Tool(
                    name="create_category",
                    description="Create a graph category for organizing verb/noun nodes and verb chains. Categories can be hierarchical (parent/child).",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Category name (neutral context: time/place/topic, no evaluation words)",
                            },
                            "parent_id": {
                                "type": "integer",
                                "description": "Parent category ID for hierarchical organization (optional)",
                            },
                        },
                        "required": ["name"],
                    },
                ),
                Tool(
                    name="list_categories",
                    description="List all graph categories in a tree structure.",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                ),
                # --- Disabled tools (rarely used) ---
                # Tool(
                #     name="get_memory_stats",
                #     description="Get statistics about stored memories.",
                #     inputSchema={"type": "object", "properties": {}, "required": []},
                # ),
                # recall_with_associations: merged into recall with chain_depth param
                Tool(
                    name="recall_divergent",
                    description="Recall memories with divergent associative thinking. Expands memory candidates and selects them through workspace-style competition.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "context": {
                                "type": "string",
                                "description": "Current conversation context or topic",
                            },
                            "n_results": {
                                "type": "integer",
                                "description": "Number of memories to recall",
                                "default": 5,
                                "minimum": 1,
                                "maximum": 20,
                            },
                            "max_branches": {
                                "type": "integer",
                                "description": "Maximum branches per node during associative expansion",
                                "default": 3,
                                "minimum": 1,
                                "maximum": 8,
                            },
                            "max_depth": {
                                "type": "integer",
                                "description": "Maximum depth during associative expansion",
                                "default": 3,
                                "minimum": 1,
                                "maximum": 5,
                            },
                            "temperature": {
                                "type": "number",
                                "description": "Selection temperature (lower is more focused)",
                                "default": 0.7,
                                "minimum": 0.1,
                                "maximum": 2.0,
                            },
                            "include_diagnostics": {
                                "type": "boolean",
                                "description": "Include diagnostic metrics in the output",
                                "default": False,
                            },
                            "freshness_min": {
                                "type": "number",
                                "description": "Minimum freshness (0.0-1.0). Higher = more recent.",
                                "minimum": 0.0,
                                "maximum": 1.0,
                            },
                            "freshness_max": {
                                "type": "number",
                                "description": "Maximum freshness (0.0-1.0). Lower = more distant.",
                                "minimum": 0.0,
                                "maximum": 1.0,
                            },
                        },
                        "required": ["context"],
                    },
                ),
                Tool(
                    name="consolidate_memories",
                    description="Run a manual replay/consolidation cycle to strengthen associations and refresh activation metadata.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "window_hours": {
                                "type": "integer",
                                "description": "Look-back window for replay candidates in hours",
                                "default": 24,
                                "minimum": 1,
                                "maximum": 168,
                            },
                            "max_replay_events": {
                                "type": "integer",
                                "description": "Maximum replay transitions to process",
                                "default": 200,
                                "minimum": 1,
                                "maximum": 1000,
                            },
                            "link_update_strength": {
                                "type": "number",
                                "description": "Strength for coactivation/link updates",
                                "default": 0.2,
                                "minimum": 0.01,
                                "maximum": 1.0,
                            },
                            "synthesize": {
                                "type": "boolean",
                                "description": "Run composite memory synthesis after consolidation",
                                "default": True,
                            },
                            "n_layers": {
                                "type": "integer",
                                "description": "Number of noise layers for boundary detection (0=base only)",
                                "default": 3,
                                "minimum": 0,
                                "maximum": 5,
                            },
                        },
                        "required": [],
                    },
                ),
                # Verb Chain Tools
                Tool(
                    name="crystallize",
                    description="Convert the sensory buffer (keyword log) into verb chains. Groups consecutive entries by shared nouns into experience chains.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "emotion": {
                                "type": "string",
                                "description": "Emotion tag (1-8)",
                                "default": "8",
                                "enum": ["1", "2", "3", "4", "5", "6", "7", "8"],
                            },
                            "importance": {
                                "type": "integer",
                                "description": "Importance level (1-5)",
                                "default": 3,
                                "minimum": 1,
                                "maximum": 5,
                            },
                            "clear_buffer": {
                                "type": "boolean",
                                "description": "Clear the buffer after crystallizing",
                                "default": False,
                            },
                            "min_verbs": {
                                "type": "integer",
                                "description": "Minimum verb steps to keep a chain",
                                "default": 2,
                                "minimum": 1,
                            },
                            "batch_size": {
                                "type": "integer",
                                "description": "Max entries to process per call (0=all)",
                                "default": 50,
                                "minimum": 0,
                            },
                            "offset": {
                                "type": "integer",
                                "description": "Start from this entry index in the buffer",
                                "default": 0,
                                "minimum": 0,
                            },
                            "graph_category": {
                                "type": "integer",
                                "description": "Graph category ID to assign crystallized chains to (optional)",
                            },
                            "merge_threshold": {
                                "type": "number",
                                "description": "Shared noun ratio threshold for merging entries into same chain (0.0-1.0). Higher = stricter splitting. Default 0.2.",
                                "default": 0.2,
                                "minimum": 0.0,
                                "maximum": 1.0,
                            },
                        },
                        "required": [],
                    },
                ),
                # remember_experience: removed (use diary with steps parameter)
                Tool(
                    name="recall_experience",
                    description="Recall verb chains (experiences) by semantic similarity. Uses time decay, emotion boost, and importance scoring.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "context": {
                                "type": "string",
                                "description": "What you want to remember about",
                            },
                            "n_results": {
                                "type": "integer",
                                "description": "Number of experiences to recall",
                                "default": 5,
                                "minimum": 1,
                                "maximum": 20,
                            },
                            "graph_category": {
                                "type": "integer",
                                "description": "Filter by graph category ID (includes subcategories)",
                            },
                            "freshness_min": {
                                "type": "number",
                                "description": "Minimum freshness (0.0-1.0). Higher = more recent.",
                                "minimum": 0.0,
                                "maximum": 1.0,
                            },
                            "freshness_max": {
                                "type": "number",
                                "description": "Maximum freshness (0.0-1.0). Lower = more distant.",
                                "minimum": 0.0,
                                "maximum": 1.0,
                            },
                            "quadrant": QUADRANT_SCHEMA,
                        },
                        "required": ["context"],
                    },
                ),
                # recall_by_verb: removed (graph expansion used internally, quadrant covers recall use cases)
                # dream: removed (buffer check covered by crystallize)
                # Update diary
                Tool(
                    name="update_diary",
                    description="Update an existing diary entry with strikethrough + amendment. Original content is preserved with ~~strikethrough~~ and the amendment is appended. Use this to correct or add context to past memories without deleting them.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "memory_id": {
                                "type": "string",
                                "description": "The ID of the diary entry to update",
                            },
                            "amendment": {
                                "type": "string",
                                "description": "The amendment text to append",
                            },
                            "emotion": {
                                "type": "string",
                                "description": "New emotion tag (1-8, optional)",
                                "enum": ["1", "2", "3", "4", "5", "6", "7", "8"],
                            },
                            "importance": {
                                "type": "integer",
                                "description": "New importance level (1-5, optional)",
                                "minimum": 1,
                                "maximum": 5,
                            },
                        },
                        "required": ["memory_id", "amendment"],
                    },
                ),
                # Recall Index
                Tool(
                    name="rebuild_recall_index",
                    description="Rebuild the recall index (pre-computed word→memory similarity table). Run this after bulk imports or if the index seems stale.",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                    },
                ),
            ]

        @self._server.call_tool()
        async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
            """Handle tool calls."""
            if self._memory_store is None:
                return [TextContent(type="text", text="Error: Memory store not connected")]

            try:
                match name:
                    case "diary":
                        content = arguments.get("content", "")
                        if not content:
                            return [TextContent(type="text", text="Error: content is required")]

                        image_path = arguments.get("image_path")
                        camera_pos_data = arguments.get("camera_position")
                        audio_path = arguments.get("audio_path")
                        transcript = arguments.get("transcript")

                        # Visual memory path
                        if image_path and camera_pos_data:
                            if self._sensory_integration is None:
                                return [TextContent(type="text", text="Error: Sensory integration not initialized")]
                            camera_position = CameraPosition(
                                pan_angle=camera_pos_data["pan_angle"],
                                tilt_angle=camera_pos_data["tilt_angle"],
                                preset_id=camera_pos_data.get("preset_id"),
                            )
                            memory = await self._sensory_integration.save_visual_memory(
                                content=content,
                                image_path=image_path,
                                camera_position=camera_position,
                                emotion=arguments.get("emotion", "8"),
                                importance=arguments.get("importance", 3),
                                resolution=arguments.get("resolution"),
                            )
                            # Update recall index for new memory
                            try:
                                await self._memory_store.update_recall_index(memory.id, "memory")
                            except Exception:
                                pass
                            return [
                                TextContent(
                                    type="text",
                                    text=f"Visual memory saved!\nID: {memory.id}\nContent: {memory.content}\nImage: {image_path}\nCamera: pan={camera_position.pan_angle}°, tilt={camera_position.tilt_angle}°\nEmotion: {memory.emotion} | Importance: {memory.importance}",
                                )
                            ]

                        # Audio memory path
                        if audio_path and transcript:
                            if self._sensory_integration is None:
                                return [TextContent(type="text", text="Error: Sensory integration not initialized")]
                            memory = await self._sensory_integration.save_audio_memory(
                                content=content,
                                audio_path=audio_path,
                                transcript=transcript,
                                emotion=arguments.get("emotion", "8"),
                                importance=arguments.get("importance", 3),
                            )
                            # Update recall index for new memory
                            try:
                                await self._memory_store.update_recall_index(memory.id, "memory")
                            except Exception:
                                pass
                            return [
                                TextContent(
                                    type="text",
                                    text=f"Audio memory saved!\nID: {memory.id}\nContent: {memory.content}\nAudio: {audio_path}\nTranscript: {transcript}\nEmotion: {memory.emotion} | Importance: {memory.importance}",
                                )
                            ]

                        # Standard text memory path
                        memory = await self._memory_store.save(
                            content=content,
                            emotion=arguments.get("emotion", "8"),
                            importance=arguments.get("importance", 3),
                            category=arguments.get("category", "daily"),
                        )

                        # Update recall index for new memory
                        try:
                            await self._memory_store.update_recall_index(memory.id, "memory")
                        except Exception:
                            pass

                        # Also save verb chain if steps provided
                        chain_info = ""
                        steps_raw = arguments.get("steps")
                        if steps_raw and self._verb_chain_store:
                            steps = tuple(
                                VerbStep(
                                    verb=s["verb"],
                                    nouns=tuple(s.get("nouns", [])),
                                )
                                for s in steps_raw
                            )
                            chain = VerbChain(
                                id=str(uuid.uuid4()),
                                steps=steps,
                                timestamp=datetime.now(timezone.utc).isoformat(),
                                emotion=arguments.get("emotion", "8"),
                                importance=arguments.get("importance", 3),
                                source="manual",
                                context=content,
                            )
                            await self._verb_chain_store.save(chain)
                            try:
                                await self._memory_store.update_recall_index(chain.id, "chain")
                            except Exception:
                                pass
                            chain_info = f"\nExperience also saved! Chain: {chain.to_document()}\nSteps: {len(chain.steps)}"

                        return [
                            TextContent(
                                type="text",
                                text=f"Diary entry saved!\nID: {memory.id}\nTimestamp: {memory.timestamp}\nEmotion: {memory.emotion}\nImportance: {memory.importance}\nCategory: {memory.category}{chain_info}",
                            )
                        ]

                    # search_memories: removed (merged into recall)

                    case "recall":
                        context = arguments.get("context", "")
                        if not context:
                            return [TextContent(type="text", text="Error: context is required")]

                        n_results = arguments.get("n_results", 5)
                        fmin = arguments.get("freshness_min")
                        fmax = arguments.get("freshness_max")
                        fw = _quadrant_to_flow_weight(arguments.get("quadrant"))
                        emotion_filter = arguments.get("emotion_filter")
                        category_filter = arguments.get("category_filter")
                        date_from = arguments.get("date_from")
                        date_to = arguments.get("date_to")

                        summary_count = max(0, 10 - n_results)
                        total_fetch = n_results + summary_count

                        results = await self._memory_store.recall(
                            context=context,
                            n_results=total_fetch * (3 if fmin or fmax else 1),
                            flow_weight=fw,
                            emotion_filter=emotion_filter,
                            category_filter=category_filter,
                            date_from=date_from,
                            date_to=date_to,
                        )
                        if fmin is not None or fmax is not None:
                            results = [r for r in results if _freshness_filter(r.memory.freshness, fmin, fmax)]
                            results = results[:total_fetch]

                        if not results:
                            return [TextContent(type="text", text="No relevant memories found.")]

                        detail_results = results[:n_results]
                        summary_results = results[n_results:]

                        output_lines = [f"Recalled {len(detail_results)} relevant memories:\n"]
                        for i, result in enumerate(detail_results, 1):
                            m = result.memory
                            image_line = ""
                            for sd in m.sensory_data:
                                if sd.sensory_type == "visual" and sd.image_data:
                                    image_line = f"Image: data:image/jpeg;base64,{sd.image_data}\n"
                                    break
                            output_lines.append(
                                f"--- Memory {i} ---\n"
                                f"ID: {m.id}\n"
                                f"[{m.freshness:.2f}] [{m.emotion}]\n"
                                f"{m.content}\n"
                                f"{image_line}"
                            )

                        if summary_results:
                            summary_lines = []
                            for result in summary_results:
                                m = result.memory
                                if not m.content:
                                    continue
                                summary = _summarize_content(m.content)
                                summary_lines.append(f"- [{m.freshness:.2f}] {summary}")
                            if summary_lines:
                                output_lines.append("\n--- Also recalled (summary) ---")
                                output_lines.extend(summary_lines)

                        return [TextContent(type="text", text="\n".join(output_lines))]

                    case "list_recent_memories":
                        memories = await self._memory_store.list_recent(
                            limit=arguments.get("limit", 10),
                            category_filter=arguments.get("category_filter"),
                        )

                        if not memories:
                            return [TextContent(type="text", text="No memories found.")]

                        output_lines = [f"Recent {len(memories)} memories:\n"]
                        for i, m in enumerate(memories, 1):
                            output_lines.append(
                                f"--- Memory {i} ---\n"
                                f"ID: {m.id}\n"
                                f"[{m.freshness:.2f}] [{m.emotion}] [{m.category}]\n"
                                f"{m.content}\n"
                            )

                        return [TextContent(type="text", text="\n".join(output_lines))]

                    case "recall_divergent":
                        context = arguments.get("context", "")
                        if not context:
                            return [TextContent(type="text", text="Error: context is required")]

                        fmin = arguments.get("freshness_min")
                        fmax = arguments.get("freshness_max")
                        n_results_div = arguments.get("n_results", 5)
                        results, diagnostics = await self._memory_store.recall_divergent(
                            context=context,
                            n_results=n_results_div * (3 if fmin or fmax else 1),
                            max_branches=arguments.get("max_branches", 3),
                            max_depth=arguments.get("max_depth", 3),
                            temperature=arguments.get("temperature", 0.7),
                            include_diagnostics=arguments.get("include_diagnostics", False),
                        )
                        if fmin is not None or fmax is not None:
                            results = [r for r in results if _freshness_filter(r.memory.freshness, fmin, fmax)]
                            results = results[:n_results_div]

                        if not results:
                            return [TextContent(type="text", text="No relevant memories found.")]

                        output_lines = [f"Divergent recall returned {len(results)} memories:\n"]
                        for i, result in enumerate(results, 1):
                            m = result.memory
                            output_lines.append(
                                f"--- Memory {i} (score: {result.distance:.4f}) ---\n"
                                f"ID: {m.id}\n"
                                f"[{m.freshness:.2f}] [{m.emotion}] [{m.category}]\n"
                                f"{m.content}\n"
                            )

                        if arguments.get("include_diagnostics", False):
                            output_lines.append(
                                "\n=== Diagnostics ===\n"
                                f"{json.dumps(diagnostics, indent=2, ensure_ascii=False)}"
                            )

                        return [TextContent(type="text", text="\n".join(output_lines))]

                    case "consolidate_memories":
                        stats = await self._memory_store.consolidate_memories(
                            window_hours=arguments.get("window_hours", 24),
                            max_replay_events=arguments.get("max_replay_events", 200),
                            link_update_strength=arguments.get("link_update_strength", 0.2),
                            synthesize=arguments.get("synthesize", True),
                            n_layers=arguments.get("n_layers", 3),
                            graph=self._memory_graph,
                        )

                        # Graph consolidation (decay + prune)
                        if self._memory_graph is not None:
                            try:
                                graph_stats = await self._memory_graph.consolidate()
                                stats.update(graph_stats)
                            except Exception as e:
                                stats["graph_error"] = str(e)

                        # Core memory compaction → MEMORY.md
                        try:
                            from .compaction import compact_core_memories
                            config = MemoryConfig.from_env()
                            compaction_stats = compact_core_memories(
                                db_path=config.db_path,
                                memory_md_path=config.memory_md_path,
                            )
                            stats["compaction"] = compaction_stats
                        except Exception as e:
                            stats["compaction_error"] = str(e)

                        return [
                            TextContent(
                                type="text",
                                text="Consolidation completed:\n"
                                f"{json.dumps(stats, indent=2, ensure_ascii=False)}",
                            )
                        ]

                    # Category tools
                    case "create_category":
                        cat_name = arguments.get("name", "")
                        if not cat_name:
                            return [TextContent(type="text", text="Error: name is required")]

                        if self._memory_graph is None:
                            return [TextContent(type="text", text="Error: Memory graph not initialized")]

                        parent_id = arguments.get("parent_id")
                        cat_id = await self._memory_graph.create_category(
                            name=cat_name, parent_id=parent_id
                        )

                        parent_info = f" (parent: {parent_id})" if parent_id else ""
                        return [
                            TextContent(
                                type="text",
                                text=f"Category created!\nID: {cat_id}\nName: {cat_name}{parent_info}",
                            )
                        ]

                    case "list_categories":
                        if self._memory_graph is None:
                            return [TextContent(type="text", text="Error: Memory graph not initialized")]

                        categories = await self._memory_graph.list_categories()

                        if not categories:
                            return [TextContent(type="text", text="No categories found.")]

                        # Build tree display
                        children_map: dict[int | None, list[dict]] = {}
                        for cat in categories:
                            pid = cat["parent_id"]
                            children_map.setdefault(pid, []).append(cat)

                        def _render_tree(parent_id: int | None, indent: int) -> list[str]:
                            lines: list[str] = []
                            for cat in children_map.get(parent_id, []):
                                prefix = "  " * indent + ("└─ " if indent > 0 else "")
                                lines.append(f"{prefix}[{cat['id']}] {cat['name']}")
                                lines.extend(_render_tree(cat["id"], indent + 1))
                            return lines

                        output_lines = ["## Categories\n"] + _render_tree(None, 0)
                        return [TextContent(type="text", text="\n".join(output_lines))]

                    # Verb Chain Tools
                    case "crystallize":
                        buf_path = os.path.join(os.path.expanduser("~"), ".claude", "sensory_buffer.jsonl")
                        if not os.path.exists(buf_path):
                            return [TextContent(type="text", text="バッファは空です。結晶化する材料がありません。")]

                        all_entries: list[dict[str, Any]] = []
                        with open(buf_path, "r", encoding="utf-8") as f:
                            for line in f:
                                line = line.strip()
                                if not line:
                                    continue
                                try:
                                    all_entries.append(json.loads(line))
                                except json.JSONDecodeError:
                                    continue

                        if not all_entries:
                            return [TextContent(type="text", text="バッファは空です。")]

                        total_entries = len(all_entries)
                        offset = arguments.get("offset", 0)
                        batch_size = arguments.get("batch_size", 50)

                        # offset適用
                        if offset >= total_entries:
                            return [TextContent(type="text", text=f"オフセット({offset})がバッファサイズ({total_entries})を超えています。")]

                        entries = all_entries[offset:]

                        # batch_size適用（0=全件）
                        remaining = 0
                        if batch_size > 0 and len(entries) > batch_size:
                            remaining = len(entries) - batch_size
                            entries = entries[:batch_size]

                        chains = crystallize_buffer(
                            entries=entries,
                            emotion=arguments.get("emotion", "8"),
                            importance=arguments.get("importance", 3),
                            min_verbs=arguments.get("min_verbs", 2),
                            merge_threshold=arguments.get("merge_threshold", 0.2),
                        )

                        if not chains:
                            return [TextContent(type="text", text="動詞が足りないため、チェーンを生成できませんでした。")]

                        verb_chain_store = self._verb_chain_store
                        graph_category = arguments.get("graph_category")
                        for chain in chains:
                            await verb_chain_store.save(chain, category_id=graph_category)

                        # clear_buffer: バッチ処理中（remaining > 0）はクリアしない
                        cleared = False
                        if arguments.get("clear_buffer", False) and remaining == 0:
                            os.remove(buf_path)
                            cleared = True

                        # サマリー出力（代表チェーン最大5つ + 統計）
                        max_preview = 5
                        output_lines = [
                            f"## 結晶化完了: {len(chains)} チェーン保存",
                            f"処理: {len(entries)}/{total_entries} エントリ (offset={offset})\n",
                        ]
                        for i, chain in enumerate(chains[:max_preview], 1):
                            # 動詞の流れだけ簡潔に表示
                            verb_flow = "→".join(s.verb for s in chain.steps)
                            top_nouns = set()
                            for s in chain.steps:
                                top_nouns.update(s.nouns)
                            nouns_str = ", ".join(list(top_nouns)[:8])
                            output_lines.append(f"  {i}. {verb_flow}")
                            if nouns_str:
                                output_lines.append(f"     ({nouns_str})")

                        if len(chains) > max_preview:
                            output_lines.append(f"  ... 他 {len(chains) - max_preview} チェーン")

                        if remaining > 0:
                            next_offset = offset + batch_size
                            output_lines.append(f"\n残り {remaining} エントリ → offset={next_offset} で続行")

                        if cleared:
                            output_lines.append("\n(バッファをクリアしました)")

                        return [TextContent(type="text", text="\n".join(output_lines))]

                    # remember_experience: removed (use diary with steps)

                    case "recall_experience":
                        context = arguments.get("context", "")
                        if not context:
                            return [TextContent(type="text", text="Error: context is required")]

                        fmin = arguments.get("freshness_min")
                        fmax = arguments.get("freshness_max")
                        fw = _quadrant_to_flow_weight(arguments.get("quadrant"))
                        n_results_exp = arguments.get("n_results", 5)
                        verb_chain_store = self._verb_chain_store
                        results = await verb_chain_store.search(
                            query=context,
                            n_results=n_results_exp * (3 if fmin or fmax else 1),
                            category_id=arguments.get("graph_category"),
                            flow_weight=fw,
                        )
                        if fmin is not None or fmax is not None:
                            results = [(c, s) for c, s in results if _freshness_filter(c.freshness, fmin, fmax)]
                            results = results[:n_results_exp]

                        # Filter out chains with very short context (< 10 chars)
                        # Keep them as fallback if not enough rich results
                        rich = [(c, s) for c, s in results if len(c.context or "") >= 10]
                        if len(rich) >= n_results_exp:
                            results = rich[:n_results_exp]
                        elif rich:
                            # Fill remaining slots with short-context chains
                            short = [(c, s) for c, s in results if len(c.context or "") < 10]
                            results = rich + short[:n_results_exp - len(rich)]

                        if not results:
                            return [TextContent(type="text", text="No experiences found.")]

                        # Boundary-aware rerank (fuzziness-based, no path)
                        try:
                            chain_ids = [c.id for c, _ in results]
                            boundary_scores = await self._memory_store.get_chain_boundary_scores(
                                chain_ids, layer_index=None,
                            )
                            BOUNDARY_WEIGHT = 0.05
                            results = [
                                (chain, score - BOUNDARY_WEIGHT * boundary_scores.get(chain.id, 0.0))
                                for chain, score in results
                            ]
                            results.sort(key=lambda x: x[1])
                        except Exception:
                            pass  # boundary データがなくても既存動作を維持

                        # Bump graph edges for recalled chains
                        for chain, _ in results:
                            try:
                                await verb_chain_store.bump_chain_edges(chain)
                            except Exception:
                                pass

                        output_lines = [f"Recalled {len(results)} experiences:\n"]
                        for i, (chain, score) in enumerate(results, 1):
                            output_lines.append(
                                f"--- Experience {i} (score: {score:.4f}) ---\n"
                                f"ID: {chain.id}\n"
                                f"[{chain.freshness:.2f}] [{chain.emotion}] (importance: {chain.importance})\n"
                                f"Chain: {chain.to_document()}\n"
                            )

                        return [TextContent(type="text", text="\n".join(output_lines))]

                    # recall_by_verb: removed (graph expansion used internally)

                    # dream: removed (buffer check covered by crystallize)

                    case "update_diary":
                        mid = arguments.get("memory_id", "")
                        amendment = arguments.get("amendment", "")
                        if not mid or not amendment:
                            return [TextContent(type="text", text="Error: memory_id and amendment are required")]

                        updated = await self._memory_store.update_diary_content(
                            memory_id=mid,
                            amendment=amendment,
                            emotion=arguments.get("emotion"),
                            importance=arguments.get("importance"),
                        )
                        if updated is None:
                            return [TextContent(type="text", text=f"Error: memory {mid} not found")]

                        # Update recall index for the updated memory
                        try:
                            await self._memory_store.update_recall_index(mid, "memory")
                        except Exception:
                            pass

                        return [
                            TextContent(
                                type="text",
                                text=f"Diary updated!\nID: {updated.id}\nContent:\n{updated.content}\nEmotion: {updated.emotion} | Importance: {updated.importance}",
                            )
                        ]

                    case "rebuild_recall_index":
                        count = await self._memory_store.rebuild_recall_index_full()
                        return [
                            TextContent(
                                type="text",
                                text=f"Recall index fully rebuilt: {count} entries",
                            )
                        ]

                    case _:
                        return [TextContent(type="text", text=f"Unknown tool: {name}")]

            except Exception as e:
                logger.exception(f"Error in tool {name}")
                return [TextContent(type="text", text=f"Error: {e!s}")]

    async def connect_memory(self) -> None:
        """Connect to memory store (chiVe 2-vector backend)."""
        config = MemoryConfig.from_env()

        # Initialize chiVe embedding (shared by MemoryStore and VerbChainStore)
        from .chive import ChiVeEmbedding
        chive = ChiVeEmbedding(config.chive_model_path)
        await asyncio.to_thread(chive._load)
        logger.info("chiVe embedding loaded (%d dims)", chive.vector_size)

        self._memory_store = MemoryStore(config, chive=chive)
        await self._memory_store.connect()
        logger.info(f"Connected to memory store at {config.db_path}")

        # Run chiVe 2-vector migration if needed
        migration_stats = await self._memory_store.migrate_to_chive_2vec()
        if migration_stats["memories_migrated"] > 0 or migration_stats["chains_migrated"] > 0:
            logger.info(
                "chiVe migration: %d memories, %d chains migrated",
                migration_stats["memories_migrated"],
                migration_stats["chains_migrated"],
            )

        # Episode manager (delegating to MemoryStore)
        self._episode_manager = EpisodeManager(self._memory_store)
        logger.info("Episode manager initialized")

        # Memory graph (weighted verb/noun co-occurrence)
        self._memory_graph = MemoryGraph(db=self._memory_store.db)
        logger.info("Memory graph initialized")

        # VerbChainStore (shares DB and chiVe with MemoryStore)
        self._verb_chain_store = VerbChainStore(
            db=self._memory_store.db,
            chive=chive,
            graph=self._memory_graph,
        )
        await self._verb_chain_store.initialize()
        logger.info("Verb chain store initialized")

        # Sensory integration
        self._sensory_integration = SensoryIntegration(self._memory_store)
        logger.info("Sensory integration initialized")

        # Recall index (pre-computed word→memory similarity)
        try:
            count = await self._memory_store.build_recall_index()
            logger.info("Recall index built: %d entries", count)
        except Exception as e:
            logger.warning("Recall index build failed (non-fatal): %s", e)

    async def disconnect_memory(self) -> None:
        """Disconnect from memory store."""
        if self._memory_store:
            await self._memory_store.disconnect()
            self._memory_store = None
            self._verb_chain_store = None
            self._memory_graph = None
            logger.info("Disconnected from memory store")

    @asynccontextmanager
    async def run_context(self):
        """Context manager for server lifecycle."""
        try:
            await self.connect_memory()
            yield
        finally:
            await self.disconnect_memory()

    async def run(self) -> None:
        """Run the MCP server."""
        async with self.run_context():
            async with stdio_server() as (read_stream, write_stream):
                await self._server.run(
                    read_stream,
                    write_stream,
                    self._server.create_initialization_options(),
                )


def main() -> None:
    """Entry point for the MCP server."""
    server = MemoryMCPServer()
    asyncio.run(server.run())


if __name__ == "__main__":
    main()

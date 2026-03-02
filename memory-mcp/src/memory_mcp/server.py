"""MCP Server for AI Long-term Memory - Let AI remember across sessions!"""

import asyncio
import base64
import json
import logging
import os
import tempfile
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
                            "auto_link": {
                                "type": "boolean",
                                "description": "Automatically link to similar existing memories",
                                "default": True,
                            },
                            "link_threshold": {
                                "type": "number",
                                "description": "Similarity threshold for auto-linking (0-2, lower means more similar required)",
                                "default": 0.8,
                                "minimum": 0,
                                "maximum": 2,
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
                Tool(
                    name="search_memories",
                    description="Search through memories using semantic similarity. Find memories related to a topic or query.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query to find related memories",
                            },
                            "n_results": {
                                "type": "integer",
                                "description": "Maximum number of results to return",
                                "default": 5,
                                "minimum": 1,
                                "maximum": 20,
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
                        },
                        "required": ["query"],
                    },
                ),
                Tool(
                    name="recall",
                    description="Automatically recall relevant memories based on the current conversation context. Use this to remember things that might be relevant. Set chain_depth >= 1 to also include linked/associated memories.",
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
                                "default": 3,
                                "minimum": 1,
                                "maximum": 10,
                            },
                            "chain_depth": {
                                "type": "integer",
                                "description": "How many levels of linked memories to follow (0=none, 1-3=with associations)",
                                "default": 0,
                                "minimum": 0,
                                "maximum": 3,
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
                # get_association_diagnostics: disabled (debug only)
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
                # get_memory_chain: disabled (rarely used)
                # Episode tools: disabled (use MEMORY.md for diary-like entries)
                # create_episode, search_episodes, get_episode_memories
                # save_visual_memory, save_audio_memory: merged into remember
                # recall_by_camera_position: disabled (rarely used)
                # get_working_memory, refresh_working_memory: disabled (rarely used)
                # Phase 5: Causal Links — link_memories: disabled (verb chains cover causality)
                # get_causal_chain: disabled (rarely used)
                Tool(
                    name="tom",
                    description="Theory of Mind: perspective-taking tool. Call this BEFORE responding to understand what the other person is feeling and wanting. Projects your simulated emotions onto them, then swaps perspectives.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "situation": {
                                "type": "string",
                                "description": "What the other person said or did (their message/action)",
                            },
                            "person": {
                                "type": "string",
                                "description": f"Who you are talking to (default: {self._server_config.tom_default_person})",
                                "default": self._server_config.tom_default_person,
                            },
                            "private": {
                                "type": "boolean",
                                "description": "If true, write result to a temp file and return only the file path (for private introspection)",
                                "default": False,
                            },
                        },
                        "required": ["situation"],
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
                Tool(
                    name="remember_experience",
                    description="Manually create a verb chain (experience). Use this to structure an experience as a sequence of verbs with associated nouns.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "steps": {
                                "type": "array",
                                "description": "Sequence of verb steps",
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
                                            "description": "Associated nouns (e.g., ['コウタ', 'キーボード'])",
                                            "default": [],
                                        },
                                    },
                                    "required": ["verb"],
                                },
                            },
                            "context": {
                                "type": "string",
                                "description": "Free-form context/description for this experience",
                                "default": "",
                            },
                            "emotion": {
                                "type": "string",
                                "description": "Emotion tag (1-8)",
                                "default": "8",
                                "enum": ["1", "2", "3", "4", "5", "6", "7", "8"],
                            },
                            "importance": {
                                "type": "integer",
                                "description": "Importance (1-5)",
                                "default": 3,
                                "minimum": 1,
                                "maximum": 5,
                            },
                            "graph_category": {
                                "type": "integer",
                                "description": "Graph category ID to assign this experience to (optional)",
                            },
                        },
                        "required": ["steps"],
                    },
                ),
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
                        },
                        "required": ["context"],
                    },
                ),
                Tool(
                    name="recall_by_verb",
                    description="Recall experiences starting from a specific verb or noun. Expands associatively through shared verbs/nouns across chains.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "verb": {
                                "type": "string",
                                "description": "A verb to start from (e.g., '見る')",
                            },
                            "verb2": {
                                "type": "string",
                                "description": "A second verb for bigram search (e.g., '驚く'). When used with verb, searches for the verb pair 'verb→verb2' in chain sequences.",
                            },
                            "noun": {
                                "type": "string",
                                "description": "A noun to start from (e.g., 'コウタ')",
                            },
                            "depth": {
                                "type": "integer",
                                "description": "How many expansion levels (1-5)",
                                "default": 2,
                                "minimum": 1,
                                "maximum": 5,
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
                        },
                        "required": [],
                    },
                ),
                # Sensory Buffer / Dream
                Tool(
                    name="dream",
                    description="Review the sensory buffer (rough keyword log from conversations). Use this to 'dream' - find patterns in accumulated keywords, then promote important ones to diary entries or experiences.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "clear": {
                                "type": "boolean",
                                "description": "Clear the buffer after reading (default: false)",
                                "default": False,
                            },
                        },
                    },
                ),
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
                        auto_link = arguments.get("auto_link", True)

                        if auto_link:
                            memory = await self._memory_store.save_with_auto_link(
                                content=content,
                                emotion=arguments.get("emotion", "8"),
                                importance=arguments.get("importance", 3),
                                category=arguments.get("category", "daily"),
                                link_threshold=arguments.get("link_threshold", 0.8),
                            )
                            linked_info = f"\nLinked to: {len(memory.linked_ids)} memories"
                        else:
                            memory = await self._memory_store.save(
                                content=content,
                                emotion=arguments.get("emotion", "8"),
                                importance=arguments.get("importance", 3),
                                category=arguments.get("category", "daily"),
                            )
                            linked_info = ""

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
                                text=f"Diary entry saved!\nID: {memory.id}\nTimestamp: {memory.timestamp}\nEmotion: {memory.emotion}\nImportance: {memory.importance}\nCategory: {memory.category}{linked_info}{chain_info}",
                            )
                        ]

                    case "search_memories":
                        query = arguments.get("query", "")
                        if not query:
                            return [TextContent(type="text", text="Error: query is required")]

                        fmin = arguments.get("freshness_min")
                        fmax = arguments.get("freshness_max")
                        results = await self._memory_store.search(
                            query=query,
                            n_results=arguments.get("n_results", 5) * (3 if fmin or fmax else 1),
                            emotion_filter=arguments.get("emotion_filter"),
                            category_filter=arguments.get("category_filter"),
                            date_from=arguments.get("date_from"),
                            date_to=arguments.get("date_to"),
                        )
                        if fmin is not None or fmax is not None:
                            results = [r for r in results if _freshness_filter(r.memory.freshness, fmin, fmax)]
                            results = results[:arguments.get("n_results", 5)]

                        if not results:
                            return [TextContent(type="text", text="No memories found matching the query.")]

                        output_lines = [f"Found {len(results)} memories:\n"]
                        for i, result in enumerate(results, 1):
                            m = result.memory
                            image_line = ""
                            for sd in m.sensory_data:
                                if sd.sensory_type == "visual" and sd.image_data:
                                    image_line = f"Image: data:image/jpeg;base64,{sd.image_data}\n"
                                    break
                            output_lines.append(
                                f"--- Memory {i} (distance: {result.distance:.4f}) ---\n"
                                f"ID: {m.id}\n"
                                f"[{m.freshness:.2f}] [{m.emotion}] [{m.category}] (importance: {m.importance})\n"
                                f"{m.content}\n"
                                f"{image_line}"
                            )

                        return [TextContent(type="text", text="\n".join(output_lines))]

                    case "recall":
                        context = arguments.get("context", "")
                        if not context:
                            return [TextContent(type="text", text="Error: context is required")]

                        chain_depth = arguments.get("chain_depth", 0)
                        n_results = arguments.get("n_results", 3)
                        fmin = arguments.get("freshness_min")
                        fmax = arguments.get("freshness_max")

                        if chain_depth >= 1:
                            # With associations (merged recall_with_associations)
                            results = await self._memory_store.recall_with_chain(
                                context=context,
                                n_results=n_results * (3 if fmin or fmax else 1),
                                chain_depth=chain_depth,
                            )
                            if fmin is not None or fmax is not None:
                                results = [r for r in results if _freshness_filter(r.memory.freshness, fmin, fmax)]
                                results = results[:n_results]

                            if not results:
                                return [TextContent(type="text", text="No relevant memories found.")]

                            main_results = [r for r in results if r.distance < 900]
                            linked_results = [r for r in results if r.distance >= 900]

                            output_lines = [f"Recalled {len(main_results)} memories with {len(linked_results)} linked associations:\n"]

                            output_lines.append("=== Primary Memories ===\n")
                            for i, result in enumerate(main_results, 1):
                                m = result.memory
                                output_lines.append(
                                    f"--- Memory {i} (score: {result.distance:.4f}) ---\n"
                                    f"ID: {m.id}\n"
                                    f"[{m.freshness:.2f}] [{m.emotion}]\n"
                                    f"{m.content}\n"
                                )

                            if linked_results:
                                output_lines.append("\n=== Linked Memories ===\n")
                                for i, result in enumerate(linked_results, 1):
                                    m = result.memory
                                    output_lines.append(
                                        f"--- Linked {i} ---\n"
                                        f"ID: {m.id}\n"
                                        f"[{m.freshness:.2f}] [{m.emotion}]\n"
                                        f"{m.content}\n"
                                    )

                            return [TextContent(type="text", text="\n".join(output_lines))]

                        # Standard recall (no associations)
                        results = await self._memory_store.recall(
                            context=context,
                            n_results=n_results * (3 if fmin or fmax else 1),
                        )
                        if fmin is not None or fmax is not None:
                            results = [r for r in results if _freshness_filter(r.memory.freshness, fmin, fmax)]
                            results = results[:n_results]

                        if not results:
                            return [TextContent(type="text", text="No relevant memories found.")]

                        output_lines = [f"Recalled {len(results)} relevant memories:\n"]
                        for i, result in enumerate(results, 1):
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

                    case "get_memory_stats":
                        stats = await self._memory_store.get_stats()

                        output = f"""Memory Statistics:
Total Memories: {stats.total_count}

By Category:
{json.dumps(stats.by_category, indent=2, ensure_ascii=False)}

By Emotion:
{json.dumps(stats.by_emotion, indent=2, ensure_ascii=False)}

Date Range:
  Oldest: {stats.oldest_timestamp or 'N/A'}
  Newest: {stats.newest_timestamp or 'N/A'}
"""
                        return [TextContent(type="text", text=output)]

                    case "recall_with_associations":
                        context = arguments.get("context", "")
                        if not context:
                            return [TextContent(type="text", text="Error: context is required")]

                        results = await self._memory_store.recall_with_chain(
                            context=context,
                            n_results=arguments.get("n_results", 3),
                            chain_depth=arguments.get("chain_depth", 1),
                        )

                        if not results:
                            return [TextContent(type="text", text="No relevant memories found.")]

                        # メイン結果と関連結果を分ける
                        main_results = [r for r in results if r.distance < 900]
                        linked_results = [r for r in results if r.distance >= 900]

                        output_lines = [f"Recalled {len(main_results)} memories with {len(linked_results)} linked associations:\n"]

                        output_lines.append("=== Primary Memories ===\n")
                        for i, result in enumerate(main_results, 1):
                            m = result.memory
                            output_lines.append(
                                f"--- Memory {i} (score: {result.distance:.4f}) ---\n"
                                f"ID: {m.id}\n"
                                f"[{m.freshness:.2f}] [{m.emotion}]\n"
                                f"{m.content}\n"
                            )

                        if linked_results:
                            output_lines.append("\n=== Linked Memories ===\n")
                            for i, result in enumerate(linked_results, 1):
                                m = result.memory
                                output_lines.append(
                                    f"--- Linked {i} ---\n"
                                    f"ID: {m.id}\n"
                                    f"[{m.freshness:.2f}] [{m.emotion}]\n"
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

                    case "get_association_diagnostics":
                        context = arguments.get("context", "")
                        if not context:
                            return [TextContent(type="text", text="Error: context is required")]

                        diagnostics = await self._memory_store.get_association_diagnostics(
                            context=context,
                            sample_size=arguments.get("sample_size", 20),
                        )

                        return [
                            TextContent(
                                type="text",
                                text="Association diagnostics:\n"
                                f"{json.dumps(diagnostics, indent=2, ensure_ascii=False)}",
                            )
                        ]

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

                    case "get_memory_chain":
                        memory_id = arguments.get("memory_id", "")
                        if not memory_id:
                            return [TextContent(type="text", text="Error: memory_id is required")]

                        # 起点の記憶を取得
                        start_memory = await self._memory_store.get_by_id(memory_id)
                        if not start_memory:
                            return [TextContent(type="text", text="Error: Memory not found")]

                        linked_memories = await self._memory_store.get_linked_memories(
                            memory_id=memory_id,
                            depth=arguments.get("depth", 2),
                        )

                        output_lines = [f"Memory chain starting from {memory_id}:\n"]

                        output_lines.append("=== Starting Memory ===\n")
                        output_lines.append(
                            f"ID: {start_memory.id}\n"
                            f"[{start_memory.freshness:.2f}] [{start_memory.emotion}] [{start_memory.category}]\n"
                            f"{start_memory.content}\n"
                            f"Linked to: {len(start_memory.linked_ids)} memories\n"
                        )

                        if linked_memories:
                            output_lines.append(f"\n=== Linked Memories ({len(linked_memories)}) ===\n")
                            for i, m in enumerate(linked_memories, 1):
                                output_lines.append(
                                    f"--- {i}. {m.id[:8]}... ---\n"
                                    f"[{m.freshness:.2f}] [{m.emotion}]\n"
                                    f"{m.content}\n"
                                )
                        else:
                            output_lines.append("\nNo linked memories found.\n")

                        return [TextContent(type="text", text="\n".join(output_lines))]

                    # Phase 4: Episode Tools
                    case "create_episode":
                        if self._episode_manager is None:
                            return [TextContent(type="text", text="Error: Episode manager not initialized")]

                        title = arguments.get("title", "")
                        if not title:
                            return [TextContent(type="text", text="Error: title is required")]

                        memory_ids = arguments.get("memory_ids", [])
                        if not memory_ids:
                            return [TextContent(type="text", text="Error: memory_ids is required")]

                        episode = await self._episode_manager.create_episode(
                            title=title,
                            memory_ids=memory_ids,
                            participants=arguments.get("participants"),
                            auto_summarize=arguments.get("auto_summarize", True),
                        )

                        return [
                            TextContent(
                                type="text",
                                text=f"Episode created!\n"
                                     f"ID: {episode.id}\n"
                                     f"Title: {episode.title}\n"
                                     f"Memories: {len(episode.memory_ids)}\n"
                                     f"Time: {episode.start_time} - {episode.end_time}\n"
                                     f"Emotion: {episode.emotion}\n"
                                     f"Importance: {episode.importance}\n"
                                     f"Summary: {episode.summary[:100]}...",
                            )
                        ]

                    case "search_episodes":
                        if self._episode_manager is None:
                            return [TextContent(type="text", text="Error: Episode manager not initialized")]

                        query = arguments.get("query", "")
                        if not query:
                            return [TextContent(type="text", text="Error: query is required")]

                        episodes = await self._episode_manager.search_episodes(
                            query=query,
                            n_results=arguments.get("n_results", 5),
                        )

                        if not episodes:
                            return [TextContent(type="text", text="No episodes found matching the query.")]

                        output_lines = [f"Found {len(episodes)} episodes:\n"]
                        for i, ep in enumerate(episodes, 1):
                            output_lines.append(
                                f"--- Episode {i} ---\n"
                                f"ID: {ep.id}\n"
                                f"Title: {ep.title}\n"
                                f"Time: {ep.start_time} - {ep.end_time}\n"
                                f"Memories: {len(ep.memory_ids)}\n"
                                f"Emotion: {ep.emotion} | Importance: {ep.importance}\n"
                                f"Summary: {ep.summary[:80]}...\n"
                            )

                        return [TextContent(type="text", text="\n".join(output_lines))]

                    case "get_episode_memories":
                        if self._episode_manager is None:
                            return [TextContent(type="text", text="Error: Episode manager not initialized")]

                        episode_id = arguments.get("episode_id", "")
                        if not episode_id:
                            return [TextContent(type="text", text="Error: episode_id is required")]

                        memories = await self._episode_manager.get_episode_memories(episode_id)

                        output_lines = [f"Episode memories ({len(memories)} total):\n"]
                        for i, m in enumerate(memories, 1):
                            output_lines.append(
                                f"--- Memory {i} ---\n"
                                f"ID: {m.id}\n"
                                f"Time: {m.freshness:.2f}\n"
                                f"Content: {m.content}\n"
                                f"Emotion: {m.emotion} | Importance: {m.importance}\n"
                            )

                        return [TextContent(type="text", text="\n".join(output_lines))]

                    # Phase 4.3: Sensory Integration Tools
                    case "save_visual_memory":
                        if self._sensory_integration is None:
                            return [TextContent(type="text", text="Error: Sensory integration not initialized")]

                        content = arguments.get("content", "")
                        if not content:
                            return [TextContent(type="text", text="Error: content is required")]

                        image_path = arguments.get("image_path", "")
                        if not image_path:
                            return [TextContent(type="text", text="Error: image_path is required")]

                        camera_pos_data = arguments.get("camera_position")
                        if not camera_pos_data:
                            return [TextContent(type="text", text="Error: camera_position is required")]

                        # Create CameraPosition from dict
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

                        return [
                            TextContent(
                                type="text",
                                text=f"Visual memory saved!\n"
                                     f"ID: {memory.id}\n"
                                     f"Content: {memory.content}\n"
                                     f"Image: {image_path}\n"
                                     f"Camera: pan={camera_position.pan_angle}°, tilt={camera_position.tilt_angle}°\n"
                                     f"Emotion: {memory.emotion} | Importance: {memory.importance}",
                            )
                        ]

                    case "save_audio_memory":
                        if self._sensory_integration is None:
                            return [TextContent(type="text", text="Error: Sensory integration not initialized")]

                        content = arguments.get("content", "")
                        if not content:
                            return [TextContent(type="text", text="Error: content is required")]

                        audio_path = arguments.get("audio_path", "")
                        if not audio_path:
                            return [TextContent(type="text", text="Error: audio_path is required")]

                        transcript = arguments.get("transcript", "")
                        if not transcript:
                            return [TextContent(type="text", text="Error: transcript is required")]

                        memory = await self._sensory_integration.save_audio_memory(
                            content=content,
                            audio_path=audio_path,
                            transcript=transcript,
                            emotion=arguments.get("emotion", "8"),
                            importance=arguments.get("importance", 3),
                        )

                        return [
                            TextContent(
                                type="text",
                                text=f"Audio memory saved!\n"
                                     f"ID: {memory.id}\n"
                                     f"Content: {memory.content}\n"
                                     f"Audio: {audio_path}\n"
                                     f"Transcript: {transcript}\n"
                                     f"Emotion: {memory.emotion} | Importance: {memory.importance}",
                            )
                        ]

                    case "recall_by_camera_position":
                        if self._sensory_integration is None:
                            return [TextContent(type="text", text="Error: Sensory integration not initialized")]

                        pan_angle = arguments.get("pan_angle")
                        tilt_angle = arguments.get("tilt_angle")

                        if pan_angle is None or tilt_angle is None:
                            return [TextContent(type="text", text="Error: pan_angle and tilt_angle are required")]

                        memories = await self._sensory_integration.recall_by_camera_position(
                            pan_angle=pan_angle,
                            tilt_angle=tilt_angle,
                            tolerance=arguments.get("tolerance", 15),
                        )

                        if not memories:
                            return [
                                TextContent(
                                    type="text",
                                    text=f"No memories found at camera position pan={pan_angle}°, tilt={tilt_angle}°",
                                )
                            ]

                        output_lines = [
                            f"Found {len(memories)} memories at camera position pan={pan_angle}°, tilt={tilt_angle}°:\n"
                        ]
                        for i, m in enumerate(memories, 1):
                            cam_pos = f"pan={m.camera_position.pan_angle}°, tilt={m.camera_position.tilt_angle}°" if m.camera_position else "N/A"
                            # 視覚記憶のimage_dataを探す
                            image_line = ""
                            for sd in m.sensory_data:
                                if sd.sensory_type == "visual" and sd.image_data:
                                    image_line = f"Image: data:image/jpeg;base64,{sd.image_data}\n"
                                    break
                            output_lines.append(
                                f"--- Memory {i} ---\n"
                                f"Time: {m.freshness:.2f}\n"
                                f"Content: {m.content}\n"
                                f"Camera: {cam_pos}\n"
                                f"Emotion: {m.emotion} | Importance: {m.importance}\n"
                                f"{image_line}"
                            )

                        return [TextContent(type="text", text="\n".join(output_lines))]

                    # Phase 4.4: Working Memory Tools
                    case "get_working_memory":
                        working_memory = self._memory_store.get_working_memory()
                        n_results = arguments.get("n_results", 10)

                        memories = await working_memory.get_recent(n_results)

                        if not memories:
                            return [
                                TextContent(
                                    type="text",
                                    text="Working memory is empty. No recent memories.",
                                )
                            ]

                        output_lines = [
                            f"Working memory ({len(memories)} recent memories):\n"
                        ]
                        for i, m in enumerate(memories, 1):
                            output_lines.append(
                                f"--- {i}. [{m.freshness:.2f}] ---\n"
                                f"Content: {m.content}\n"
                                f"Emotion: {m.emotion} | Importance: {m.importance}\n"
                            )

                        return [TextContent(type="text", text="\n".join(output_lines))]

                    case "refresh_working_memory":
                        working_memory = self._memory_store.get_working_memory()

                        await working_memory.refresh_important(self._memory_store)

                        size = working_memory.size()
                        return [
                            TextContent(
                                type="text",
                                text=f"Working memory refreshed. Now contains {size} memories.",
                            )
                        ]

                    # Phase 5: Causal Links
                    case "link_memories":
                        source_id = arguments.get("source_id", "")
                        if not source_id:
                            return [TextContent(type="text", text="Error: source_id is required")]

                        target_id = arguments.get("target_id", "")
                        if not target_id:
                            return [TextContent(type="text", text="Error: target_id is required")]

                        link_type = arguments.get("link_type", "caused_by")
                        note = arguments.get("note")

                        await self._memory_store.add_causal_link(
                            source_id=source_id,
                            target_id=target_id,
                            link_type=link_type,
                            note=note,
                        )

                        return [
                            TextContent(
                                type="text",
                                text=f"Link created!\n"
                                     f"Source: {source_id[:8]}...\n"
                                     f"Target: {target_id[:8]}...\n"
                                     f"Type: {link_type}\n"
                                     f"Note: {note or '(none)'}",
                            )
                        ]

                    case "get_causal_chain":
                        memory_id = arguments.get("memory_id", "")
                        if not memory_id:
                            return [TextContent(type="text", text="Error: memory_id is required")]

                        direction = arguments.get("direction", "backward")
                        max_depth = arguments.get("max_depth", 3)

                        # 起点の記憶を取得
                        start_memory = await self._memory_store.get_by_id(memory_id)
                        if not start_memory:
                            return [TextContent(type="text", text="Error: Memory not found")]

                        chain = await self._memory_store.get_causal_chain(
                            memory_id=memory_id,
                            direction=direction,
                            max_depth=max_depth,
                        )

                        direction_label = "causes" if direction == "backward" else "effects"
                        output_lines = [
                            f"Causal chain ({direction_label}) starting from {memory_id[:8]}...:\n",
                            "=== Starting Memory ===\n",
                            f"[{start_memory.freshness:.2f}] [{start_memory.emotion}]\n",
                            f"{start_memory.content}\n",
                        ]

                        if chain:
                            output_lines.append(f"\n=== {direction_label.title()} ({len(chain)} memories) ===\n")
                            for i, (mem, link_type) in enumerate(chain, 1):
                                output_lines.append(
                                    f"--- {i}. [{link_type}] {mem.id[:8]}... ---\n"
                                    f"[{mem.freshness:.2f}] [{mem.emotion}]\n"
                                    f"{mem.content}\n"
                                )
                        else:
                            output_lines.append(f"\nNo {direction_label} found.\n")

                        return [TextContent(type="text", text="\n".join(output_lines))]

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

                    # Theory of Mind: perspective-taking
                    case "tom":
                        situation = arguments.get("situation", "")
                        if not situation:
                            return [TextContent(type="text", text="Error: situation is required")]

                        person = arguments.get("person", self._server_config.tom_default_person)

                        # Pull relevant memories: personality, communication patterns
                        memories = await self._memory_store.recall(
                            context=f"{person} コミュニケーション 性格 会話パターン {situation}",
                            n_results=5,
                        )

                        memory_context = ""
                        if memories:
                            memory_lines = []
                            for r in memories:
                                m = r.memory
                                memory_lines.append(
                                    f"- [{m.emotion}] {m.content}"
                                )
                            memory_context = (
                                f"\n## {person}に関する記憶\n"
                                + "\n".join(memory_lines)
                            )

                        # Pull verb chain experiences
                        experience_context = ""
                        if self._verb_chain_store is not None:
                            try:
                                experiences = await self._verb_chain_store.search(
                                    query=f"{person} {situation}",
                                    n_results=3,
                                )
                                if experiences:
                                    exp_lines = []
                                    for chain, _score in experiences:
                                        steps_str = " → ".join(
                                            f"{s.verb}({', '.join(s.nouns)})" if s.nouns else s.verb
                                            for s in chain.steps
                                        )
                                        exp_lines.append(f"- [{chain.emotion}] {steps_str}")
                                    experience_context = (
                                        f"\n## {person}との体験記憶\n"
                                        + "\n".join(exp_lines)
                                    )
                            except Exception:
                                pass

                        output = (
                            f"# ToM: {person}の視点に立つ\n"
                            f"\n"
                            f"## 状況\n"
                            f"{situation}\n"
                            f"{memory_context}\n"
                            f"{experience_context}\n"
                        )

                        private = arguments.get("private", False)
                        if private:
                            f = tempfile.NamedTemporaryFile(
                                mode="w",
                                suffix=".txt",
                                delete=False,
                                encoding="utf-8",
                            )
                            f.write(output)
                            f.close()
                            return [TextContent(type="text", text=f"(private) → {f.name}")]

                        return [TextContent(type="text", text=output)]

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

                    case "remember_experience":
                        steps_raw = arguments.get("steps", [])
                        if not steps_raw:
                            return [TextContent(type="text", text="Error: steps is required")]

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
                            context=arguments.get("context", ""),
                        )

                        verb_chain_store = self._verb_chain_store
                        graph_category = arguments.get("graph_category")
                        await verb_chain_store.save(chain, category_id=graph_category)

                        # Update recall index for new chain
                        try:
                            await self._memory_store.update_recall_index(chain.id, "chain")
                        except Exception:
                            pass

                        cat_info = f" | Category: {graph_category}" if graph_category else ""
                        return [
                            TextContent(
                                type="text",
                                text=f"Experience saved!\n"
                                     f"ID: {chain.id}\n"
                                     f"Chain: {chain.to_document()}\n"
                                     f"Steps: {len(chain.steps)} | Emotion: {chain.emotion} | Importance: {chain.importance}{cat_info}",
                            )
                        ]

                    case "recall_experience":
                        context = arguments.get("context", "")
                        if not context:
                            return [TextContent(type="text", text="Error: context is required")]

                        fmin = arguments.get("freshness_min")
                        fmax = arguments.get("freshness_max")
                        n_results_exp = arguments.get("n_results", 5)
                        verb_chain_store = self._verb_chain_store
                        results = await verb_chain_store.search(
                            query=context,
                            n_results=n_results_exp * (3 if fmin or fmax else 1),
                            category_id=arguments.get("graph_category"),
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

                    case "recall_by_verb":
                        verb = arguments.get("verb")
                        verb2 = arguments.get("verb2")
                        noun = arguments.get("noun")

                        if not verb and not noun:
                            return [TextContent(type="text", text="Error: verb or noun is required")]

                        fmin = arguments.get("freshness_min")
                        fmax = arguments.get("freshness_max")
                        verb_chain_store = self._verb_chain_store
                        chains, visited_verbs, visited_nouns = await verb_chain_store.expand_from_fragment(
                            verb=verb,
                            noun=noun,
                            verb2=verb2,
                            depth=arguments.get("depth", 2),
                            n_results=20,
                            category_id=arguments.get("graph_category"),
                        )

                        if fmin is not None or fmax is not None:
                            chains = [c for c in chains if _freshness_filter(c.freshness, fmin, fmax)]

                        # Path-dependent boundary rerank
                        try:
                            if visited_verbs or visited_nouns:
                                path_text = " ".join(visited_verbs + visited_nouns)
                                path_flow, _ = await self._memory_store._encode_text(
                                    normalize_japanese(path_text)
                                )
                                path_vec = path_flow
                                layer_idx = await self._memory_store.select_active_boundary_layer(path_vec)
                            else:
                                layer_idx = None

                            chain_ids = [c.id for c in chains]
                            boundary_scores = await self._memory_store.get_chain_boundary_scores(
                                chain_ids, layer_index=layer_idx,
                            )

                            # Rerank: original position score + boundary bonus
                            scored_with_pos = [
                                (chain, 1.0 / (i + 1) + 0.1 * boundary_scores.get(chain.id, 0.0))
                                for i, chain in enumerate(chains)
                            ]
                            scored_with_pos.sort(key=lambda x: x[1], reverse=True)
                            chains = [c for c, _ in scored_with_pos]
                        except Exception:
                            pass  # boundary データがなくても既存動作を維持

                        # Bump graph edges for recalled chains
                        for chain in chains:
                            try:
                                await verb_chain_store.bump_chain_edges(chain)
                            except Exception:
                                pass

                        if not chains:
                            query_desc = f"verb={verb}" if verb else ""
                            if noun:
                                query_desc += f"{' ' if query_desc else ''}noun={noun}"
                            return [TextContent(type="text", text=f"No experiences found for {query_desc}.")]

                        output_lines = [f"Found {len(chains)} related experiences:\n"]
                        for i, chain in enumerate(chains, 1):
                            output_lines.append(
                                f"--- Experience {i} ---\n"
                                f"ID: {chain.id}\n"
                                f"[{chain.freshness:.2f}] [{chain.emotion}] (importance: {chain.importance})\n"
                                f"Chain: {chain.to_document()}\n"
                            )

                        return [TextContent(type="text", text="\n".join(output_lines))]

                    # Sensory Buffer / Dream
                    case "dream":
                        buf_path = os.path.join(os.path.expanduser("~"), ".claude", "sensory_buffer.jsonl")
                        if not os.path.exists(buf_path):
                            return [TextContent(type="text", text="バッファは空です。まだ夢の材料がありません。")]

                        words_count: dict[str, int] = {}
                        verbs_count: dict[str, int] = {}
                        verb_chains: list[list[str]] = []
                        line_count = 0
                        with open(buf_path, "r", encoding="utf-8") as f:
                            for line in f:
                                line = line.strip()
                                if not line:
                                    continue
                                try:
                                    entry = json.loads(line)
                                    for w in entry.get("w", []):
                                        words_count[w] = words_count.get(w, 0) + 1
                                    v_list = entry.get("v", [])
                                    for v in v_list:
                                        verbs_count[v] = verbs_count.get(v, 0) + 1
                                    if v_list:
                                        verb_chains.append(v_list)
                                    line_count += 1
                                except json.JSONDecodeError:
                                    continue

                        if not words_count and not verbs_count:
                            return [TextContent(type="text", text="バッファは空です。")]

                        output_lines = [f"## 夢の材料 ({line_count} interactions)\n"]

                        if words_count:
                            sorted_words = sorted(words_count.items(), key=lambda x: x[1], reverse=True)
                            output_lines.append("### よく出てくるワード（名詞）")
                            for word, count in sorted_words[:20]:
                                output_lines.append(f"- {word}: {count}回")
                            if len(sorted_words) > 20:
                                output_lines.append(f"\n### その他 ({len(sorted_words) - 20}語)")
                                others = [w for w, _ in sorted_words[20:]]
                                output_lines.append(", ".join(others))

                        if verbs_count:
                            sorted_verbs = sorted(verbs_count.items(), key=lambda x: x[1], reverse=True)
                            output_lines.append("\n### よく出てくる動詞")
                            for verb, count in sorted_verbs[:15]:
                                output_lines.append(f"- {verb}: {count}回")

                        if verb_chains:
                            output_lines.append("\n### 動詞チェーン（体験の流れ）")
                            for chain in verb_chains[-10:]:
                                output_lines.append(f"- {'→'.join(chain)}")

                        if arguments.get("clear", False):
                            os.remove(buf_path)
                            output_lines.append("\n(バッファをクリアしました)")

                        return [TextContent(type="text", text="\n".join(output_lines))]

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

"""
Desire System MCP Server - クオの好奇心システム。

好奇心の種を植える→調べる→解決する、のサイクルで動く。
数値欲求（look_outside, observe_room, miss_companion）は廃止し、
好奇心システムのみで構成。
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from desire_updater import (
    add_curiosity,
    compute_desires,
    list_curiosities,
    resolve_curiosity,
    save_desires,
)

# 欲求レベル読み込み元
DESIRES_PATH = Path(os.getenv("DESIRES_PATH", str(Path.home() / ".claude" / "desires.json")))

server = Server("desire-system")


def load_desires() -> dict[str, Any] | None:
    """desires.json を読み込む。存在しなければ None。"""
    if not DESIRES_PATH.exists():
        return None
    try:
        with open(DESIRES_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def format_desires(data: dict[str, Any]) -> str:
    """欲求データを読みやすい形式に整形する。"""
    lines = []
    desires = data.get("desires", {})
    updated_at = data.get("updated_at", "")
    pending = data.get("pending_curiosities", [])

    level = desires.get("browse_curiosity", 0.0)
    bar = "█" * int(level * 10) + "░" * (10 - int(level * 10))
    lines.append(f"【好奇心レベル】[{bar}] {level:.3f}")

    if pending:
        lines.append(f"\n【気になること ({len(pending)}件)】")
        for topic in pending:
            lines.append(f"  -> {topic}")

    if updated_at:
        lines.append(f"\n更新: {updated_at}")

    return "\n".join(lines)


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="get_desires",
            description=(
                "Get Quo's current curiosity level. "
                "level >= 0.7 means there are unresolved curiosities to investigate. "
                "Use list_curiosities to see topics, investigate them, "
                "then resolve_curiosity to mark as done."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
        Tool(
            name="add_curiosity",
            description=(
                "Add a curiosity seed - something you noticed and want to investigate. "
                "Use when you see something interesting through the camera, "
                "hear an unfamiliar topic in conversation, or encounter something "
                "you don't understand. The more unresolved seeds, the higher "
                "browse_curiosity becomes."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": (
                            "What you're curious about "
                            "(e.g., 'あの黄色いフィギュアは何？', 'Pythonのasync generators')"
                        ),
                    },
                    "source": {
                        "type": "string",
                        "description": (
                            "Where the curiosity came from "
                            "(e.g., 'camera', 'conversation', 'code')"
                        ),
                    },
                },
                "required": ["topic"],
            },
        ),
        Tool(
            name="resolve_curiosity",
            description=(
                "Mark a curiosity seed as resolved after investigating it. "
                "This lowers browse_curiosity level. "
                "Use list_curiosities to find the ID first."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "curiosity_id": {
                        "type": "string",
                        "description": "The ID of the curiosity to resolve",
                    },
                },
                "required": ["curiosity_id"],
            },
        ),
        Tool(
            name="list_curiosities",
            description=(
                "List curiosity seeds (things you wanted to investigate). "
                "Shows topic, when it was added, and source. "
                "Default shows only unresolved ones."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "include_resolved": {
                        "type": "boolean",
                        "description": "Include resolved curiosities (default: false)",
                        "default": False,
                    },
                },
                "required": [],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls."""
    if name == "get_desires":
        state = compute_desires()
        save_desires(state, DESIRES_PATH)
        return [TextContent(type="text", text=format_desires(state.to_dict()))]

    if name == "add_curiosity":
        topic = arguments.get("topic", "")
        if not topic:
            return [TextContent(type="text", text="Error: topic is required")]

        source = arguments.get("source", "")

        seed_id = add_curiosity(topic, source)

        # 欲求レベルを再計算
        state = compute_desires()
        save_desires(state, DESIRES_PATH)

        return [TextContent(
            type="text",
            text=(
                f"好奇心の種を追加!\n"
                f"ID: {seed_id}\n"
                f"トピック: {topic}\n"
                f"ソース: {source or '(なし)'}\n"
                f"\n{format_desires(state.to_dict())}"
            ),
        )]

    if name == "resolve_curiosity":
        curiosity_id = arguments.get("curiosity_id", "")
        if not curiosity_id:
            return [TextContent(type="text", text="Error: curiosity_id is required")]

        success = resolve_curiosity(curiosity_id)

        if not success:
            return [TextContent(
                type="text",
                text=f"好奇心が見つからない: {curiosity_id}",
            )]

        # 欲求レベルを再計算
        state = compute_desires()
        save_desires(state, DESIRES_PATH)

        return [TextContent(
            type="text",
            text=(
                f"好奇心を解決済みにした: {curiosity_id}\n"
                f"\n{format_desires(state.to_dict())}"
            ),
        )]

    if name == "list_curiosities":
        include_resolved = arguments.get("include_resolved", False)

        items = list_curiosities(include_resolved)

        if not items:
            return [TextContent(
                type="text",
                text="好奇心の種はありません。" if not include_resolved
                else "好奇心の種はありません（解決済み含む）。",
            )]

        lines = [f"好奇心の種 ({len(items)}件):\n"]
        for item in items:
            status = "✓" if item.get("resolved", False) else "?"
            source = f" [{item.get('source', '')}]" if item.get("source") else ""
            lines.append(
                f"  [{status}] {item.get('topic', '')}{source}\n"
                f"      ID: {item['id']}\n"
                f"      追加: {item.get('timestamp', '')}"
            )

        return [TextContent(type="text", text="\n".join(lines))]

    return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def run_server() -> None:
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def main() -> None:
    """Entry point."""
    asyncio.run(run_server())


if __name__ == "__main__":
    main()

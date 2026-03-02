"""Configuration for Memory MCP Server."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class MemoryConfig:
    """Memory storage configuration."""

    db_path: str
    collection_name: str
    chive_model_path: str
    enable_bm25: bool = True
    memory_md_path: str = ""

    @classmethod
    def from_env(cls) -> "MemoryConfig":
        """Create config from environment variables."""
        default_path = str(Path.home() / ".claude" / "memories" / "memory.db")

        chive_path = os.getenv("CHIVE_MODEL_PATH", "")
        if not chive_path:
            raise ValueError(
                "CHIVE_MODEL_PATH environment variable is required. "
                "Set it to the path of your chiVe Word2Vec model file."
            )

        return cls(
            db_path=os.getenv("MEMORY_DB_PATH", default_path),
            collection_name=os.getenv("MEMORY_COLLECTION_NAME", "claude_memories"),
            chive_model_path=chive_path,
            enable_bm25=os.getenv("MEMORY_ENABLE_BM25", "true").lower() != "false",
            memory_md_path=os.getenv("MEMORY_MD_PATH", ""),
        )


@dataclass(frozen=True)
class ServerConfig:
    """MCP Server configuration."""

    name: str = "memory-mcp"
    version: str = "0.1.0"
    tom_default_person: str = "コウタ"

    @classmethod
    def from_env(cls) -> "ServerConfig":
        """Create config from environment variables."""
        return cls(
            name=os.getenv("MCP_SERVER_NAME", "memory-mcp"),
            version=os.getenv("MCP_SERVER_VERSION", "0.1.0"),
            tom_default_person=os.getenv("TOM_DEFAULT_PERSON", "コウタ"),
        )

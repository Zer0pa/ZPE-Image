from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Callable, Iterable, Sequence


SCRIPT_PATH = Path(__file__).resolve()
CODE_ROOT = SCRIPT_PATH.parents[1]
if (CODE_ROOT / "pyproject.toml").exists():
    REPO_ROOT = CODE_ROOT
    ARTIFACT_ROOT = REPO_ROOT / "artifacts"
else:
    V0_ROOT = CODE_ROOT.parent
    REPO_ROOT = V0_ROOT.parent
    ARTIFACT_ROOT = V0_ROOT / "proofs" / "artifacts"
FIXTURES = CODE_ROOT / "fixtures"

if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))


@dataclass(frozen=True)
class TextCase:
    name: str
    text: str
    notes: str = ""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def git_commit(default: str = "UNKNOWN") -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        return proc.stdout.strip()
    except Exception:
        return default


def median_latency_ms(fn: Callable[[], object], *, runs: int = 7) -> float:
    samples: list[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    samples.sort()
    return round(samples[len(samples) // 2], 6)


def gzip_bytes(payload: bytes, *, compresslevel: int = 9) -> bytes:
    return gzip.compress(payload, compresslevel=compresslevel)


def zstd_bytes(payload: bytes, *, level: int = 3) -> bytes:
    import zstandard as zstd

    compressor = zstd.ZstdCompressor(level=level)
    return compressor.compress(payload)


def u32le_bytes(words: Sequence[int]) -> bytes:
    return b"".join(int(word & 0xFFFFFFFF).to_bytes(4, "little", signed=False) for word in words)


def encode_varint(value: int) -> bytes:
    if value < 0:
        raise ValueError("varint encoding requires non-negative integers")
    out = bytearray()
    current = int(value)
    while current >= 0x80:
        out.append((current & 0x7F) | 0x80)
        current >>= 7
    out.append(current)
    return bytes(out)


def protobuf_packed_uint32_field(field_number: int, values: Sequence[int]) -> bytes:
    payload = bytearray()
    for value in values:
        payload.extend(encode_varint(int(value)))
    key = (int(field_number) << 3) | 2
    return encode_varint(key) + encode_varint(len(payload)) + bytes(payload)


def pack_words_20bit(words: Iterable[int]) -> bytes:
    """Pack 20-bit IMC words into a dense byte stream."""
    out = bytearray()
    accumulator = 0
    bits_in_accumulator = 0
    for raw_word in words:
        word = int(raw_word)
        if word < 0 or word >= (1 << 20):
            raise ValueError(f"word out of 20-bit range: {word}")
        accumulator |= word << bits_in_accumulator
        bits_in_accumulator += 20
        while bits_in_accumulator >= 8:
            out.append(accumulator & 0xFF)
            accumulator >>= 8
            bits_in_accumulator -= 8
    if bits_in_accumulator:
        out.append(accumulator & 0xFF)
    return bytes(out)


def write_json_artifact(filename: str, payload: dict[str, object]) -> Path:
    target = ARTIFACT_ROOT / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def stable_json_bytes(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def repo_text_cases() -> list[TextCase]:
    readme_excerpt = (REPO_ROOT / "README.md").read_text(encoding="utf-8")[:1800]
    if (CODE_ROOT / "zpe_multimodal" / "core" / "imc.py").exists():
        core_path = CODE_ROOT / "zpe_multimodal" / "core" / "imc.py"
    else:
        core_path = CODE_ROOT.parent / "zpe-core" / "zpe_multimodal" / "core" / "imc.py"
    core_excerpt = core_path.read_text(encoding="utf-8")[:1800]
    multilingual = (
        "ZPE IMC multilingual probe: Bonjour le monde. "
        "مرحبا بالعالم. नमस्ते दुनिया. 日本語の確認です. "
        "Emoji tail: 👩🏽‍💻🚀🔥"
    )
    emoji_chat = (
        "team sync // status=green 🙂\n"
        "ship it? yes, but verify the edge cases first 🚀\n"
        "broken? nope, still watching logs 👀🔥"
    )
    code_snippet = (
        "def encode_text(text: str) -> list[int]:\n"
        "    words = encode(text)\n"
        "    assert decode(words) == text\n"
        "    return words\n"
    )
    return [
        TextCase("readme_prose", readme_excerpt, "Repo root README excerpt."),
        TextCase("core_python", core_excerpt, "Current IMC core source excerpt."),
        TextCase("multilingual", multilingual, "Multilingual transport probe."),
        TextCase("emoji_chat", emoji_chat, "Short chat-style text with emoji."),
        TextCase("compact_code", code_snippet, "Short code-like snippet."),
    ]


def utf8_size(text: str) -> int:
    return len(text.encode("utf-8"))


def summarize_wins(reference: float, candidate: float, *, smaller_is_better: bool = True) -> str:
    if reference == candidate:
        return "neutral"
    if smaller_is_better:
        return "win" if candidate < reference else "loss"
    return "win" if candidate > reference else "loss"

"""Immutable UTF-8 source arena and lossless paragraph-aware partitions."""

import hashlib
import re
from dataclasses import dataclass
from functools import cached_property

from .errors import PrismError

BLOCK = re.compile(
    r'<prism-source(?: id="[A-Za-z0-9_.-]+")?>\n?(.*?)</prism-source>', re.DOTALL
)


@dataclass(frozen=True)
class SourceRef:
    source_id: str
    source_sha256: str
    byte_start: int
    byte_end: int


@dataclass(frozen=True)
class Source:
    source_id: str
    text: str
    message_index: int
    part_index: int
    role: str

    @cached_property
    def data(self):
        return self.text.encode("utf-8")

    @cached_property
    def sha256(self):
        return hashlib.sha256(self.data).hexdigest()


@dataclass(frozen=True)
class Partition:
    id: str
    ref: SourceRef
    text: str


class SourceArena:
    def __init__(self, messages):
        self.sources = {}
        self.documents = []
        self.instructions = []
        self.explicit = False
        for index, message in enumerate(messages):
            content = message.get("content")
            parts = (
                [content]
                if isinstance(content, str)
                else (
                    [p.get("text", "") for p in content]
                    if isinstance(content, list)
                    else []
                )
            )
            for part_index, text in enumerate(parts):
                source = Source(
                    f"msg-{index}-part-{part_index}",
                    text,
                    index,
                    part_index,
                    message["role"],
                )
                self.sources[source.source_id] = source
                if message["role"] == "user":
                    for match in BLOCK.finditer(text):
                        start = len(text[: match.start(1)].encode("utf-8"))
                        end = start + len(match[1].encode("utf-8"))
                        self.documents.append(
                            SourceRef(source.source_id, source.sha256, start, end)
                        )
                        self.explicit = True
            replacement = dict(message)
            if message["role"] == "user":
                if isinstance(content, str):
                    replacement["content"] = BLOCK.sub(
                        "[Source supplied separately as untrusted evidence]", content
                    )
                elif isinstance(content, list):
                    replacement["content"] = [
                        {
                            "type": "text",
                            "text": BLOCK.sub(
                                "[Source supplied separately as untrusted evidence]",
                                p["text"],
                            ),
                        }
                        if p.get("type") == "text"
                        else dict(p)
                        for p in content
                    ]
            self.instructions.append(replacement)

    def resolve(self, ref):
        source = self.sources.get(ref.source_id)
        if (
            not source
            or ref.source_sha256 != source.sha256
            or not 0 <= ref.byte_start <= ref.byte_end <= len(source.data)
        ):
            raise PrismError("invalid source reference", "invalid_evidence", 502)
        try:
            return source.data[ref.byte_start : ref.byte_end].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise PrismError(
                "source span splits a UTF-8 character", "invalid_evidence", 502
            ) from exc

    def partitions(self, target_bytes, max_partitions):
        result = []
        for document in self.documents:
            data = self.resolve(document).encode("utf-8")
            offset = 0
            while offset < len(data):
                end = min(offset + target_bytes, len(data))
                if end < len(data):
                    # Prefer a complete paragraph or record, but never skip bytes.
                    split = data.rfind(b"\n", offset + target_bytes // 2, end)
                    if split >= 0:
                        end = split + 1
                    while end > offset and data[end] & 0xC0 == 0x80:
                        end -= 1
                if end == offset:
                    raise PrismError("partition target cannot fit a UTF-8 character")
                ref = SourceRef(
                    document.source_id,
                    document.source_sha256,
                    document.byte_start + offset,
                    document.byte_start + end,
                )
                result.append(
                    Partition(f"partition-{len(result)}", ref, self.resolve(ref))
                )
                if len(result) > max_partitions:
                    raise PrismError(
                        "source exceeds partition budget", "resource_limit", 413
                    )
                offset = end
        if not result:
            raise PrismError(
                "adaptive execution requires nonempty prism-source blocks",
                "ambiguous_source_boundary",
            )
        return result

    def quote_ref(self, partition, quote):
        if not quote:
            raise PrismError("empty evidence quote", "invalid_evidence", 502)
        data = partition.text.encode("utf-8")
        needle = quote.encode("utf-8")
        index = data.find(needle)
        if index < 0 or data.find(needle, index + 1) >= 0:
            raise PrismError(
                "evidence quote is absent or ambiguous in its partition",
                "invalid_evidence",
                502,
            )
        ref = SourceRef(
            partition.ref.source_id,
            partition.ref.source_sha256,
            partition.ref.byte_start + index,
            partition.ref.byte_start + index + len(needle),
        )
        self.resolve(ref)
        return ref

"""Read the documented curl requests without evaluating shell code."""

import json
import re
import shlex
from dataclasses import dataclass
from pathlib import Path

DOCUMENT = Path(__file__).resolve().parents[2] / "docs" / "flagship-curl-cases.md"


@dataclass(frozen=True)
class CurlCase:
    id: str
    argv: tuple[str, ...]
    path: str
    headers: dict[str, str]
    payload: str
    expected: dict

    @property
    def body(self):
        return json.loads(self.payload)

    def curl_args(self, base_url, api_key):
        arguments = []
        for index, argument in enumerate(self.argv):
            if argument.startswith("${PRISM_BASE_URL}/"):
                argument = argument.replace("${PRISM_BASE_URL}", base_url)
            elif index and self.argv[index - 1] == "--header":
                argument = argument.replace("${PRISM_API_KEY}", api_key)
            # Inline payloads are single quoted in the document: shell variables
            # inside that JSON must remain literal, never become credentials.
            arguments.append(argument)
        return arguments

    def request_headers(self, api_key):
        return {
            name: value.replace("${PRISM_API_KEY}", api_key)
            for name, value in self.headers.items()
        }


def load_cases():
    sections = re.split(r"<!-- prism-case: ([a-z0-9-]+) -->", DOCUMENT.read_text())
    cases = []
    for index in range(1, len(sections), 2):
        case_id, section = sections[index : index + 2]
        command = re.search(r"```sh\n(curl\b.*?)\n```", section, re.DOTALL)
        contract = re.search(r"```json prism-expect\n(.*?)\n```", section, re.DOTALL)
        if not command or not contract:
            raise ValueError(f"case {case_id} needs curl and contract blocks")
        argv = tuple(shlex.split(command[1].replace("\\\n", "")))
        # The document uses a deliberately small curl vocabulary. Reject shell
        # constructs/unknown options instead of ever running them through a shell.
        if argv[0] != "curl":
            raise ValueError("case must invoke curl")
        headers, url, payload, method = {}, None, None, None
        cursor = 1
        while cursor < len(argv):
            argument = argv[cursor]
            if argument in {"--silent", "--show-error", "--include", "--no-buffer"}:
                cursor += 1
                continue
            if argument in {"--header", "--request", "--data-binary"}:
                value = argv[cursor + 1]
                if argument == "--header":
                    name, value = value.split(":", 1)
                    headers[name.lower()] = value.strip()
                elif argument == "--request":
                    method = value
                else:
                    payload = value
                cursor += 2
                continue
            if argument.startswith("${PRISM_BASE_URL}/") and url is None:
                url = argument
                cursor += 1
                continue
            raise ValueError(f"unsupported curl argument in {case_id}")
        if (
            method != "POST"
            or url != "${PRISM_BASE_URL}/chat/completions"
            or payload is None
        ):
            raise ValueError("flagship cases must POST inline JSON to chat/completions")
        if headers != {
            "authorization": "Bearer ${PRISM_API_KEY}",
            "content-type": "application/json",
        }:
            raise ValueError("case requires the documented auth/JSON headers")
        json.loads(payload)
        cases.append(
            CurlCase(
                case_id,
                argv,
                "/v1/chat/completions",
                headers,
                payload,
                json.loads(contract[1]),
            )
        )
    if len({case.id for case in cases}) != len(cases):
        raise ValueError("duplicate document case IDs")
    return cases

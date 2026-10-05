"""Fail before building if versions disagree or this release already exists on PyPI."""

import ast
import sys
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


def release_version(root):
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    assignments = ast.parse((root / "src/prism/__init__.py").read_text()).body
    runtime = next(
        ast.literal_eval(node.value)
        for node in assignments
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "__version__" for t in node.targets)
    )
    if runtime != project["version"]:
        raise ValueError(
            "pyproject.toml and prism.__version__ must use the same release version"
        )
    return project["name"], project["version"]


def require_unpublished(name, version):
    url = "https://pypi.org/pypi/{}/{}/json".format(
        urllib.parse.quote(name, safe=""), urllib.parse.quote(version, safe="")
    )
    request = urllib.request.Request(url, headers={"User-Agent": "Prism-release-check"})
    try:
        with urllib.request.urlopen(request, timeout=20):
            pass
    except urllib.error.HTTPError as error:
        error.close()
        if error.code == 404:
            return
        raise ValueError(
            f"Cannot check PyPI (HTTP {error.code}); retry after resolving the service error"
        ) from error
    except (urllib.error.URLError, TimeoutError) as error:
        raise ValueError(
            "Cannot reach PyPI to verify the release version; retry when connectivity is restored"
        ) from error
    raise ValueError(
        f"{name} {version} already exists on PyPI. Bump pyproject.toml and "
        "prism.__version__, commit and push, then start a new workflow run. "
        "Re-running an old job reuses its old version."
    )


def main():
    try:
        name, version = release_version(Path(__file__).resolve().parents[1])
        require_unpublished(name, version)
    except (ValueError, KeyError, StopIteration, OSError) as error:
        print(f"Release check failed: {error}", file=sys.stderr)
        return 2
    print(f"{name} {version} is available for a new PyPI release")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from contextlib import nullcontext
from urllib.error import HTTPError, URLError

import pytest

from scripts import check_release


def test_release_versions_must_match_without_importing_package(tmp_path):
    source = tmp_path / "src/prism"
    source.mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "prism"\nversion = "0.4.0"\n'
    )
    # Checking version metadata must work before runtime dependencies are installed.
    (source / "__init__.py").write_text(
        'import nonexistent_dependency\n__version__ = "0.4.0"\n'
    )
    assert check_release.release_version(tmp_path) == ("prism", "0.4.0")
    (source / "__init__.py").write_text('__version__ = "0.3.1"\n')
    with pytest.raises(ValueError, match="same release version"):
        check_release.release_version(tmp_path)


def test_existing_release_stops_with_actionable_error(monkeypatch, capsys):
    monkeypatch.setattr(
        check_release.urllib.request, "urlopen", lambda *args, **kwargs: nullcontext()
    )
    assert check_release.main() == 2
    message = capsys.readouterr().err
    assert "already exists on PyPI" in message
    assert "new workflow run" in message


def test_only_404_allows_a_new_release(monkeypatch, capsys):
    def unpublished(request, timeout):
        assert request.full_url == "https://pypi.org/pypi/mirrorneuron-prism/0.4.0/json"
        assert timeout == 20
        raise HTTPError(request.full_url, 404, "Not Found", {}, None)

    monkeypatch.setattr(check_release.urllib.request, "urlopen", unpublished)
    assert check_release.main() == 0
    assert "available for a new PyPI release" in capsys.readouterr().out


@pytest.mark.parametrize(
    "error",
    [
        HTTPError("https://pypi.org", 503, "Unavailable", {}, None),
        URLError("offline"),
        TimeoutError(),
    ],
)
def test_service_errors_do_not_mistake_a_release_for_unpublished(error, monkeypatch):
    def failing(*args, **kwargs):
        raise error

    monkeypatch.setattr(check_release.urllib.request, "urlopen", failing)
    with pytest.raises(ValueError, match="Cannot"):
        check_release.require_unpublished("mirrorneuron-prism", "0.4.0")

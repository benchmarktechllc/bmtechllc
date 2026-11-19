from __future__ import annotations

from pathlib import Path

import pytest

from article_aggregator.config import ConfigError, load_config

REPO_CONFIG = Path(__file__).parent.parent / "config"


def write(tmp_path: Path, feeds: str, keywords: str) -> Path:
    (tmp_path / "feeds.yaml").write_text(feeds, encoding="utf-8")
    (tmp_path / "keywords.yaml").write_text(keywords, encoding="utf-8")
    return tmp_path


def test_loads_valid_config(config_dir: Path) -> None:
    config = load_config(config_dir)
    assert [f.name for f in config.enabled_feeds] == ["RSS Fixture", "Atom Fixture"]
    assert len(config.feeds) == 3
    assert config.keywords[1] == "zero day"


def test_repository_config_is_valid() -> None:
    config = load_config(REPO_CONFIG)
    assert config.enabled_feeds and config.keywords
    assert config.feed_caps == {"LangChain Blog": 3, "arXiv cs.AI": 2, "TechCrunch AI": 2}


def test_feed_caps_only_include_capped_feeds(tmp_path: Path) -> None:
    config = load_config(write(
        tmp_path,
        "- {name: A, url: 'https://a.example', max_per_digest: 2}\n"
        "- {name: B, url: 'https://b.example'}\n",
        "- x\n",
    ))
    assert config.feed_caps == {"A": 2}


def test_keywords_are_whitespace_normalized(tmp_path: Path) -> None:
    config = load_config(write(tmp_path, "- {name: A, url: 'https://a.example'}\n",
                               "- '  zero   day '\n"))
    assert config.keywords == ("zero day",)


@pytest.mark.parametrize(
    ("feeds", "keywords", "message"),
    [
        ("- {name: A, url: 'ftp://a.example'}\n", "- x\n", "http(s) URL"),
        ("- {name: A, url: 'https://a.example', colour: red}\n", "- x\n", "colour"),
        ("- {name: '  ', url: 'https://a.example'}\n", "- x\n", "blank"),
        ("- {name: A, url: 'https://a.example'}\n- {name: B, url: 'https://a.example'}\n",
         "- x\n", "duplicate feed url"),
        ("- {name: A, url: 'https://a.example'}\n- {name: A, url: 'https://b.example'}\n",
         "- x\n", "duplicate feed name"),
        ("- {name: A, url: 'https://a.example', max_per_digest: 0}\n", "- x\n",
         "greater than or equal to 1"),
        ("[]\n", "- x\n", "at least one feed"),
        ("- {name: A, url: 'https://a.example', enabled: false}\n", "- x\n", "every feed"),
        ("- {name: A, url: 'https://a.example'}\n", "[]\n", "at least one keyword"),
        ("- {name: A, url: 'https://a.example'}\n", "- AI\n- ai\n", "duplicate keyword"),
        ("- {name: A, url: 'https://a.example'}\n", "- ' '\n", "blank keyword"),
        ("feeds: [unclosed\n", "- x\n", "invalid YAML"),
    ],
)
def test_invalid_config_has_clear_message(
    tmp_path: Path, feeds: str, keywords: str, message: str
) -> None:
    with pytest.raises(ConfigError, match=message.replace("(", r"\(").replace(")", r"\)")):
        load_config(write(tmp_path, feeds, keywords))


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="feeds.yaml: file not found"):
        load_config(tmp_path)

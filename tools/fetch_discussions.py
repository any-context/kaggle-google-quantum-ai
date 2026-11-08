#!/usr/bin/env python3
"""Fetch the competition discussion forum through the Kaggle CLI and build a readable digest.

Everything goes through `kaggle competitions topics list` / `kaggle competitions topic-messages`
(Kaggle CLI >= 2.2), so no cookies, scraping or private endpoints are involved.

Outputs (under build/discussions/ by default):
    topics.json              all topics (id, title, votes, commentCount, postDate)
    topics/<id>.json         raw messages of one topic (as returned by the CLI)
    digest.md                human-readable digest: newest first, HTML stripped, long posts trimmed
    state.json               last fetch time + per-topic comment counts (for incremental refresh)

Usage:
    tools/fetch_discussions.py                      # incremental: new topics + topics whose comment count changed
    tools/fetch_discussions.py --full               # re-download every topic
    tools/fetch_discussions.py --grep salt --grep private   # print matching posts (case-insensitive) and exit
    tools/fetch_discussions.py --show 756824        # print one topic in full
    tools/fetch_discussions.py --max-chars 4000     # longer excerpts in the digest
"""
from __future__ import annotations

import argparse
import html
import json
import re
import subprocess
import sys
import time
from pathlib import Path

COMPETITION = "google-verifiable-quantum-circuit-compilation"
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "build" / "discussions"


def kaggle_json(args: list[str], timeout: int = 120) -> list | dict:
    """Run a Kaggle CLI command with --format json and parse the JSON payload.

    The CLI appends a 'Next Page Token = ...' line after the JSON on paged commands and prints
    warnings on stderr, so only the leading JSON document is parsed.
    """
    cmd = ["kaggle"] + args + ["--format", "json"]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    text = proc.stdout.strip()
    if not text:
        if proc.returncode != 0:
            raise RuntimeError(f"{' '.join(cmd)}: exit {proc.returncode}\n{proc.stderr[-800:]}")
        return []
    if text.startswith("No ") and "found" in text.splitlines()[0]:  # e.g. "No topics found"
        return []
    dec = json.JSONDecoder()
    start = text.find("[") if text.lstrip().startswith("[") else text.find("{")
    if start < 0:
        raise RuntimeError(f"{' '.join(cmd)}: no JSON in output:\n{text[:400]}")
    obj, _ = dec.raw_decode(text[start:])
    return obj


def list_topics(sort_by: str = "new", max_pages: int = 50) -> list[dict]:
    topics: dict[int, dict] = {}
    for page in range(1, max_pages + 1):
        rows = kaggle_json(["competitions", "topics", "list", COMPETITION, "--sort-by", sort_by, "-p", str(page)])
        if not rows:
            break
        new = 0
        for r in rows:
            if r["id"] not in topics:
                topics[r["id"]] = r
                new += 1
        if new == 0:  # the CLI repeats the last page when the index runs out
            break
    return sorted(topics.values(), key=lambda r: r.get("postDate", ""), reverse=True)


def fetch_topic(topic_id: int) -> list | dict:
    return kaggle_json(["competitions", "topic-messages", COMPETITION, str(topic_id), "-n", "-1", "--sort-by", "old"])


def strip_html(s: str) -> str:
    s = re.sub(r"<(br|/p|/h\d|/li|/tr)\s*/?>", "\n", s, flags=re.I)
    s = re.sub(r"<(td|th)[^>]*>", " | ", s, flags=re.I)
    s = re.sub(r"<[^>]+>", "", s)
    s = html.unescape(s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n\s*\n+", "\n\n", s)
    return s.strip()


def iter_messages(payload, depth: int = 0):
    """Yield (depth, message_dict) for every message / reply in a topic payload."""
    if isinstance(payload, dict):
        if "content" in payload or "rawMarkdown" in payload:
            yield depth, payload
            for key in ("replies", "comments", "children"):
                for child in payload.get(key) or []:
                    yield from iter_messages(child, depth + 1)
            return
        for v in payload.values():
            yield from iter_messages(v, depth)
    elif isinstance(payload, list):
        for v in payload:
            yield from iter_messages(v, depth)


def message_text(m: dict) -> str:
    return strip_html(str(m.get("rawMarkdown") or m.get("content") or ""))


def author_of(m: dict) -> str:
    a = m.get("author")
    if isinstance(a, dict):
        return a.get("displayName") or a.get("userName") or "?"
    return m.get("authorName") or m.get("authorDisplayName") or "?"


def render_topic(topic: dict, payload, max_chars: int) -> str:
    lines = [f"## [{topic['id']}] {topic['title']}",
             f"*posted {topic.get('postDate', '')[:16]} · votes {topic.get('votes', 0)} · comments {topic.get('commentCount', 0)}*",
             f"https://www.kaggle.com/competitions/{COMPETITION}/discussion/{topic['id']}", ""]
    for depth, m in iter_messages(payload):
        txt = message_text(m)
        if len(txt) > max_chars:
            txt = txt[:max_chars].rstrip() + f" … [{len(txt) - max_chars} more chars; --show {topic['id']}]"
        indent = "> " * depth
        head = f"{indent}**{author_of(m)}** ({str(m.get('postDate', ''))[:16]}, votes {m.get('votes', 0)})"
        body = "\n".join(indent + ln for ln in txt.splitlines()) if txt else indent + "(empty)"
        lines += [head, body, ""]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--full", action="store_true", help="re-download every topic (default: incremental)")
    ap.add_argument("--no-fetch", action="store_true", help="only rebuild digest / grep from cached files")
    ap.add_argument("--max-chars", type=int, default=2500, help="excerpt length per post in the digest")
    ap.add_argument("--grep", action="append", default=[], help="case-insensitive pattern; print matching posts")
    ap.add_argument("--show", type=int, action="append", default=[], help="print these topics in full")
    a = ap.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "topics").mkdir(exist_ok=True)
    state_path = a.out / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"comment_counts": {}}

    if not a.no_fetch:
        topics = list_topics()
        (a.out / "topics.json").write_text(json.dumps(topics, indent=1, ensure_ascii=False))
        fetched = 0
        for t in topics:
            tid = str(t["id"])
            cached = (a.out / "topics" / f"{tid}.json").exists()
            changed = state["comment_counts"].get(tid) != t.get("commentCount")
            if a.full or not cached or changed:
                try:
                    payload = fetch_topic(t["id"])
                except Exception as e:  # keep going; one bad topic must not kill the refresh
                    print(f"  ! topic {tid}: {e}", file=sys.stderr)
                    continue
                (a.out / "topics" / f"{tid}.json").write_text(json.dumps(payload, indent=1, ensure_ascii=False))
                state["comment_counts"][tid] = t.get("commentCount")
                fetched += 1
                print(f"  fetched {tid} {t['title'][:70]}")
        state["last_fetch_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        state_path.write_text(json.dumps(state, indent=1))
        print(f"{len(topics)} topics, {fetched} (re)fetched")
    else:
        topics = json.loads((a.out / "topics.json").read_text())

    def payload_of(t):
        p = a.out / "topics" / f"{t['id']}.json"
        return json.loads(p.read_text()) if p.exists() else []

    # Full digest, newest first.
    digest = [f"# {COMPETITION} — discussion digest", f"*generated {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}, "
              f"{len(topics)} topics; excerpts cut at {a.max_chars} chars*", ""]
    for t in topics:
        digest.append(render_topic(t, payload_of(t), a.max_chars))
    (a.out / "digest.md").write_text("\n".join(digest))
    print(f"digest -> {a.out / 'digest.md'}")

    for tid in a.show:
        t = next((x for x in topics if x["id"] == tid), {"id": tid, "title": "?"})
        print(render_topic(t, payload_of(t), 10 ** 9))

    if a.grep:
        pats = [re.compile(p, re.I) for p in a.grep]
        for t in topics:
            for depth, m in iter_messages(payload_of(t)):
                txt = message_text(m)
                hay = t["title"] + "\n" + txt
                if all(p.search(hay) for p in pats):
                    print(f"--- [{t['id']}] {t['title']}  ({author_of(m)}, {str(m.get('postDate', ''))[:10]})")
                    for p in pats:
                        for mm in p.finditer(txt):
                            s = max(0, mm.start() - 160)
                            print("   …" + txt[s:mm.end() + 220].replace("\n", " ") + "…")
                            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

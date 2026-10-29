"""Command line for the same operations: lance-kb ingest|search|note|versions|serve."""

import argparse
import json

from .store import KnowledgeBase


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="lance-kb", description="A LanceDB knowledge base with an MCP server."
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("ingest", help="index a folder of Markdown/text files")
    s.add_argument("folder")
    s.add_argument("--project", default="default")
    s = sub.add_parser("search", help="hybrid search")
    s.add_argument("query")
    s.add_argument("-k", type=int, default=5)
    s.add_argument("--project")
    s.add_argument("--mode", default="hybrid", choices=["hybrid", "vector", "text"])
    s = sub.add_parser("note", help="add a note")
    s.add_argument("title")
    s.add_argument("text")
    s.add_argument("--project", default="default")
    s.add_argument("--tag", action="append", default=[])
    sub.add_parser("versions", help="list versions")
    sub.add_parser("serve", help="run the MCP server on stdio")
    a = p.parse_args(argv)
    if a.cmd == "serve":
        from .server import main as serve

        return serve()
    kb = KnowledgeBase()
    if a.cmd == "ingest":
        print(json.dumps(kb.ingest_folder(a.folder, a.project)))
    elif a.cmd == "search":
        for hit in kb.search(a.query, k=a.k, project=a.project, mode=a.mode):
            where = (
                f"{hit['source']}:{hit['line_start']}-{hit['line_end']}"
                if hit["source"]
                else f"note {hit['id']}"
            )
            print(
                f"{hit['score']:.4f}  {hit['title']}  ({where})\n    {hit['text'][:160]!r}"
            )
    elif a.cmd == "note":
        print(kb.add_note(a.title, a.text, a.project, a.tag))
    elif a.cmd == "versions":
        for v in kb.versions():
            print(v["version"], v["timestamp"])

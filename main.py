import sys
import logging
import dotenv
dotenv.load_dotenv()

from mangrag import ingest, query
from mangrag.db import get_collection

logging.basicConfig(
    level=logging.WARNING,  # keep CLI output clean
    format="%(levelname)s: %(message)s",
)


def _progress(current: int, total: int) -> None:
    print(f"  Embedded {current}/{total}", end="\r", flush=True)


def cmd_ingest(source: str) -> None:
    col = get_collection()
    print(f"Ingesting: {source}")
    count = ingest.run(source, col, on_progress=_progress)
    print(f"\nDone — stored {count} chunks.")


def cmd_ask(user_query: str) -> None:
    print(f"\nQuestion: {user_query}\n")
    answer, docs = query.run(user_query)
    print(f"Answer:\n{answer}\n")
    print("Sources:")
    for d in docs:
        print(f"  score={d['score']:.3f}  page={d.get('page', '?')}  {d.get('source', '')[:60]}")


USAGE = """\
Usage:
  python main.py ingest <pdf_path_or_url>
  python main.py ask "<question>"
"""


def main() -> None:
    if len(sys.argv) < 3:
        print(USAGE)
        sys.exit(1)

    cmd = sys.argv[1]
    arg = sys.argv[2]

    if cmd == "ingest":
        cmd_ingest(arg)
    elif cmd == "ask":
        cmd_ask(arg)
    else:
        print(f"Unknown command: '{cmd}'\n{USAGE}")
        sys.exit(1)


if __name__ == "__main__":
    main()

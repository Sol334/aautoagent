#!/usr/bin/env python3
"""Launch the Galactic Capital A2A Agent Hub on localhost:8001."""
import argparse
import sys
import uvicorn
from pathlib import Path

CAPITAL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CAPITAL_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the Galactic Capital A2A Agent Hub")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    uvicorn.run("api.agent_hub:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()

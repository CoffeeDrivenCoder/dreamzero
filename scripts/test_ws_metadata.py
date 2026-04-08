#!/usr/bin/env python3
"""Minimal websocket connectivity test for DreamZero policy server."""

import argparse
import json
import logging

from eval_utils.policy_client import WebsocketClientPolicy


def main() -> None:
    parser = argparse.ArgumentParser(description="Test websocket connectivity and print server metadata.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6006)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    client = WebsocketClientPolicy(host=args.host, port=args.port)
    metadata = client.get_server_metadata()
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

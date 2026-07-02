from __future__ import annotations

import argparse

import uvicorn

from cognid_genrec.service.kuairec_api import create_kuairec_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve KuaiRec recommender API.")
    parser.add_argument("--data", default="data/processed/kuairec")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8002)
    args = parser.parse_args()
    uvicorn.run(create_kuairec_app(data_dir=args.data), host=args.host, port=args.port)


if __name__ == "__main__":
    main()

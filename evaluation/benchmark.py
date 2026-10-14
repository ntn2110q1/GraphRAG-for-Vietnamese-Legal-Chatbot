"""Run the source-labeled retrieval benchmark using the same backend service code."""

import json

from backend.database.db import SessionLocal, init_db
from backend.services.evaluation_service import run_evaluation


def main() -> None:
    init_db()
    with SessionLocal() as db:
        print(json.dumps(run_evaluation(db, "both", 5), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

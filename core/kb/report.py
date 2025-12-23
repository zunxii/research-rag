from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path


class BuildReport:
    """
    Tracks KB build statistics for reproducibility and auditing.
    """

    def __init__(self):
        self.start_time = datetime.utcnow().isoformat()

        self.total_rows = 0
        self.processed_rows = 0
        self.failed_rows = 0

        self.failures = []  # list of dicts

        self.category_counts = Counter()
        self.anatomy_region_counts = Counter()
        self.anatomy_mention_count = 0

    # -------------------------
    # Logging hooks
    # -------------------------
    def log_row_seen(self):
        self.total_rows += 1

    def log_success(self, category: str, anatomy_region: str):
        self.processed_rows += 1
        self.category_counts[category] += 1
        self.anatomy_region_counts[anatomy_region] += 1

        if anatomy_region != "unknown":
            self.anatomy_mention_count += 1

    def log_failure(self, row_index: int, reason: str):
        self.failed_rows += 1
        self.failures.append({
            "row_index": row_index,
            "reason": reason
        })

    # -------------------------
    # Finalization
    # -------------------------
    def finalize(self):
        self.end_time = datetime.utcnow().isoformat()

    def to_dict(self):
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_rows": self.total_rows,
            "processed_rows": self.processed_rows,
            "failed_rows": self.failed_rows,
            "category_distribution": dict(self.category_counts),
            "anatomy_region_distribution": dict(self.anatomy_region_counts),
            "rows_with_explicit_anatomy": self.anatomy_mention_count,
            "failures": self.failures
        }

    def save(self, path: Path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

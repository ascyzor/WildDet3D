"""Load OWLv2 text labels from the bundled lvisplus CSV."""

import os
from typing import List

_LABELS_DIR = os.path.dirname(os.path.abspath(__file__))


def load_lvisplus_labels() -> List[str]:
    """Return the lvisplus label list (one label per line from lvisplus_classes.csv)."""
    csv_path = os.path.join(_LABELS_DIR, "lvisplus_classes.csv")
    with open(csv_path, "r") as f:
        labels = [line.strip() for line in f if line.strip()]
    return labels
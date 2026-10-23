"""One-off knowledge-base seeding: ``python -m resume_optimizer.seed`` (run as a Kubernetes Job)."""
import logging
from dataclasses import replace

from .config import get_settings
from .vectorstore import build_store, seed

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    store = build_store(replace(get_settings(), seed_on_startup=False))
    print(f"inserted={seed(store)} total={store.count()}")

"""Public site counters: visits and optimized resumes.

Kept in the same Milvus / Zilliz Cloud cluster as the knowledge base when there is one, so the
counts survive app restarts; otherwise in memory. Every event is its own row: an insert is atomic,
so visitors arriving at the same moment can't overwrite each other's count the way a
read-increment-write counter could. Rows hold only the event name and the UTC day, no visitor data.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import Counter
from typing import Optional

from .config import Settings
from .schemas import SiteStats
from .vectorstore import KnowledgeStore, MilvusStore

log = logging.getLogger(__name__)

VISIT, OPTIMIZATION = "visit", "optimization"


def _today() -> str:
    return time.strftime("%Y-%m-%d", time.gmtime())


class SiteCounters:
    label = ""

    def _insert(self, event: str, day: str) -> None:
        raise NotImplementedError

    def _count(self, event: str, day: Optional[str] = None) -> int:
        raise NotImplementedError

    def record(self, event: str) -> None:
        """Never raises: a counter problem must not break the page or a run."""
        try:
            self._insert(event, _today())
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not record a %s (%s)", event, exc)

    def stats(self) -> Optional[SiteStats]:
        """None when the counts can't be read, so the UI can hide the counter."""
        try:
            return SiteStats(visits=self._count(VISIT), visits_today=self._count(VISIT, _today()),
                             optimizations=self._count(OPTIMIZATION), storage=self.label)
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not read the site counters (%s)", exc)
            return None


class MemoryCounters(SiteCounters):
    label = "In-memory (resets on restart)"

    def __init__(self):
        self._counts: Counter = Counter()
        self._lock = threading.Lock()

    def _insert(self, event, day):
        with self._lock:
            self._counts[(event, day)] += 1

    def _count(self, event, day=None):
        with self._lock:
            return sum(n for (e, d), n in self._counts.items() if e == event and day in (None, d))


class MilvusCounters(SiteCounters):
    QUERY_LIMIT = 16384  # Milvus caps query results

    def __init__(self, client, collection: str, label: str):
        self.client, self.collection, self.label = client, collection, label
        if not self.client.has_collection(self.collection):
            try:
                # Milvus requires a vector field, so every row carries the same 2-d placeholder vector.
                self.client.create_collection(collection_name=self.collection, dimension=2, metric_type="L2",
                                              auto_id=True, enable_dynamic_field=True)
                log.info("Created Milvus collection %s", self.collection)
            except Exception:  # noqa: BLE001 - another replica may have created it first
                if not self.client.has_collection(self.collection):
                    raise

    def _insert(self, event, day):
        self.client.insert(collection_name=self.collection, data=[{"vector": [1.0, 0.0], "event": event, "day": day}])

    def _count(self, event, day=None):
        expr = f'event == "{event}"' + (f' and day == "{day}"' if day else "")
        try:
            rows = self.client.query(self.collection, filter=expr, output_fields=["count(*)"],
                                     consistency_level="Strong")
            return int(rows[0]["count(*)"]) if rows else 0
        except Exception:  # noqa: BLE001 - some managed tiers reject count(*); fall back to listing
            return len(self.client.query(self.collection, filter=expr, output_fields=["id"],
                                         limit=self.QUERY_LIMIT, consistency_level="Strong"))


def build_counters(settings: Settings, store: KnowledgeStore) -> SiteCounters:
    """Milvus counters when the knowledge base is in Milvus (sharing its connection), else in memory."""
    if isinstance(store, MilvusStore):
        try:
            return MilvusCounters(store.client, f"{settings.milvus_collection}_events", store.label)
        except Exception as exc:  # noqa: BLE001 - counters are optional; never block start-up
            log.warning("Site counters can't use Milvus (%s); counting in memory", exc)
    return MemoryCounters()

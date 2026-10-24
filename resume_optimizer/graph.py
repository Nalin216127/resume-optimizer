"""LangGraph wiring: a supervisor hub with Context, Writer, Reviewer and Human spokes.

    START → supervisor ─┬─► context  ─┐
                        ├─► writer   ─┤
                        ├─► reviewer ─┼─► supervisor (loop)
                        ├─► human    ─┘   (interrupt: pauses for approval)
                        └─► END
"""
from __future__ import annotations

import logging
import os
import sqlite3

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from .agents import human_review, make_context_agent, make_reviewer_agent, make_writer_agent, supervisor
from .config import Settings
from .llm import LLM
from .state import ResumeState
from .vectorstore import KnowledgeStore

log = logging.getLogger(__name__)

AGENTS = ("context", "writer", "reviewer", "human")


def build_checkpointer(settings: Settings) -> BaseCheckpointSaver:
    backend = settings.checkpoint_backend
    if backend == "memory":
        return MemorySaver()
    if backend == "sqlite":
        from langgraph.checkpoint.sqlite import SqliteSaver

        os.makedirs(os.path.dirname(os.path.abspath(settings.sqlite_path)), exist_ok=True)
        return SqliteSaver(sqlite3.connect(settings.sqlite_path, check_same_thread=False))
    if backend == "postgres":
        # Needed when running several API replicas: every pod must see every paused thread.
        from langgraph.checkpoint.postgres import PostgresSaver
        from psycopg import Connection
        from psycopg.rows import dict_row

        conn = Connection.connect(settings.database_url, autocommit=True, prepare_threshold=0, row_factory=dict_row)
        saver = PostgresSaver(conn)
        saver.setup()
        return saver
    raise ValueError(f"Unknown CHECKPOINT_BACKEND: {backend}")


def build_graph(llm: LLM, store: KnowledgeStore, settings: Settings, checkpointer: BaseCheckpointSaver):
    g = StateGraph(ResumeState)
    g.add_node("supervisor", supervisor)
    g.add_node("context", make_context_agent(llm, store, settings))
    g.add_node("writer", make_writer_agent(llm, settings))
    g.add_node("reviewer", make_reviewer_agent(llm, settings, store.embedder))
    g.add_node("human", human_review)

    g.add_edge(START, "supervisor")
    g.add_conditional_edges(
        "supervisor", lambda s: s["next_agent"], {**{a: a for a in AGENTS}, "end": END}
    )
    for agent in AGENTS:
        g.add_edge(agent, "supervisor")
    return g.compile(checkpointer=checkpointer)

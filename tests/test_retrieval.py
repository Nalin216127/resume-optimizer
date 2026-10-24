"""Knowledge base integrity and hybrid retrieval (BM25 + vector, RRF fusion, metadata filters)."""
import json

import pytest

from resume_optimizer.agents.context import detect_seniority
from resume_optimizer.retrieval import BM25Index, HybridRetriever, tokenize
from resume_optimizer.samples import SAMPLE_JOB, SAMPLE_RESUME
from resume_optimizer.schemas import StartSessionRequest
from resume_optimizer.vectorstore import KNOWLEDGE_FILE, SEED_KNOWLEDGE, HashingEmbedder, InMemoryStore

ROLES = {"backend", "frontend", "ml_ai", "data_eng", "data_sci", "devops", "mobile", "security", "qa", "product", "design"}


@pytest.fixture(scope="module")
def retriever():
    store = InMemoryStore(HashingEmbedder())
    store.add(SEED_KNOWLEDGE)
    return HybridRetriever(store)


def test_knowledge_file_is_well_formed():
    rows = [json.loads(line) for line in open(KNOWLEDGE_FILE)]
    assert len(rows) >= 150
    assert len({r["id"] for r in rows}) == len(rows)
    for r in rows:
        assert r["kind"] in {"guideline", "example_bullet", "role_profile"}
        assert r["seniority"] in {"any", "junior", "mid", "senior"}
        assert set(r["roles"]) <= ROLES | {"any"}
        assert len(r["text"].split()) >= 8
    profiles = {r["roles"][0] for r in rows if r["kind"] == "role_profile"}
    assert profiles == ROLES
    for role in ROLES:  # every role family has a useful number of examples
        assert sum(1 for r in rows if r["kind"] == "example_bullet" and role in r["roles"]) >= 8


def test_tokenize_keeps_tech_terms():
    assert {"ci/cd", "node.js", "c++"} <= set(tokenize("Built CI/CD for Node.js and C++ services"))


def test_bm25_prefers_rare_exact_terms():
    docs = [{"title": "a", "text": "Deployed services on Kubernetes with Helm"},
            {"title": "b", "text": "Deployed services to virtual machines"}]
    hits = BM25Index(docs).search("kubernetes", 5)
    assert [i for i, _ in hits] == [0]


def test_hybrid_marks_which_method_found_each_result(retriever):
    hits = retriever.search("Kubernetes Helm Argo CD deploy", k=3, kind="example_bullet")
    assert hits[0]["id"] == "ex-devops-01"
    assert set(hits[0]["matched_by"]) == {"semantic", "keyword"}
    assert all(h["kind"] == "example_bullet" for h in hits)


def test_filters_by_role_and_seniority(retriever):
    hits = retriever.search("tests automation pipeline", k=10, kind="example_bullet", roles=["qa"], seniority="junior")
    assert hits and all(h["roles"] == ["qa"] and h["seniority"] in ("junior", "any") for h in hits)


def test_keyword_mode_finds_exact_skill_semantic_hashing_may_miss(retriever):
    kw = retriever.search("LangGraph", k=3, mode="keyword")
    assert kw and kw[0]["id"] in {"ex-ml_ai-02", "profile-ml_ai"}


def test_multi_search_fuses_queries(retriever):
    hits = retriever.multi_search(["kafka streaming", "airflow spark etl", "dbt snowflake"], k=5,
                                  kind="example_bullet", roles=["data_eng"])
    ids = {h["id"] for h in hits}
    assert {"ex-data_eng-01", "ex-data_eng-02", "ex-data_eng-03"} <= ids
    assert all(h["queries"] for h in hits)


@pytest.mark.parametrize("jd,role", [
    ("Senior AI Engineer building RAG, LangGraph agents, embeddings and LLM evaluation", "ml_ai"),
    ("Kubernetes, Terraform, CI/CD, Prometheus and SLOs for our platform team", "devops"),
    ("Product Manager to own roadmapping, customer research and experimentation", "product"),
    ("iOS engineer with Swift, SwiftUI and App Store release experience", "mobile"),
    ("Data engineer for Airflow, Spark, Kafka and dbt pipelines on Snowflake", "data_eng"),
])
def test_detect_roles(retriever, jd, role):
    assert retriever.detect_roles(jd)[0] == role


def test_detect_seniority():
    assert detect_seniority("Senior AI Engineer") == "senior"
    assert detect_seniority("New grad software engineer") == "junior"
    assert detect_seniority("Product Manager") == "mid"


def test_add_refreshes_keyword_index(retriever):
    retriever.add([{"id": "custom-1", "kind": "guideline", "title": "Quantum", "text": "Mention qubit error correction work explicitly."}])
    assert retriever.search("qubit", k=1, mode="keyword")[0]["id"] == "custom-1"


def test_session_exposes_rag_sources(service):
    view = service.start(StartSessionRequest(resume_text=SAMPLE_RESUME, job_description=SAMPLE_JOB, max_auto_revisions=0))
    assert view.target_roles and view.target_roles[0] == "ml_ai"
    assert any(d["kind"] == "role_profile" for d in view.retrieved)
    assert view.exemplars and all(e["kind"] == "example_bullet" for e in view.exemplars)
    assert all("ml_ai" in e["roles"] for e in view.exemplars)



def test_cascade_follows_the_embedders_preferred_method(retriever):
    # Hashing embedder prefers keyword: the cascade's order equals BM25's.
    assert retriever.primary_method == "keyword"
    q = "Terraform infrastructure as code policy"
    cascade = [h["id"] for h in retriever.search(q, k=5, kind="example_bullet")]
    keyword = [h["id"] for h in retriever.search(q, k=5, kind="example_bullet", mode="keyword")]
    assert cascade[:len(keyword)] == keyword[:len(cascade)]


def test_semantic_failure_falls_back_to_keyword():
    class DownEmbedder(HashingEmbedder):
        preferred_method = "semantic"

        def embed(self, text, task="x"):
            raise RuntimeError("429 rate limited")

    store = InMemoryStore(HashingEmbedder())
    store.add(SEED_KNOWLEDGE)
    store.embedder = DownEmbedder()  # documents already embedded; queries now fail
    r = HybridRetriever(store)
    hits = r.search("Kubernetes Helm Argo CD", k=3, kind="example_bullet")
    assert hits[0]["id"] == "ex-devops-01" and hits[0]["matched_by"] == ["keyword"]


def test_rrf_option_still_available(retriever):
    hits = retriever.search("Kafka Flink streaming", k=3, kind="example_bullet", fusion="rrf", keyword_weight=0.5)
    assert hits[0]["id"] == "ex-data_eng-03"


def test_precomputed_vectors_cover_the_knowledge_base():
    from resume_optimizer.vectorstore import _embed_text, _normalise_doc, cache_key, load_precomputed, precomputed_path

    vectors = load_precomputed(precomputed_path("gemini-embedding-001", 768))
    keys = {cache_key("RETRIEVAL_DOCUMENT", _embed_text(_normalise_doc(d))) for d in SEED_KNOWLEDGE}
    assert keys <= set(vectors), "re-run scripts/precompute_embeddings.py after editing knowledge.jsonl"
    assert all(len(v) == 768 for v in list(vectors.values())[:5])

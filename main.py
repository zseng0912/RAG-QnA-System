import logging
from fastapi import FastAPI
import inngest
import inngest.fast_api
from inngest.experimental import ai
from dotenv import load_dotenv
import uuid
import os
import datetime
import re
from data_loader import load_and_chunk, embed_texts
from vector_db import QdrantStorage
from custom_types import RAGQueryResult, RAGSearchResult, RAGUpsertResult, RAGChunkAndSrc

load_dotenv()

inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer()
)

@inngest_client.create_function(
    fn_id="RAG: Ingest Source",
    trigger=inngest.TriggerEvent(event="rag/ingest_source"),
    # Easily add Throttling with Flow Control
    throttle=inngest.Throttle(
        limit=2,
        period=datetime.timedelta(minutes=2) # 2 execution/min, it will delay the execution not reject if the limit is reach
    ),
    rate_limit=inngest.RateLimit(
        limit=1,
        period=datetime.timedelta(hours=4), # limit 1 same file within 4 hrs
        key="event.data.source_id",
    ),
)
async def rag_ingest_source(ctx: inngest.Context):
    def _load(ctx: inngest.Context) -> RAGChunkAndSrc:
        input_str = ctx.event.data["input"]
        source_id = ctx.event.data.get("source_id", input_str)
        chunks = load_and_chunk(input_str)
        return RAGChunkAndSrc(chunks=chunks, source_id=source_id)

    def _upsert(chunks_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
        chunks = chunks_and_src.chunks
        source_id = chunks_and_src.source_id
        vecs = embed_texts(chunks)
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}")) for i in range(len(chunks))]
        payloads = [{"source": source_id, "text":chunks[i]} for i in range(len(chunks))]
        store = QdrantStorage()
        # Write chunk-level vectors into the primary document index
        store.upsert(ids, vecs, payloads)

        
        sentences = []
        sent_ids = []
        sent_payloads = []
        for i, t in enumerate(chunks):
            # Split each chunk into sentences for sentence-window retrieval
            s_list = [s for s in re.split(r"(?<=[.!?])\s+", t) if s.strip()]
            for j, s in enumerate(s_list):
                sid = f"{source_id}:{i}:{j}"
                sent_ids.append(sid)
                sentences.append(s)
                # Lightweight entity extraction for graph-style overlap scoring
                ents = [w for w in re.findall(r"[A-Z][a-zA-Z0-9_]+", s)]
                sent_payloads.append({"source": source_id, "text": s, "chunk_index": i, "sentence_index": j, "entities": ents})
        if sentences:
            # Store sentence vectors and metadata into dedicated sentence index
            sent_vecs = embed_texts(sentences)
            store.upsert_sentences(sent_ids, sent_vecs, sent_payloads)
        return RAGUpsertResult(ingested=len(chunks))

    chunks_and_src = await ctx.step.run("load-and-chunk", lambda: _load(ctx), output_type=RAGChunkAndSrc)
    ingested = await ctx.step.run("embed-and-upsert", lambda: _upsert(chunks_and_src), output_type=RAGUpsertResult)
    return ingested.model_dump()

@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai")
)
async def rag_query_pdf_ai(ctx: inngest.Context):
    def _search(question: str, top_k: int =5) -> RAGSearchResult:
        query_vec = embed_texts([question])[0]
        store = QdrantStorage()
        # Retriever 1: document/chunk vector search
        vec_items = store.search_docs_detailed(query_vec, top_k*2)
        # Retriever 2: sentence-level search (enables windowed context)
        sent_items = store.search_sentences_detailed(query_vec, top_k*4)
        # Retriever 3 (graph-style): extract entities from the question for overlap scoring
        q_ents = [w for w in re.findall(r"[A-Z][a-zA-Z0-9_]+", question)]
        candidates = []
        for it in vec_items[:top_k*2]:
            if it.get("text"):
                # Seed candidates from document retriever (uses vector score as base)
                candidates.append({"text": it["text"], "source": it.get("source"), "base": it.get("score", 0.0) or 0.0})
        for it in sent_items[:top_k*2]:
            sid = str(it["id"]) if it.get("id") is not None else None
            window = []
            if sid:
                parts = str(sid).split(":")
                if len(parts) >= 3:
                    s = parts[0]
                    ci = int(parts[1])
                    si = int(parts[2])
                    # Assemble surrounding sentences to preserve local continuity
                    ids = [f"{s}:{ci}:{k}" for k in range(max(0, si-2), si+3)]
                    payloads = store.retrieve_sentences_by_ids(ids)
                    texts = [p.get("text", "") for k,p in payloads.items() if p]
                    window = [t for t in texts if t]
            win_text = it.get("text") if not window else " ".join(window)
            if win_text:
                # Add sentence-window candidates with base from sentence vector score
                candidates.append({"text": win_text, "source": it.get("source"), "base": it.get("score", 0.0) or 0.0})
        for it in sent_items[:top_k*2]:
            ents = it.get("entities", [])
            overlap = len(set(q_ents) & set(ents))
            if overlap > 0:
                sid = str(it["id"]) if it.get("id") is not None else None
                window = []
                if sid:
                    parts = str(sid).split(":")
                    if len(parts) >= 3:
                        s = parts[0]
                        ci = int(parts[1])
                        si = int(parts[2])
                        # Smaller window focused around entity-bearing sentence
                        ids = [f"{s}:{ci}:{k}" for k in range(max(0, si-1), si+2)]
                        payloads = store.retrieve_sentences_by_ids(ids)
                        texts = [p.get("text", "") for k,p in payloads.items() if p]
                        window = [t for t in texts if t]
                win_text = it.get("text") if not window else " ".join(window)
                # Boost base score with entity-overlap as graph-style signal
                score = (it.get("score", 0.0) or 0.0) + 0.05*overlap
                candidates.append({"text": win_text, "source": it.get("source"), "base": score})
        uniq = {}
        for c in candidates:
            t = c["text"]
            s = c.get("source") or ""
            b = c.get("base") or 0.0
            if t in uniq:
                if b > uniq[t]["base"]:
                    uniq[t] = {"base": b, "source": s}
            else:
                uniq[t] = {"base": b, "source": s}
        texts = list(uniq.keys())
        # Re-rank: compute semantic relevance (cosine) against the question
        vecs = embed_texts(texts) if texts else []
        def _cos(a, b):
            import math
            s = sum(x*y for x,y in zip(a,b))
            na = math.sqrt(sum(x*x for x in a))
            nb = math.sqrt(sum(x*x for x in b))
            return s/(na*nb) if na and nb else 0.0
        scored = []
        for i,t in enumerate(texts):
            sim = _cos(vecs[i], query_vec) if i < len(vecs) else 0.0
            base = uniq[t]["base"]
            # Final score blends semantic similarity and base retrieval signal
            scored.append((t, uniq[t]["source"], 0.7*sim + 0.3*base))
        scored.sort(key=lambda x: x[2], reverse=True)
        contexts = [x[0] for x in scored[:top_k]]
        sources = []
        for x in scored[:top_k]:
            if x[1] and x[1] not in sources:
                sources.append(x[1])
        return RAGSearchResult(contexts=contexts, sources=sources)

    question = ctx.event.data["question"]
    top_k = int(ctx.event.data.get("top_k", 5))

    found = await ctx.step.run("embed-and-search", lambda: _search(question, top_k), output_type=RAGSearchResult)

    context_block = "\n\n".join(f"- {c}" for c in found.contexts)
    user_content = (
        "Use the following context to answer the question.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n"
        "Answer concisely using the context above."
    )

    adapter = ai.openai.Adapter(
        auth_key =os.getenv("GOOGLE_API_KEY"),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        model="gemini-2.5-flash"
    )

    res = await ctx.step.ai.infer(
        "llm-answer",
        adapter=adapter,
        body={
            "max_tokens": 1024,
            "temperature": 0.2,
            "messages":[
                {"role": "system", "content": "You answer questions using only the provided context."},
                {"role": "user", "content": user_content}
            ]
        }
    )

    answer = res["choices"][0]["message"]["content"].strip()
    return{"answer": answer, "sources": found.sources, "num_contexts":len(found.contexts)}

app = FastAPI()

inngest.fast_api.serve(app, inngest_client, [rag_ingest_source, rag_query_pdf_ai])
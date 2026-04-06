# Ragpipe Quality Baseline (Phase 0)

**Date:** 2026-04-06 18:45 UTC
**Eval pairs:** 14
**Target:** ragpipe (Qwen3-32B Q4_K_M via Vulkan/gfx1151)
**Judge:** Qwen3-32B Q4_K_M (same model, direct endpoint)

## Aggregate Scores

| Metric | Score |
|--------|-------|
| Faithfulness | 0.700 |
| Answer Relevance | 0.843 |
| Context Precision | 0.714 |
| Context Recall | 0.250 |

## Per-Route Scores

| Route | N | Faithfulness | Answer Rel. | Context Prec. | Context Recall |
|-------|---|--------------|-------------|---------------|----------------|
| personnel | 4 | 0.967 | 0.975 | 0.933 | 0.500 |
| analysis | 3 | 1.000 | 0.733 | 1.000 | N/A |
| lookup | 4 | 0.333 | 0.700 | 0.400 | N/A |
| general | 3 | N/A | 0.967 | N/A | 0.000 |

## Per-Query Results

| # | Route | Question | Grounding | F | AR | CP | CR |
|---|-------|----------|-----------|---|----|----|-----|
| 1 | personnel | What is Adam Clater's current job title? | general | N/A | 1.000 | N/A | 0.000 |
| 2 | personnel | Who does Adam Clater work for and how long has he ... | corpus | 1.000 | 1.000 | 1.000 | 1.000 |
| 3 | personnel | Describe Adam Clater's professional background | corpus | 1.000 | 1.000 | 1.000 | N/A |
| 4 | analysis | What are the key trends in NATO AI adoption strate... | general | N/A | 1.000 | N/A | N/A |
| 5 | analysis | Summarize the NATO capability targets analysis | corpus | 1.000 | 0.700 | 1.000 | N/A |
| 6 | analysis | What does the defense authorization act say about ... | general | N/A | 0.500 | N/A | N/A |
| 7 | lookup | What does the MPEP say about prior art requirement... | corpus | 1.000 | 0.900 | 0.800 | N/A |
| 8 | lookup | What section of the patent manual covers patent el... | corpus | 0.000 | 1.000 | 0.000 | N/A |
| 9 | lookup | What are the MPEP requirements for a valid patent ... | corpus | 0.000 | 0.900 | 0.400 | N/A |
| 10 | general | What is the capital of France? | general | N/A | 1.000 | N/A | 0.000 |
| 11 | general | Explain how TCP/IP works | general | N/A | 1.000 | N/A | N/A |
| 12 | personnel | Compare Adam Clater's experience with the NATO AI ... | corpus | 0.900 | 0.900 | 0.800 | N/A |
| 13 | general | What is the meaning of life? | general | N/A | 0.900 | N/A | 0.000 |
| 14 | lookup | Tell me about the patent requirements for AI syste... | general | N/A | 0.000 | N/A | N/A |

## Notes

- This baseline was recorded before any agentic RAG improvements (CRAG, Self-RAG, etc.)
- The judge model is the same LLM as the response model — scores may be optimistic
- Context Recall requires ground_truth; queries without it show N/A
- General route has `rag_enabled: false` so context metrics reflect no-RAG behavior
- Contexts hydrated from Postgres docstore (chunks table) using cited chunk IDs
- All future agentic improvements must demonstrate improvement relative to these scores

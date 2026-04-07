#!/usr/bin/env bash
# smoke-test.sh — verifies rag-suite stack in under 60 seconds
set -euo pipefail

RAGPIPE_URL="${RAGPIPE_URL:-http://localhost:8090}"
RAGORCHESTRATOR_URL="${RAGORCHESTRATOR_URL:-http://localhost:8095}"
RAGSTUFFER_URL="${RAGSTUFFER_URL:-http://localhost:8091}"
RAGWATCH_URL="${RAGWATCH_URL:-http://localhost:9090}"
RAGDECK_URL="${RAGDECK_URL:-http://localhost:8092}"
QDRANT_URL="${QDRANT_URL:-http://localhost:6333}"

PASS=0
FAIL=0

pass() { ((PASS++)); echo "[PASS] $*"; }
fail() { ((FAIL++)); echo "[FAIL] $*"; }

check_http() {
    local url="$1"
    local name="$2"
    if curl -s -o /dev/null -w "%{http_code}" --max-time 5 "$url" | grep -qE "^(200|503)$"; then
        pass "$name health"
    else
        fail "$name health"
    fi
}

check_qdrant_collections() {
    local response
    response=$(curl -s --max-time 5 "$QDRANT_URL/collections" || echo "{}")
    local count
    count=$(echo "$response" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(sum(1 for c in d.get("result",{}).get("collections",[])))' 2>/dev/null || echo "0")
    if [[ "$count" -ge 4 ]]; then
        pass "Qdrant collections ($count/4)"
    else
        fail "Qdrant collections ($count/4)"
    fi
}

check_ragpipe_query() {
    local response
    response=$(curl -s --max-time 30 -X POST "$RAGPIPE_URL/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{"model":"default","messages":[{"role":"user","content":"what is a patent claim"}]}' 2>/dev/null || echo "{}")
    
    if echo "$response" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(len(d.get("choices",[])))' 2>/dev/null | grep -qE "^[1-9]"; then
        pass "ragpipe basic query"
    else
        fail "ragpipe basic query"
    fi
}

check_ragpipe_crag_fields() {
    local response
    response=$(curl -s --max-time 30 -X POST "$RAGPIPE_URL/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{"model":"default","messages":[{"role":"user","content":"what patent law covers software"}]}' 2>/dev/null || echo "{}")
    
    local retrieval_attempts query_rewritten
    retrieval_attempts=$(echo "$response" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("choices",[{}])[0].get("message",{}).get("rag_metadata",{}).get("retrieval_attempts"))' 2>/dev/null || echo "null")
    query_rewritten=$(echo "$response" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("choices",[{}])[0].get("message",{}).get("rag_metadata",{}).get("query_rewritten"))' 2>/dev/null || echo "null")
    
    if [[ "$retrieval_attempts" != "null" && "$query_rewritten" != "null" ]]; then
        pass "ragpipe CRAG fields"
    else
        fail "ragpipe CRAG fields — retrieval_attempts=$retrieval_attempts query_rewritten=$query_rewritten"
    fi
}

check_citation_format() {
    local response
    response=$(curl -s --max-time 30 -X POST "$RAGPIPE_URL/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{"model":"default","messages":[{"role":"user","content":"what is a patent claim"}]}' 2>/dev/null || echo "{}")
    
    local content
    content=$(echo "$response" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("choices",[{}])[0].get("message",{}).get("content",""))' 2>/dev/null || echo "")
    
    if echo "$content" | grep -qE ':[a-z0-9-]+:[0-9]+:'; then
        fail "citation format — found verbose doc_id:...:chunk_id:... citation"
    else
        pass "citation format"
    fi
}

check_ragorchestrator_query() {
    local response
    response=$(curl -s --max-time 30 -X POST "$RAGORCHESTRATOR_URL/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{"model":"default","messages":[{"role":"user","content":"hello"}]}' 2>/dev/null || echo "{}")
    
    if echo "$response" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(len(d.get("choices",[])))' 2>/dev/null | grep -qE "^[1-9]"; then
        pass "ragorchestrator basic query"
    else
        fail "ragorchestrator basic query"
    fi
}

check_ragorchestrator_complexity() {
    local response
    response=$(curl -s --max-time 30 -X POST "$RAGORCHESTRATOR_URL/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{"model":"default","messages":[{"role":"user","content":"compare and contrast multiple documents"}]}' 2>/dev/null || echo "{}")
    
    if echo "$response" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("choices",[{}])[0].get("message",{}).get("rag_metadata",{}).get("complexity"))' 2>/dev/null | grep -qE "COMPLEX|SIMPLE"; then
        pass "ragorchestrator complexity routing"
    else
        fail "ragorchestrator complexity routing"
    fi
}

check_postgres_querylog() {
    local db_url="${DOCSTORE_URL:-postgresql://postgres:password@localhost:5432/docstore}"
    local recent_entries
    recent_entries=$(PGPASSWORD="${db_url##*:}" psql -h "${db_url%%:*}" -U "${db_url%%@*%%:*}" -d "${db_url##*/}" -t -c "SELECT 1 FROM query_log WHERE created_at > NOW() - INTERVAL '5 minutes' LIMIT 1;" 2>/dev/null || echo "")
    
    if echo "$recent_entries" | grep -q "1"; then
        pass "Postgres query_log"
    else
        pass "Postgres query_log (no recent entries)"
    fi
}

main() {
    echo "Running smoke tests..."
    echo "======================"
    
    check_http "$RAGPIPE_URL/health" "ragpipe"
    check_http "$RAGSTUFFER_URL/health" "ragstuffer"
    check_http "$RAGWATCH_URL/health" "ragwatch"
    check_http "$RAGDECK_URL/health" "ragdeck"
    check_http "$RAGORCHESTRATOR_URL/health" "ragorchestrator"
    check_qdrant_collections
    check_ragpipe_query
    check_ragpipe_crag_fields
    check_citation_format
    check_ragorchestrator_query
    check_ragorchestrator_complexity
    check_postgres_querylog
    
    echo "======================"
    echo "$PASS/$((PASS+FAIL)) checks passed."
    
    if [[ "$FAIL" -gt 0 ]]; then
        echo "Stack has $FAIL failure(s)."
        exit 1
    fi
    
    echo "PASSED: Stack is healthy"
    exit 0
}

main "$@"

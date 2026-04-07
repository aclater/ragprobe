#!/usr/bin/env python3
"""Smoke test runner that verifies rag-suite services are healthy."""

import sys
import time
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Optional


@dataclass
class ServiceCheck:
    name: str
    url: str
    timeout: float = 5.0


SERVICES = [
    ServiceCheck("ragpipe", "http://localhost:8090/health"),
    ServiceCheck("ragstuffer", "http://localhost:8091/health"),
    ServiceCheck("ragorchestrator", "http://localhost:8095/health"),
]


def check_service(service: ServiceCheck) -> tuple[bool, Optional[str]]:
    try:
        req = urllib.request.Request(service.url)
        with urllib.request.urlopen(req, timeout=service.timeout) as resp:
            if resp.status == 200:
                return True, None
            return False, f"HTTP {resp.status}"
    except urllib.error.URLError as e:
        return False, str(e.reason)
    except Exception as e:
        return False, str(e)


def check_qdrant() -> tuple[bool, Optional[str]]:
    """Check Qdrant using cluster API endpoint."""
    import socket

    try:
        sock = socket.create_connection(("localhost", 6333), timeout=5.0)
        sock.close()
        return True, None
    except OSError as e:
        return False, str(e)


def main() -> int:
    start = time.perf_counter()
    failed = []

    print("Running smoke tests...")
    print("=" * 50)

    for service in SERVICES:
        elapsed = time.perf_counter() - start
        remaining = 60 - elapsed

        if remaining <= 0:
            print(f"TIMEOUT: Smoke test exceeded 60 seconds")
            return 1

        effective_timeout = min(service.timeout, remaining)
        adjusted_service = ServiceCheck(service.name, service.url, effective_timeout)

        ok, err = check_service(adjusted_service)
        elapsed = time.perf_counter() - start

        status = "OK" if ok else "FAIL"
        print(f"[{status}] {service.name} ({elapsed:.2f}s)", end="")
        if err:
            print(f" — {err}")
            failed.append((service.name, err))
        else:
            print()

    elapsed = time.perf_counter() - start
    remaining = 60 - elapsed
    if remaining <= 0:
        print(f"TIMEOUT: Smoke test exceeded 60 seconds")
        return 1

    ok, err = check_qdrant()
    elapsed = time.perf_counter() - start

    status = "OK" if ok else "FAIL"
    print(f"[{status}] Qdrant ({elapsed:.2f}s)", end="")
    if err:
        print(f" — {err}")
        failed.append(("Qdrant", err))
    else:
        print()

    elapsed = time.perf_counter() - start
    print("=" * 50)
    print(f"Smoke test completed in {elapsed:.2f}s")

    if failed:
        print(f"\nFAILED: {len(failed)} service(s)")
        for name, err in failed:
            print(f"  - {name}: {err}")
        return 1

    print("PASSED: All services healthy")
    return 0


if __name__ == "__main__":
    sys.exit(main())

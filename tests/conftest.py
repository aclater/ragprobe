"""Shared pytest configuration for ragprobe tests."""


def pytest_addoption(parser):
    parser.addoption(
        "--ragpipe-url",
        default="http://127.0.0.1:8090",
        help="ragpipe endpoint URL",
    )
    parser.addoption(
        "--ragorchestrator-url",
        default="http://127.0.0.1:8095",
        help="ragorchestrator endpoint URL",
    )
    parser.addoption(
        "--live",
        action="store_true",
        default=False,
        help="Run live integration tests against running services",
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--live"):
        skip_live = __import__("pytest").mark.skip(reason="need --live option to run")
        for item in items:
            if "live" in item.keywords:
                item.add_marker(skip_live)

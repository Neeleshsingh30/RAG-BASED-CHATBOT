from routing import query_router


class FakeRouterChain:
    def __init__(self, response: str):
        self.response = response

    def invoke(self, data):
        return self.response


def test_classify_general_query(monkeypatch):
    monkeypatch.setattr(
        query_router,
        "_get_router_chain",
        lambda: FakeRouterChain("general"),
    )

    result = query_router.classify_query("hi")

    assert result == "general"


def test_classify_document_query(monkeypatch):
    monkeypatch.setattr(
        query_router,
        "_get_router_chain",
        lambda: FakeRouterChain("document"),
    )

    result = query_router.classify_query(
        "What does the document say about refunds?"
    )

    assert result == "document"


def test_unexpected_router_output_defaults_to_document(monkeypatch):
    monkeypatch.setattr(
        query_router,
        "_get_router_chain",
        lambda: FakeRouterChain("something unexpected"),
    )

    result = query_router.classify_query("What is the refund policy?")

    assert result == "document"
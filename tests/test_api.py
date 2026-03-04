import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from ai_dentistry.api.app import app


def test_templates_endpoint_returns_default_template() -> None:
    client = TestClient(app)
    response = client.get("/api/templates")
    assert response.status_code == 200
    payload = response.json()
    assert payload["default_template_id"] == "dental_default"
    assert len(payload["templates"]) >= 1


def test_protocol_validate_endpoint_reports_error() -> None:
    client = TestClient(app)
    response = client.post(
        "/api/protocol/validate",
        json={"protocol": {"queries": [], "classification": {}, "outputs": {}}},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["valid"] is False
    assert "missing" in payload["error"].lower() or "must define" in payload["error"].lower()

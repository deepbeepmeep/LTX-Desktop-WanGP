"""Tests for /api/runtime-policy endpoint."""

from __future__ import annotations


def test_runtime_policy_true(client, test_state):
    test_state.config.local_generations_mode = "unsupported"

    response = client.get("/api/runtime-policy")
    assert response.status_code == 200
    assert response.json() == {"force_api_generations": True}


def test_runtime_policy_false(client, test_state):
    test_state.config.local_generations_mode = "full_models_loading"

    response = client.get("/api/runtime-policy")
    assert response.status_code == 200
    assert response.json() == {"force_api_generations": False}

import os

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_users.db")
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)
os.environ["APP_DB_PATH"] = DB_PATH

from fastapi.testclient import TestClient

from api.index import app


client = TestClient(app)


def test_register_user_success():
    response = client.post(
        "/api/register",
        json={
            "name": "Alice Example",
            "email": "alice@example.com",
            "password": "StrongPass123!",
        },
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["user"]["email"] == "alice@example.com"
    assert payload["user"]["name"] == "Alice Example"
    assert "password" not in payload["user"]


def test_register_user_duplicate_email_fails():
    response = client.post(
        "/api/register",
        json={
            "name": "Alice Again",
            "email": "alice@example.com",
            "password": "AnotherPass123!",
        },
    )

    assert response.status_code == 400
    assert "already registered" in response.json()["detail"].lower()

import os

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_users.db")
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)
os.environ["APP_DB_PATH"] = DB_PATH

from fastapi.testclient import TestClient

from api.index import app


client = TestClient(app)


def test_login_success():
    client.post(
        "/api/register",
        json={
            "name": "Bob User",
            "email": "bob@example.com",
            "password": "StrongPass123!",
        },
    )

    response = client.post(
        "/api/login",
        json={
            "email": "bob@example.com",
            "password": "StrongPass123!",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["user"]["email"] == "bob@example.com"
    assert "token" in payload
    assert payload["message"] == "Login successful."


def test_login_wrong_password_fails():
    response = client.post(
        "/api/login",
        json={
            "email": "bob@example.com",
            "password": "WrongPassword123!",
        },
    )

    assert response.status_code == 401
    assert "invalid email or password" in response.json()["detail"].lower()


def test_me_route_requires_auth():
    response = client.get("/api/me")

    assert response.status_code == 401


def test_optimize_requires_auth():
    response = client.post(
        "/api/optimize",
        files={"resume_file": ("resume.txt", "John Doe\nPython developer\n", "text/plain")},
        data={"job_description_text": "We need a Python engineer."},
    )

    assert response.status_code == 401


def test_me_route_returns_logged_in_user():
    login_response = client.post(
        "/api/login",
        json={
            "email": "bob@example.com",
            "password": "StrongPass123!",
        },
    )
    token = login_response.json()["token"]

    response = client.get(
        "/api/me",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["email"] == "bob@example.com"


def test_dashboard_requires_auth():
    response = client.get("/api/dashboard")

    assert response.status_code == 401


def test_dashboard_returns_user_and_recent_activity():
    login_response = client.post(
        "/api/login",
        json={
            "email": "bob@example.com",
            "password": "StrongPass123!",
        },
    )
    token = login_response.json()["token"]

    response = client.get(
        "/api/dashboard",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["user"]["email"] == "bob@example.com"
    assert "recent_activity" in payload


def test_login_missing_user_fails():
    response = client.post(
        "/api/login",
        json={
            "email": "nobody@example.com",
            "password": "StrongPass123!",
        },
    )

    assert response.status_code == 401
    assert "invalid email or password" in response.json()["detail"].lower()

import os

import requests


class ApiError(RuntimeError):
    pass


class ApiClient:
    def __init__(self, base_url: str | None = None, token: str | None = None):
        self.base_url = (base_url or os.getenv("LEGAL_API_URL", "http://127.0.0.1:8000")).rstrip("/")
        self.token = token or ""

    def request(self, method: str, path: str, **kwargs):
        headers = kwargs.pop("headers", {})
        if self.token:
            headers["X-Admin-Token"] = self.token
        try:
            response = requests.request(method, f"{self.base_url}{path}", headers=headers, timeout=kwargs.pop("timeout", 30), **kwargs)
        except requests.RequestException as exc:
            raise ApiError(f"Không kết nối được backend tại {self.base_url}. Hãy khởi động FastAPI rồi thử lại.") from exc
        if response.status_code >= 400:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            raise ApiError(f"HTTP {response.status_code}: {detail}")
        if response.status_code == 204:
            return None
        try:
            return response.json()
        except ValueError:
            return response.text

    def get(self, path: str, **kwargs):
        return self.request("GET", path, **kwargs)

    def post(self, path: str, **kwargs):
        return self.request("POST", path, **kwargs)

    def patch(self, path: str, **kwargs):
        return self.request("PATCH", path, **kwargs)

    def put(self, path: str, **kwargs):
        return self.request("PUT", path, **kwargs)

    def delete(self, path: str, **kwargs):
        return self.request("DELETE", path, **kwargs)

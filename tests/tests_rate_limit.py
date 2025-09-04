import asyncio
import types
import pytest

from src.ingestion import fetch_data as FD


class DummyResponse:
    def __init__(self, status_code=200, json_data=None, headers=None):
        self.status_code = status_code
        self._json = json_data if json_data is not None else {}
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            import httpx
            request = httpx.Request("GET", "http://test")
            response = httpx.Response(self.status_code, request=request, headers=self.headers)
            raise httpx.HTTPStatusError("error", request, response)

    def json(self):
        return self._json


class DummyClient:
    def __init__(self, sequence):
        # sequence of (status_code, json, headers)
        self._seq = list(sequence)
        self.calls = 0

    async def get(self, path, params=None):
        self.calls += 1
        if self._seq:
            sc, js, hdrs = self._seq.pop(0)
        else:
            sc, js, hdrs = 200, {}, {}
        return DummyResponse(sc, js, hdrs)


@pytest.mark.asyncio
async def test_backoff_on_429(monkeypatch):
    # Prepare dummy client: two 429 with no Retry-After, then 200
    client = DummyClient([
        (429, {}, {}),
        (429, {}, {}),
        (200, {"ok": True}, {}),
    ])

    async def _fake_get_client():
        return client

    # Patch client getter
    monkeypatch.setattr(FD, "_get_client", _fake_get_client)

    # Capture sleeps and force deterministic jitter
    sleeps = []

    async def fake_sleep(t):
        sleeps.append(float(t))
        return None

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    monkeypatch.setattr("random.random", lambda: 0.0)

    out = await FD._get_json("/x", {"a": 1}, symbol_key="BTC-USD")
    assert out == {"ok": True}
    # Expect two backoff sleeps with base backoff from settings (0.25s, 0.5s) capped by max
    assert len(sleeps) == 2
    assert sleeps[0] >= 0.25 - 1e-6
    assert sleeps[1] >= 0.5 - 1e-6


@pytest.mark.asyncio
async def test_single_flight(monkeypatch):
    # Prepare a client that returns after a delay once
    calls = {"n": 0}

    class SlowClient:
        async def get(self, path, params=None):
            calls["n"] += 1
            # Return simple coinbase candles schema (timestamp, low, high, open, close, volume)
            await asyncio.sleep(0.01)
            return DummyResponse(200, [[1, 10, 20, 12, 18, 100]])

    async def _fake_get_client():
        return SlowClient()

    monkeypatch.setattr(FD, "_get_client", _fake_get_client)

    async def run_two():
        t1 = asyncio.create_task(FD.fetch_ohlcv("BTC-USD", interval="1m", limit=1, use_cache=False))
        t2 = asyncio.create_task(FD.fetch_ohlcv("BTC-USD", interval="1m", limit=1, use_cache=False))
        r1, r2 = await asyncio.gather(t1, t2)
        return r1, r2

    out1, out2 = await run_two()
    # Only one underlying HTTP call should have been made due to single-flight
    assert calls["n"] == 1
    assert out1 == out2
    assert isinstance(out1, list)
    assert out1 and "t" in out1[0]


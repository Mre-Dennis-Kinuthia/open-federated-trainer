"""
Coarse, privacy-preserving presence map of active clients.

Client IPs are geolocated to city level via ip-api.com, rounded to one
decimal degree, and offset per-client by a deterministic jitter. Raw IPs
are kept only in an in-memory cache keyed for deduplication; they are
never exposed by the API or written to disk.
"""

import ipaddress
import json
import logging
import os
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_LOOKUP_FIELDS = "status,lat,lon,city,country"
_LOOKUP_TIMEOUT_SECONDS = 5


def _is_private(ip: str) -> bool:
    if not ip:
        return True
    try:
        return not ipaddress.ip_address(ip).is_global
    except ValueError:
        return True


def _fnv1a(value: str) -> int:
    h = 0x811C9DC5
    for ch in value.encode("utf-8"):
        h ^= ch
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


class GeoPresence:
    """Track when and (coarsely) where clients last contacted the coordinator."""

    def __init__(self, state_path: Optional[str] = None, repo=None):
        default_path = (
            Path(__file__).resolve().parents[2] / "data" / "geo_presence.json"
        )
        self.state_path = Path(
            state_path or os.getenv("GEO_STATE_PATH", str(default_path))
        )
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.repo = repo
        self._lock = threading.Lock()
        # client_id -> {lat, lng, city, country, last_seen}
        self._clients: Dict[str, Dict[str, Any]] = {}
        # ip -> {lat, lng, city, country} | {"failed_at": ts}; never persisted
        self._ip_cache: Dict[str, Dict[str, Any]] = {}
        self._pending_ips: set = set()
        self._last_persist = 0.0
        self._load()

    @property
    def lookup_enabled(self) -> bool:
        return os.getenv("GEO_LOOKUP_DISABLED", "false").lower() not in (
            "1",
            "true",
            "yes",
        )

    def _load(self) -> None:
        if self.repo is not None:
            try:
                self._clients = self.repo.list_all()
                return
            except Exception as exc:
                logger.warning(f"Cannot load geo presence from SQL: {exc}")
        if not self.state_path.exists():
            return
        try:
            raw = json.loads(self.state_path.read_text(encoding="utf-8"))
            self._clients = dict(raw.get("clients", {}))
        except (OSError, ValueError) as exc:
            logger.warning(f"Cannot load geo presence state: {exc}")

    def _persist(self, force: bool = False) -> None:
        now = time.time()
        if not force and now - self._last_persist < 5:
            return
        self._last_persist = now
        if self.repo is not None:
            try:
                for client_id, entry in self._clients.items():
                    safe = {
                        k: v
                        for k, v in entry.items()
                        if k in ("lat", "lng", "city", "country", "last_seen")
                    }
                    self.repo.upsert(client_id, safe)
            except Exception as exc:
                logger.warning(f"Cannot persist geo presence to SQL: {exc}")
            return
        payload = {"version": 1, "clients": self._clients}
        try:
            tmp = self.state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload), encoding="utf-8")
            os.replace(tmp, self.state_path)
        except OSError as exc:
            logger.warning(f"Cannot persist geo presence state: {exc}")

    def record(self, client_id: str, ip: Optional[str]) -> None:
        """Update a client's last-seen time and (if known) coarse location."""
        if not client_id:
            return
        # All non-global addresses (localhost, LAN, containers) resolve to
        # the coordinator host's own public location.
        key = "" if _is_private(ip or "") else str(ip)
        with self._lock:
            entry = self._clients.setdefault(client_id, {})
            entry["last_seen"] = time.time()
            entry["ip_key"] = key
            cached = self._ip_cache.get(key)
            if cached and "lat" in cached:
                self._apply_location(client_id, entry, cached)
            elif self.lookup_enabled and self._should_lookup(key):
                self._pending_ips.add(key)
                threading.Thread(
                    target=self._resolve, args=(key,), daemon=True
                ).start()
            self._persist()

    def _should_lookup(self, key: str) -> bool:
        if key in self._pending_ips:
            return False
        cached = self._ip_cache.get(key)
        if cached is None:
            return True
        if "lat" in cached:
            return False
        # Retry failed lookups at most every 10 minutes.
        return time.time() - cached.get("failed_at", 0) > 600

    def _apply_location(
        self, client_id: str, entry: Dict[str, Any], geo: Dict[str, Any]
    ) -> None:
        h = _fnv1a(client_id)
        jitter_lat = (((h >> 4) % 80) - 40) / 100.0  # up to ±0.4°
        jitter_lng = (((h >> 12) % 80) - 40) / 100.0
        entry["lat"] = round(geo["lat"] + jitter_lat, 3)
        entry["lng"] = round(geo["lng"] + jitter_lng, 3)
        entry["city"] = geo.get("city")
        entry["country"] = geo.get("country")

    def _resolve(self, key: str) -> None:
        url = f"http://ip-api.com/json/{key}?fields={_LOOKUP_FIELDS}"
        result: Optional[Dict[str, Any]] = None
        try:
            with urllib.request.urlopen(url, timeout=_LOOKUP_TIMEOUT_SECONDS) as res:
                data = json.loads(res.read().decode("utf-8"))
            if data.get("status") == "success":
                result = {
                    # City-level coarseness before per-client jitter.
                    "lat": round(float(data["lat"]), 1),
                    "lng": round(float(data["lon"]), 1),
                    "city": data.get("city"),
                    "country": data.get("country"),
                }
        except Exception as exc:  # network errors must never break serving
            logger.info(f"Geo lookup failed: {exc}")
        with self._lock:
            self._pending_ips.discard(key)
            if result is None:
                self._ip_cache[key] = {"failed_at": time.time()}
                return
            self._ip_cache[key] = result
            for client_id, entry in self._clients.items():
                if entry.get("ip_key") == key:
                    self._apply_location(client_id, entry, result)
            self._persist(force=True)

    def snapshot(
        self,
        online_window_seconds: int = 300,
        max_age_seconds: int = 86_400,
    ) -> List[Dict[str, Any]]:
        """Anonymized located nodes seen recently; no IDs or IPs exposed."""
        if self.repo is not None:
            try:
                with self._lock:
                    # Merge SQL SoT; keep local ip_key only in memory
                    remote = self.repo.list_all()
                    for cid, entry in remote.items():
                        local = self._clients.get(cid, {})
                        merged = dict(entry)
                        if "ip_key" in local:
                            merged["ip_key"] = local["ip_key"]
                        self._clients[cid] = merged
            except Exception as exc:
                logger.warning(f"Cannot refresh geo from SQL: {exc}")
        now = time.time()
        nodes: List[Dict[str, Any]] = []
        with self._lock:
            for entry in self._clients.values():
                last_seen = entry.get("last_seen", 0)
                if now - last_seen > max_age_seconds:
                    continue
                if "lat" not in entry or "lng" not in entry:
                    continue
                nodes.append(
                    {
                        "lat": entry["lat"],
                        "lng": entry["lng"],
                        "city": entry.get("city"),
                        "country": entry.get("country"),
                        "last_seen": last_seen,
                        "online": now - last_seen <= online_window_seconds,
                    }
                )
        nodes.sort(key=lambda n: n["last_seen"], reverse=True)
        return nodes


_geo_presence: Optional[GeoPresence] = None


def get_geo_presence(repo=None) -> GeoPresence:
    global _geo_presence
    if _geo_presence is None:
        _geo_presence = GeoPresence(repo=repo)
    elif repo is not None and _geo_presence.repo is None:
        _geo_presence.repo = repo
        _geo_presence._load()
    return _geo_presence

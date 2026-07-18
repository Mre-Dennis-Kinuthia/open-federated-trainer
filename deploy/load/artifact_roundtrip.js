import http from "k6/http";
import { check, sleep } from "k6";

const BASE = __ENV.BASE_URL || "http://127.0.0.1:18080";
const PROTO = { "X-Protocol-Version": "2.0", "Content-Type": "application/json" };

export const options = {
  vus: 2,
  duration: "15s",
  thresholds: {
    http_req_failed: ["rate<0.05"],
  },
};

export function setup() {
  const name = `k6-${Date.now()}`;
  const res = http.post(
    `${BASE}/v2/node/register`,
    JSON.stringify({ client_name: name, protocol_version: "2.0" }),
    { headers: PROTO }
  );
  check(res, { "register 200": (r) => r.status === 200 });
  const body = res.json();
  return { client_id: body.client_id || name, api_key: body.api_key };
}

export default function (data) {
  const url =
    `${BASE}/v2/artifacts?client_id=${encodeURIComponent(data.client_id)}` +
    `&artifact_type=weight_delta`;
  const blob = JSON.stringify({ hello: "fed-compute-artifact", n: 42, t: Date.now() });
  const put = http.post(url, blob, {
    headers: {
      "Content-Type": "application/octet-stream",
      "X-Protocol-Version": "2.0",
      "X-Api-Key": data.api_key,
    },
  });
  const okPut = check(put, {
    "artifact upload ok": (r) => r.status === 200 || r.status === 201,
  });
  if (!okPut) {
    sleep(0.5);
    return;
  }
  let hash = null;
  try {
    const j = put.json();
    hash = (j.manifest && j.manifest.content_hash) || j.content_hash;
  } catch (e) {
    hash = null;
  }
  if (hash) {
    const get = http.get(
      `${BASE}/v2/artifacts/${hash}?client_id=${encodeURIComponent(data.client_id)}`,
      {
        headers: {
          "X-Protocol-Version": "2.0",
          "X-Api-Key": data.api_key,
        },
      }
    );
    check(get, { "artifact download 200": (r) => r.status === 200 });
  }
  sleep(0.3);
}

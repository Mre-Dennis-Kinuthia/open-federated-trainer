"""
General volunteer/edge worker for non-training jobs.

Polls the coordinator job queue for inference / label / compute work units.
Training FL rounds remain on client.py; run this alongside or instead.

Usage:
  WORK_MODES=inference,compute,label python worker.py
  JOB_TYPES=compute python worker.py
"""

from __future__ import annotations

import os
import sys
import threading
import time
import uuid

from config import config
from api import (
    register_client,
    claim_job,
    extend_job_lease,
    submit_job_result,
    CoordinatorAPIError,
    CoordinatorConnectionError,
)
from jobs import run_job
from security import get_api_key, save_api_key
from utils.logger import setup_client_logger, log_event

logger = setup_client_logger()


def _job_types() -> str:
    raw = os.getenv("JOB_TYPES") or config.WORK_MODES
    if raw.strip().lower() in ("all", "*"):
        return "inference,label,compute"
    # Strip train if present — this worker is non-training
    parts = [p.strip() for p in raw.split(",") if p.strip() and p.strip() != "train"]
    return ",".join(parts) or "inference,label,compute"


def _lease_heartbeat(
    stop: threading.Event,
    job_id: str,
    client_id: str,
    api_key: str,
    interval: float,
) -> None:
    while not stop.wait(interval):
        try:
            extend_job_lease(job_id, client_id, api_key=api_key)
        except Exception as exc:
            print(f"[{client_id}] Lease heartbeat failed for {job_id}: {exc}")


def main() -> None:
    print("=" * 60)
    print("fed-compute general job worker")
    print("=" * 60)
    print(f"Coordinator: {config.COORDINATOR_URL}")
    types = _job_types()
    print(f"Job types:   {types}")
    print("=" * 60)

    client_name = config.CLIENT_NAME or f"worker-{uuid.uuid4().hex[:8]}"
    print(f"Client Name: {client_name}")

    try:
        client_id, api_key = register_client(client_name)
        save_api_key(api_key)
        print(f"Registered as {client_id}")
    except (CoordinatorAPIError, CoordinatorConnectionError) as e:
        print(f"Registration failed: {e}")
        sys.exit(1)

    sleep_s = float(os.getenv("JOB_POLL_SECONDS", str(config.SLEEP_BETWEEN_ROUNDS)))
    heartbeat_s = float(os.getenv("JOB_LEASE_HEARTBEAT_SECONDS", "60"))

    while True:
        try:
            job = claim_job(client_id, api_key=api_key, types=types)
            if not job:
                print(f"[{client_id}] No jobs; sleeping {sleep_s}s")
                time.sleep(sleep_s)
                continue

            jid = job["job_id"]
            jtype = job["job_type"]
            print(f"[{client_id}] Claimed {jtype} job {jid}")
            log_event(logger, "job_claimed", client_id=client_id, extra_fields={
                "job_id": jid,
                "job_type": jtype,
            })

            stop = threading.Event()
            hb = threading.Thread(
                target=_lease_heartbeat,
                args=(stop, jid, client_id, api_key, heartbeat_s),
                daemon=True,
            )
            hb.start()
            try:
                result = run_job(job, client_id)
                submit_job_result(jid, client_id, result, api_key=api_key, success=True)
                print(f"[{client_id}] Completed job {jid}")
                log_event(logger, "job_completed", client_id=client_id, extra_fields={
                    "job_id": jid,
                    "job_type": jtype,
                })
            except Exception as e:
                print(f"[{client_id}] Job {jid} failed: {e}")
                try:
                    submit_job_result(
                        jid,
                        client_id,
                        {"error": str(e)},
                        api_key=api_key,
                        success=False,
                        error=str(e),
                    )
                except Exception as submit_err:
                    print(f"[{client_id}] Could not report failure: {submit_err}")
            finally:
                stop.set()

        except KeyboardInterrupt:
            print("\nShutting down worker")
            break
        except CoordinatorConnectionError as e:
            print(f"Coordinator unavailable: {e}")
            time.sleep(config.RETRY_DELAY)
        except CoordinatorAPIError as e:
            print(f"API error: {e}")
            time.sleep(sleep_s)
        except Exception as e:
            print(f"Unexpected error: {e}")
            time.sleep(sleep_s)


if __name__ == "__main__":
    main()

"""Legacy local capacity and endpoint checks, with co-location allowed.

Explicit --min-* thresholds are optional operator requirements. Runtime placement
uses measured per-stage contracts via shard.deployment, rather than these totals.
Repeated public addresses are a route diagnostic, not evidence of a broken route.
"""
import argparse
import os
import shutil
import socket
import sys
import time
from typing import Dict, List, Optional, Tuple


def get_total_ram_gb() -> float:
    """Read total system RAM in GiB across Linux/Windows."""
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = float(line.split()[1])
                        return kb / (1024.0 * 1024.0)
        except Exception:
            pass
    try:
        import psutil
        return psutil.virtual_memory().total / (1024.0 ** 3)
    except Exception:
        # Fallback to sysconf if available
        try:
            return (os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')) / (1024.0 ** 3)
        except Exception:
            return 0.0  # Unknown capacity never becomes a passing invented measurement.


def get_disk_free_gb(path: str = ".") -> float:
    """Read free disk space in GiB for the target directory."""
    try:
        target = path if os.path.exists(path) else "/"
        total, used, free = shutil.disk_usage(target)
        return free / (1024.0 ** 3)
    except Exception:
        return 0.0


def measure_tcp_rtt_ms(host: str, port: int, timeout: float = 3.0) -> Optional[float]:
    """Measure single TCP handshake RTT in milliseconds."""
    t0 = time.perf_counter()
    s = None
    try:
        s = socket.create_connection((host, port), timeout=timeout)
        t1 = time.perf_counter()
        return (t1 - t0) * 1000.0
    except Exception:
        return None
    finally:
        if s is not None:
            s.close()


def detect_hairpin_hazard(endpoints: List[str]) -> List[str]:
    """Detect if multiple remote stages share identical external IP (NAT hairpin hazard)."""
    ip_map: Dict[str, List[int]] = {}
    hazards = []
    for idx, ep in enumerate(endpoints):
        if not ep or ep == "none":
            continue
        host = ep.split(":")[0].strip()
        if host in ("127.0.0.1", "localhost", "0.0.0.0"):
            continue
        ip_map.setdefault(host, []).append(idx)

    for host, stages in ip_map.items():
        if len(stages) > 1:
            hazards.append(
                f"NAT Hairpin Hazard diagnostic: endpoint address {host!r} is shared by stages {stages}. "
                f"Verify each actual route; repeated addresses alone do not imply failure or one physical host."
            )
    return hazards


def run_preflight_checks(
    work_dir: str = "/root",
    min_ram_gb: float = 0.0,
    min_disk_gb: float = 0.0,
    endpoints: Optional[List[str]] = None,
    enforce: bool = True,
) -> Dict:
    """Execute all preflight gate assertions. Returns dictionary of findings."""
    import math
    if any(isinstance(x, bool) or not math.isfinite(x) or x < 0 for x in (min_ram_gb, min_disk_gb)):
        raise ValueError("resource thresholds must be finite nonnegative GiB")
    reasons, warnings = [], []
    
    # 1. RAM check
    actual_ram = get_total_ram_gb()
    # Allow a small 5% buffer for kernel reservation (e.g. 64GB box reporting 62.4GB)
    if actual_ram < (min_ram_gb * 0.95):
        reasons.append(
            f"Insufficient Host RAM: detected {actual_ram:.1f}GB, requires >= {min_ram_gb:.1f}GB. "
            f"Host expert caching and pinned memory will fail or cause OS swap death."
        )

    # 2. Disk space check
    actual_disk = get_disk_free_gb(work_dir)
    if actual_disk < (min_disk_gb * 0.95):
        reasons.append(
            f"Insufficient Disk Space: directory {work_dir!r} has only {actual_disk:.1f}GB free, requires >= {min_disk_gb:.1f}GB. "
            f"Cannot store full FP4 model checkpoint and runtime caches."
        )

    # 3. Hairpin hazard detection
    if endpoints:
        hazards = detect_hairpin_hazard(endpoints)
        warnings.extend(hazards)

    # 4. Latency matrix (if endpoints provided and accessible)
    rtt_matrix = {}
    if endpoints:
        for ep in endpoints:
            if not ep or ep == "none":
                continue
            host, sep, port_str = ep.rpartition(":")
            try:
                port = int(port_str)
                if not sep or not host or not 1 <= port <= 65535:
                    raise ValueError("invalid endpoint")
                rtt = measure_tcp_rtt_ms(host.strip("[]"), port, timeout=1.0)
                rtt_matrix[ep] = rtt
                if rtt is None:
                    reasons.append(f"Unreachable endpoint: {ep}")
                elif rtt > 250.0:
                    warnings.append(f"High handshake latency: {ep} measured {rtt:.1f}ms; price the actual route in placement")
            except ValueError:
                reasons.append(f"Malformed endpoint: {ep}")

    passed = len(reasons) == 0
    result = {
        "ok": passed,
        "ram_gb": round(actual_ram, 1),
        "disk_free_gb": round(actual_disk, 1),
        "rtt_matrix": rtt_matrix,
        "reasons": reasons,
        "warnings": warnings,
        "scope": "local totals and TCP reachability; not measured runtime placement or ring readiness",
    }

    if not passed and enforce:
        err_msg = "\n".join(f"  [X] {r}" for r in reasons)
        raise RuntimeError(
            f"\n[PREFLIGHT REJECTED] Cluster node failed preflight admission gate:\n{err_msg}\n"
            f"Refusing to form ring to save unnecessary hourly cloud spend!\n"
        )

    if passed:
        print(f"[preflight] PASS: local thresholds RAM={actual_ram:.1f}GiB, Disk={actual_disk:.1f}GiB. Use measured deployment validation for runtime admission.", flush=True)

    return result


def main():
    parser = argparse.ArgumentParser(description="FlyAI cluster node preflight admission gate.")
    parser.add_argument("--dir", default="/root", help="Working directory to check disk capacity")
    parser.add_argument("--min-ram", type=float, default=0.0, help="Optional minimum total RAM in GiB")
    parser.add_argument("--min-disk", type=float, default=0.0, help="Optional minimum free disk in GiB")
    parser.add_argument("--endpoints", default="", help="Comma-separated stage endpoints (host:port)")
    parser.add_argument("--no-enforce", action="store_true", help="Print findings without exiting non-zero")
    args = parser.parse_args()

    eps = [x.strip() for x in args.endpoints.split(",") if x.strip()] if args.endpoints else None
    try:
        res = run_preflight_checks(
            work_dir=args.dir,
            min_ram_gb=args.min_ram,
            min_disk_gb=args.min_disk,
            endpoints=eps,
            enforce=not args.no_enforce,
        )
        print(f"Preflight status: OK={res['ok']}")
    except RuntimeError as e:
        print(e, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

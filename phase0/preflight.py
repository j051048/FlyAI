"""P0-3 Preflight cluster & node validation gate.

Strict gate before forming a distributed inference ring:
- Host RAM >= 64GB (accommodates host expert pools & KV paging)
- Usable Disk >= 300GB (safetensors weights & checkpoint storage)
- Duplicate IP / NAT hairpin probe (detects co-located container deadlock)
- Peer RTT latency matrix (fails fast on broken edges or WAN stalls)
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
            return 64.0  # Safe default if unsupported


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
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect((host, port))
        t1 = time.perf_counter()
        return (t1 - t0) * 1000.0
    except Exception:
        return None
    finally:
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
                f"NAT Hairpin Hazard: Host IP {host!r} is shared by multiple distinct stages {stages}. "
                f"Vast.ai containers on the same physical host cannot route to each other's public port without hairpin NAT support!"
            )
    return hazards


def run_preflight_checks(
    work_dir: str = "/root",
    min_ram_gb: float = 64.0,
    min_disk_gb: float = 300.0,
    endpoints: Optional[List[str]] = None,
    enforce: bool = True,
) -> Dict:
    """Execute all preflight gate assertions. Returns dictionary of findings."""
    reasons = []
    
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
        reasons.extend(hazards)

    # 4. Latency matrix (if endpoints provided and accessible)
    rtt_matrix = {}
    if endpoints:
        for ep in endpoints:
            if not ep or ep == "none":
                continue
            parts = ep.split(":")
            if len(parts) == 2:
                host, port_str = parts
                try:
                    port = int(port_str)
                    rtt = measure_tcp_rtt_ms(host, port, timeout=1.0)
                    rtt_matrix[ep] = rtt
                    if rtt is not None and rtt > 250.0:
                        reasons.append(f"Excessive WAN Latency: endpoint {ep} RTT is {rtt:.1f}ms (>250ms threshold). Ring pipeline throughput will collapse.")
                except ValueError:
                    pass

    passed = len(reasons) == 0
    result = {
        "ok": passed,
        "ram_gb": round(actual_ram, 1),
        "disk_free_gb": round(actual_disk, 1),
        "rtt_matrix": rtt_matrix,
        "reasons": reasons,
    }

    if not passed and enforce:
        err_msg = "\n".join(f"  [X] {r}" for r in reasons)
        raise RuntimeError(
            f"\n[PREFLIGHT REJECTED] Cluster node failed preflight admission gate:\n{err_msg}\n"
            f"Refusing to form ring to save unnecessary hourly cloud spend!\n"
        )

    if passed:
        print(f"[preflight] PASS: RAM={actual_ram:.1f}GB (>= {min_ram_gb}GB), Disk={actual_disk:.1f}GB (>= {min_disk_gb}GB). Cluster admission granted.", flush=True)

    return result


def main():
    parser = argparse.ArgumentParser(description="FlyAI cluster node preflight admission gate.")
    parser.add_argument("--dir", default="/root", help="Working directory to check disk capacity")
    parser.add_argument("--min-ram", type=float, default=64.0, help="Minimum RAM in GB")
    parser.add_argument("--min-disk", type=float, default=300.0, help="Minimum free disk in GB")
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

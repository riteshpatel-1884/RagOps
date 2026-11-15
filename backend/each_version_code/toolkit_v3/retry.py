import re
import time

MAX_RETRIES = 8
LONG_WAIT_LIMIT = 120   # seconds; if Groq asks for longer, stop cleanly so you can resume later


def parse_wait_seconds(msg: str):
    """Parse 'try again in 14m22.5s' / '5.5s' / '340ms' from Groq's message."""
    m = re.search(r"try again in ([0-9hms.]+)", msg)
    if not m:
        return None
    total = 0.0
    for value, unit in re.findall(r"([\d.]+)(ms|h|m|s)", m.group(1)):
        v = float(value)
        total += {"h": v * 3600, "m": v * 60, "s": v, "ms": v / 1000}[unit]
    return total


def with_retry(fn, label):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn()
        except Exception as e:
            msg = str(e)

            if "429" in msg or "rate_limit" in msg.lower():
                wait = parse_wait_seconds(msg)
                if wait is not None and wait > LONG_WAIT_LIMIT:
                    print(f"\nLong rate limit: Groq asks to wait ~{wait / 60:.0f} min.")
                    print("Progress is saved. Re-run the same command after that time.")
                    raise SystemExit(1)
                wait = (wait + 3) if wait is not None else 20
                reason = "rate limit"
            elif "Connection error" in msg or "ReadError" in msg or "timed out" in msg.lower():
                wait = 10
                reason = "connection error"
            else:
                raise  # unknown error, do not hide it

            print(f"  [{label}] {reason}, retry {attempt}/{MAX_RETRIES} in {wait:.0f}s")
            time.sleep(wait)

    raise RuntimeError(f"{label} failed after {MAX_RETRIES} retries")

#!/usr/bin/env python3
"""Daily check of NASA data product endpoint availability.

Probes known DAAC endpoints with HEAD requests and updates
docs/data-status.md and docs/data-status.png with results.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import requests

# ── Configuration ────────────────────────────────────────────────────────────

PRODUCTS = {
    "MODIS (LP DAAC)": "https://e4ftl01.cr.usgs.gov/MOLT/",
    "ERA5 (CDS)": "https://cds.climate.copernicus.eu/api",
    "MERRA-2 (GES DISC)": "https://goldsmr5.gesdisc.eosdis.nasa.gov/opendap/",
    "SMAP (NSIDC)": "https://n5eil01u.ecs.nsidc.org/SMAP/",
    "GPM (GES DISC)": "https://gpm1.gesdisc.eosdis.nasa.gov/opendap/",
}

DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"
STATUS_MD = DOCS_DIR / "data-status.md"
HISTORY_JSON = DOCS_DIR / ".data-status-history.json"
STATUS_PNG = DOCS_DIR / "data-status.png"

TIMEOUT = 15  # seconds per HEAD request


# ── Helpers ──────────────────────────────────────────────────────────────────

def check_endpoint(url: str) -> str:
    """Return 'Available' or 'Unavailable' based on a HEAD request."""
    try:
        resp = requests.head(url, timeout=TIMEOUT, allow_redirects=True)
        return "Available" if resp.status_code < 400 else "Unavailable"
    except requests.RequestException:
        return "Unavailable"


def load_history() -> dict:
    """Load the JSON history file, or return an empty structure."""
    if HISTORY_JSON.exists():
        try:
            return json.loads(HISTORY_JSON.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_history(history: dict) -> None:
    HISTORY_JSON.write_text(json.dumps(history, indent=2) + "\n")


def status_changed(history: dict, current: dict[str, str]) -> bool:
    """Return True if any product status differs from the last recorded check."""
    last = history.get("latest", {})
    return any(current.get(p) != last.get(p) for p in PRODUCTS)


# ── Main logic ───────────────────────────────────────────────────────────────

def run_checks() -> dict[str, str]:
    """Probe every endpoint and return {product: status}."""
    results: dict[str, str] = {}
    for name, url in PRODUCTS.items():
        results[name] = check_endpoint(url)
        print(f"  {name}: {results[name]}")
    return results


def update_markdown(results: dict[str, str], now: datetime) -> None:
    """Write docs/data-status.md with the current status table."""
    timestamp = now.strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# NASA Data Product Availability Status\n",
        f"*Last checked: {timestamp}*\n",
        "",
        "| Product | Status | Last Checked |",
        "| ------- | ------ | ------------ |",
    ]
    for product, status in results.items():
        icon = "\u2705" if status == "Available" else "\u274c"
        lines.append(f"| {product} | {icon} {status} | {timestamp} |")

    lines += [
        "",
        "## Availability Timeline",
        "",
        "![Availability Timeline](data-status.png)",
        "",
        "---",
        "*Generated automatically by the daily-update workflow.*",
        "",
    ]
    STATUS_MD.write_text("\n".join(lines))


def update_chart(history: dict) -> None:
    """Generate a timeline chart from the history records."""
    records = history.get("records", [])
    if not records:
        return

    products = list(PRODUCTS.keys())
    dates = [datetime.fromisoformat(r["date"]) for r in records]

    fig, ax = plt.subplots(figsize=(10, 4))

    for i, product in enumerate(products):
        statuses = [
            1 if r.get("statuses", {}).get(product) == "Available" else 0
            for r in records
        ]
        colors = ["#2ecc71" if s else "#e74c3c" for s in statuses]
        ax.scatter(dates, [i] * len(dates), c=colors, s=60, zorder=3)
        if len(dates) > 1:
            ax.plot(dates, [i] * len(dates), color="#bdc3c7", linewidth=1, zorder=1)

    ax.set_yticks(range(len(products)))
    ax.set_yticklabels(products, fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate(rotation=30)
    ax.set_title("NASA Data Product Availability Timeline", fontsize=12)
    ax.set_xlabel("Date")
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ecc71",
               markersize=8, label="Available"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#e74c3c",
               markersize=8, label="Unavailable"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

    plt.tight_layout()
    fig.savefig(STATUS_PNG, dpi=150)
    plt.close(fig)
    print(f"  Chart saved to {STATUS_PNG}")


def main() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc)
    print("Checking NASA data product endpoints ...")
    results = run_checks()

    history = load_history()

    if not history or status_changed(history, results):
        print("Status change detected -- updating docs.")
    else:
        print("No status change -- updating timestamp only.")

    # Always record today's check in history
    history.setdefault("records", [])
    history["records"].append({
        "date": now.isoformat(),
        "statuses": results,
    })
    # Keep at most 90 days of history
    history["records"] = history["records"][-90:]
    history["latest"] = results
    save_history(history)

    update_markdown(results, now)
    update_chart(history)
    print("Done.")


if __name__ == "__main__":
    main()

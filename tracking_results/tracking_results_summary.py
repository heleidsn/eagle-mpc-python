from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List


def parse_stats_file(file_path: Path) -> Dict[str, object]:
    """Parse one *_stats.txt file into a dict."""
    stats: Dict[str, object] = {}
    for line in file_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()

        if key in {"samples"}:
            stats[key] = int(value)
        elif key in {
            "duration_s",
            "pos_rmse_m",
            "pos_mean_m",
            "pos_max_m",
            "att_rmse_deg",
            "att_mean_deg",
            "att_max_deg",
            "solve_ms_mean",
            "solve_ms_p95",
            "solve_ms_max",
        }:
            stats[key] = float(value)
        else:
            stats[key] = value

    # Keep source file for traceability
    stats["stats_file"] = file_path.name
    return stats


def build_summary_rows(stats_dir: Path) -> List[Dict[str, object]]:
    """Read all *_stats.txt under stats_dir and return summary rows."""
    stats_files = sorted(stats_dir.glob("*_stats.txt"))
    if not stats_files:
        raise FileNotFoundError(f"No *_stats.txt found in: {stats_dir}")

    rows: List[Dict[str, object]] = [parse_stats_file(path) for path in stats_files]
    rows.sort(key=lambda row: str(row.get("trajectory_name", "")))
    return rows


def render_text_table(rows: List[Dict[str, object]], columns: List[str]) -> str:
    """Render a plain text table."""
    if not rows:
        return "(empty table)"

    def value_to_text(value: object) -> str:
        if isinstance(value, float):
            return f"{value:.6f}"
        return str(value)

    matrix: List[List[str]] = []
    matrix.append(columns)
    for row in rows:
        matrix.append([value_to_text(row.get(col, "")) for col in columns])

    widths = [max(len(matrix[r][c]) for r in range(len(matrix))) for c in range(len(columns))]

    def format_line(items: List[str]) -> str:
        return " | ".join(item.ljust(widths[i]) for i, item in enumerate(items))

    separator = "-+-".join("-" * w for w in widths)
    lines = [format_line(matrix[0]), separator]
    for row in matrix[1:]:
        lines.append(format_line(row))
    return "\n".join(lines)


def save_csv(rows: List[Dict[str, object]], columns: List[str], output_path: Path) -> None:
    """Save rows to CSV."""
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def main() -> None:
    current_dir = Path(__file__).resolve().parent
    summary_rows = build_summary_rows(current_dir)

    columns = [
        "trajectory_name",
        "controller_mode",
        "robot_name",
        "odom_source",
        "samples",
        "duration_s",
        "pos_rmse_m",
        "pos_mean_m",
        "pos_max_m",
        "att_rmse_deg",
        "att_mean_deg",
        "att_max_deg",
        "solve_ms_mean",
        "solve_ms_p95",
        "solve_ms_max",
        "stats_file",
    ]

    # Print a readable table in terminal
    print("\n=== Tracking Stats Summary ===")
    print(render_text_table(summary_rows, columns))

    # Save summary for later analysis
    output_csv = current_dir / "stats_summary.csv"
    save_csv(summary_rows, columns, output_csv)
    print(f"\nSaved summary CSV: {output_csv}")


if __name__ == "__main__":
    main()

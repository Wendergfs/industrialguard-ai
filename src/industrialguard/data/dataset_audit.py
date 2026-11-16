"""Orchestrate read-only audits and write machine-readable reports."""

from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path

from .archive_reader import ArchiveReader
from .discovery import discover_images
from .statistics import summarize
from .validation import validate_records


def read_config(path: str | Path) -> dict:
    """Load JSON-compatible YAML, expanding required ``${ENV_VAR}`` values."""
    raw = Path(path).read_text(encoding="utf-8")

    def expand(match: re.Match[str]) -> str:
        variable = match.group(1)
        value = os.environ.get(variable)
        if not value:
            raise ValueError(f"Required environment variable {variable} is not set")
        return json.dumps(value)[1:-1]

    return json.loads(re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", expand, raw))


def audit_dataset(name: str, spec: dict, output: Path) -> tuple[list, dict, dict]:
    archive = Path(spec["archive"])
    if not archive.is_file():
        raise FileNotFoundError(f"Dataset archive does not exist: {archive}")
    reader = ArchiveReader(archive, spec["format"])
    # ZIP has a dedicated CRC test. For TAR.XZ, names() traverses the complete
    # compressed stream and validation then reads each image payload in order.
    if spec["format"] == "zip":
        bad_member = reader.test_integrity()
        if bad_member is not None:
            raise OSError(f"Archive integrity check failed at member: {bad_member}")
    names = reader.names()
    records, discovery_warnings = discover_images(
        name, names, spec["categories"], spec.get("expected_root")
    )
    validation = validate_records(reader, records, names)
    validation["warnings"] = discovery_warnings + validation["warnings"]
    summary = summarize(name, records, names, validation, spec["categories"])
    summary.update(
        {
            "archive_path": str(archive),
            "archive_size_bytes": archive.stat().st_size,
            "file_member_count": len(names),
            "directory_member_count": reader.directory_member_count,
            "archive_member_count": reader.archive_member_count,
            "integrity_check": "passed",
        }
    )
    write_inventory(output / f"{name}_inventory.csv", records)
    (output / f"{name}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return records, summary, validation


def write_inventory(path: Path, records) -> None:
    fields = (
        list(records[0].as_dict())
        if records
        else [
            "dataset",
            "category",
            "image_path",
            "image_type",
            "split",
            "defect_type",
            "mask_path",
            "width",
            "height",
            "image_format",
            "mask_width",
            "mask_height",
            "mask_empty",
            "defective_pixels_percent",
        ]
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(record.as_dict() for record in records)


def write_validation_report(path: Path, reports: dict[str, dict]) -> None:
    payload = {
        name: {
            "errors": result["errors"],
            "warnings": result["warnings"],
            "counts": result["counts"],
            "mask_member_count": result["mask_member_count"],
        }
        for name, result in reports.items()
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_markdown(
    path: Path, summaries: dict[str, dict], validations: dict[str, dict]
) -> None:
    lines = [
        "# IndustrialGuard AI dataset audit",
        "",
        "Read-only inspection of the original archives; no extraction or source changes were performed.",
        "",
        "MPDD is the primary mechanical-component benchmark. MVTec AD is a complementary broader industrial anomaly-detection benchmark.",
        "",
        "## Dataset summaries",
        "",
        "| Dataset | Archive size (bytes) | Members | Files | Categories | Images | Normal | Anomalous | Masks | Missing masks |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, s in summaries.items():
        lines.append(
            f"| {name} | {s['archive_size_bytes']} | {s['archive_member_count']} | {s['file_member_count']} | {s['category_count']} | {s['image_count']} | {s['normal_images']} | {s['anomalous_images']} | {s['ground_truth_mask_members']} | {s['missing_masks']} |"
        )
    for name, s in summaries.items():
        v = validations[name]
        lines.extend(
            [
                "",
                f"## {name}",
                "",
                f"Archive: `{s['archive_path']}`",
                "",
                f"Expected categories found: {s['category_count']}/{len(s['expected_categories'])}.",
                "",
                f"Validation errors: {len(v['errors'])}; warnings: {len(v['warnings'])}.",
                "",
                "### Categories",
                "",
                "| Category | Images | Normal | Anomalous | Masks | Defect types |",
                "|---|---:|---:|---:|---:|---|",
            ]
        )
        for category, c in s["by_category"].items():
            lines.append(
                f"| {category} | {c['images']} | {c['normal']} | {c['anomalous']} | {c['masks']} | {', '.join(c['defect_types'])} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_audit(
    config_path: str | Path, output_dir: str | Path
) -> tuple[dict, dict, dict]:
    config = read_config(config_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summaries, validations, all_records = {}, {}, {}
    for name, spec in config["datasets"].items():
        print(f"Auditing {name} archive: {spec['archive']}", flush=True)
        records, summary, validation = audit_dataset(name, spec, output)
        summaries[name], validations[name], all_records[name] = (
            summary,
            validation,
            records,
        )
    write_validation_report(output / "validation_report.json", validations)
    write_markdown(output / "dataset_audit_report.md", summaries, validations)
    return summaries, validations, all_records

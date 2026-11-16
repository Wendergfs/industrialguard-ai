"""Dataset counts and descriptive statistics derived from inventory records."""

from __future__ import annotations

from collections import Counter

from .discovery import ImageRecord


def summarize(
    dataset: str,
    records: list[ImageRecord],
    names: list[str],
    validation: dict,
    expected_categories: list[str],
) -> dict:
    categories = sorted({r.category for r in records})
    by_category = {}
    for category in expected_categories:
        subset = [r for r in records if r.category == category]
        by_category[category] = {
            "images": len(subset),
            "normal": sum(r.image_type == "normal" for r in subset),
            "anomalous": sum(r.image_type == "anomalous" for r in subset),
            "masks": sum(bool(r.mask_path) for r in subset),
            "missing_masks": sum(
                r.image_type == "anomalous" and not r.mask_path for r in subset
            ),
            "defect_types": dict(
                Counter(r.defect_type for r in subset if r.defect_type)
            ),
        }
    return {
        "dataset": dataset,
        "archive_member_count": len(names),
        "categories_found": categories,
        "expected_categories": expected_categories,
        "missing_categories": sorted(set(expected_categories) - set(categories)),
        "category_count": len(categories),
        "image_count": len(records),
        "normal_images": sum(r.image_type == "normal" for r in records),
        "anomalous_images": sum(r.image_type == "anomalous" for r in records),
        "ground_truth_mask_members": validation["mask_member_count"],
        "matched_masks": sum(bool(r.mask_path) for r in records),
        "missing_masks": sum(
            r.image_type == "anomalous" and not r.mask_path for r in records
        ),
        "defect_types": dict(Counter(r.defect_type for r in records if r.defect_type)),
        "image_resolutions": validation["counts"]["dimensions"],
        "validation_counts": validation["counts"],
        "by_category": by_category,
    }

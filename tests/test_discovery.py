from industrialguard.data.discovery import discover_images


def test_discovers_category_roles_and_matching_masks():
    names = [
        "MPDD/bracket_black/train/good/normal.png",
        "MPDD/bracket_black/test/good/good.png",
        "MPDD/bracket_black/test/hole/001.png",
        "MPDD/bracket_black/ground_truth/hole/001_mask.png",
    ]
    records, warnings = discover_images("mpdd", names, ["bracket_black"], "MPDD")
    assert not warnings
    assert [(r.image_type, r.split) for r in records] == [
        ("normal", "train"),
        ("normal", "test"),
        ("anomalous", "test"),
    ]
    assert records[-1].mask_path == names[-1]


def test_reports_missing_category_and_mask():
    records, warnings = discover_images(
        "mvtec", ["bottle/test/crack/a.png"], ["bottle", "cable"]
    )
    assert records[0].mask_path == ""
    assert "Expected category missing: cable" in warnings

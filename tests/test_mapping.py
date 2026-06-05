"""T1/T2 매핑 순수 함수 검증 (합성 fixture)."""
from src.augment_bbq import bbq_unknown_idx, cells_needed, map_bbq_row, pair_image_for_bbq
from src.map_sbbench import (
    canon_axis,
    derive_context_condition,
    detect_unknown_idx,
    map_sbbench_row,
)


def test_canon_axis():
    assert canon_axis("Age") == "Age"
    assert canon_axis("Race_ethnicity") == "Race_ethnicity"
    assert canon_axis("Disability") == "Disability_status"
    assert canon_axis("SES") == "SES"
    assert canon_axis("Race_x_gender") == "Intersectional"
    assert canon_axis("") == "Other"


def test_detect_unknown_idx(lexicon):
    assert detect_unknown_idx(["A", "B", "Cannot be determined"], lexicon) == 2
    assert detect_unknown_idx(["Unknown", "A", "B"], lexicon) == 0
    assert detect_unknown_idx(["A", "B", "C"], lexicon) is None


def test_derive_context_condition_explicit():
    amb, inf = derive_context_condition({"additional_metadata": {"context_condition": "ambig"}}, 2, 2)
    assert amb is True and inf is False
    amb, inf = derive_context_condition({"additional_metadata": {"context_condition": "disambig"}}, 1, 2)
    assert amb is False and inf is False


def test_derive_context_condition_heuristic():
    # 메타/ID 없음 → label==unknown_idx 추정
    amb, inf = derive_context_condition({"id": "x"}, 0, 0)
    assert amb is True and inf is True
    amb, inf = derive_context_condition({"id": "x"}, 1, 0)
    assert amb is False and inf is True


def test_map_sbbench_row_all(tiny_sbbench_rows, lexicon):
    recs = [map_sbbench_row(r, lexicon) for r in tiny_sbbench_rows]
    assert all(r is not None for r in recs)
    r0 = recs[0]
    assert r0["axis"] == "Age" and r0["ambig"] is True and r0["unknown_idx"] == 2
    assert r0["options"][r0["label"]] == "Cannot be determined"
    assert r0["source"] == "sb-bench" and r0["license"] == "CC-BY-NC-4.0"
    # disambig 행
    assert recs[1]["ambig"] is False and recs[1]["unknown_idx"] == 0
    # 휴리스틱 행: label==unknown_idx=0 → ambig 추정
    assert recs[2]["ambig"] is True and recs[2]["meta"]["ambig_inferred"] is True


def test_map_sbbench_row_rejects_no_unknown(lexicon):
    bad = {"category": "Age", "ans0": "A", "ans1": "B", "ans2": "C", "label": 0}
    assert map_sbbench_row(bad, lexicon) is None


def test_bbq_unknown_idx():
    ai = {"ans0": ["old"], "ans1": ["young"], "ans2": ["unknown"]}
    assert bbq_unknown_idx(ai, ["a", "b", "c"]) == 2
    assert bbq_unknown_idx({"ans0": ["x"]}, ["a", "b", "c"]) is None


def test_map_bbq_row(tiny_bbq_rows):
    recs = [map_bbq_row(r) for r in tiny_bbq_rows]
    assert all(r is not None for r in recs)
    assert recs[0]["axis"] == "Age" and recs[0]["ambig"] is True
    assert recs[0]["options"][recs[0]["unknown_idx"]] == "Not known"
    assert recs[1]["axis"] == "Nationality" and recs[1]["ambig"] is False
    assert recs[1]["license"] == "CC-BY-4.0"


def test_cells_needed():
    counts = {("Age", True, "neg"): 5, ("Age", False, "neg"): 1250}
    need = cells_needed(counts, 1250)
    assert need[("Age", True, "neg")] == 1245
    assert ("Age", False, "neg") not in need  # 충족 셀 제외


def test_pair_image_same_axis():
    import random
    sb = {"Age": [("./images/a1.jpg", "sb-bench", "CC-BY-NC-4.0")],
          "Religion": [("./images/r1.jpg", "sb-bench", "CC-BY-NC-4.0")]}
    # 동일 axis SB-Bench (외부 풀 없음) → 그 항목 반환 (ref, source, license)
    ref, src, lic = pair_image_for_bbq({"axis": "Age"}, {}, sb, random.Random(0))
    assert ref == "./images/a1.jpg" and src == "sb-bench" and lic == "CC-BY-NC-4.0"
    # 풀에 없는 axis → 전체 풀에서 선택
    assert pair_image_for_bbq({"axis": "Nationality"}, {}, sb, random.Random(0)) is not None
    # 빈 풀 → None
    assert pair_image_for_bbq({"axis": "Age"}, {}, {}, random.Random(0)) is None


def test_pair_image_prefers_external():
    """동일 axis에 외부(FairFace/MMBias)가 있으면 SB-Bench보다 우선 → 라이선스 정화."""
    import random
    ext = {"Age": [("./images/fairface_000001.jpg", "fairface", "CC-BY-4.0")]}
    sb = {"Age": [("./images/train_img_000001.jpg", "sb-bench", "CC-BY-NC-4.0")]}
    for seed in range(10):
        ref, src, lic = pair_image_for_bbq({"axis": "Age"}, ext, sb, random.Random(seed))
        assert src == "fairface" and lic == "CC-BY-4.0"
    # 외부에 없는 axis(SES)는 SB-Bench로 폴백
    _, src2, _ = pair_image_for_bbq({"axis": "SES"}, ext,
                                    {"SES": [("./images/x.jpg", "sb-bench", "CC-BY-NC-4.0")]},
                                    random.Random(0))
    assert src2 == "sb-bench"


def test_build_image_pool_tuples():
    from src.augment_bbq import build_image_pool
    recs = [
        {"axis": "Age", "image_ref": "./images/a.jpg",
         "image_source": "sb-bench", "image_license": "CC-BY-NC-4.0"},
        {"axis": "Age", "image_ref": None},                       # 이미지 없는 행 제외
        {"axis": "Religion", "image_ref": "./images/r.jpg"},      # source/license 미기록 → 기본값
    ]
    pool = build_image_pool(recs)
    assert pool["Age"] == [("./images/a.jpg", "sb-bench", "CC-BY-NC-4.0")]
    assert pool["Religion"] == [("./images/r.jpg", "sb-bench", "CC-BY-NC-4.0")]


def test_metadata_is_nc_logic():
    """텍스트·이미지 중 하나라도 NC면 is_nc=True. BBQ 텍스트+FairFace 이미지는 깨끗."""
    from src.metadata import make_metadata_row
    # BBQ(CC-BY) 텍스트 + FairFace(CC-BY) 이미지 → 깨끗
    clean = make_metadata_row(
        {"source": "bbq", "license": "CC-BY-4.0", "image_ref": "./images/fairface_0.jpg",
         "image_source": "fairface", "image_license": "CC-BY-4.0", "axis": "Age"}, "TRAIN_000000")
    assert clean["is_nc"] is False and clean["image_source"] == "fairface"
    # BBQ 텍스트 + SB-Bench(NC) 이미지 재사용 → NC
    nc = make_metadata_row(
        {"source": "bbq", "license": "CC-BY-4.0", "image_ref": "./images/train_img_0.jpg",
         "image_source": "sb-bench", "image_license": "CC-BY-NC-4.0", "axis": "SES"}, "TRAIN_000001")
    assert nc["is_nc"] is True
    # 하위호환: image_source 미기록 → 텍스트 출처로 폴백
    legacy = make_metadata_row(
        {"source": "sb-bench", "license": "CC-BY-NC-4.0", "image_ref": "./images/x.jpg",
         "axis": "Age"}, "TRAIN_000002")
    assert legacy["image_source"] == "sb-bench" and legacy["is_nc"] is True

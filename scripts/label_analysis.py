"""PHASE 3 - Label analysis for the 12 target abnormalities.

Reports per-label totals/prevalence, label-value domains, pairwise
co-occurrence (joint positive counts and Jaccard), studies-per-positive-count,
and basic report-text facts. Writes outputs/reports/label_analysis.json.
"""

from __future__ import annotations

import sys

import pandas as pd

from common import PROJECT_ROOT, write_report

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from orthovision.data.tabular import LABELS, cooccurrence_matrix, label_statistics, load_train


def main() -> int:
    train = load_train(PROJECT_ROOT / "data" / "raw" / "train.csv")

    stats = label_statistics(train, LABELS)
    report: dict = {
        "n_studies": int(len(train)),
        "n_studies_with_any_label": int(train[list(LABELS)].notna().any(axis=1).sum()),
        "n_studies_fully_unlabeled": int(train[list(LABELS)].isna().all(axis=1).sum()),
        "per_label": [s.as_dict() for s in stats],
    }

    labeled = train[train[list(LABELS)].notna().any(axis=1)]
    report["labeled_subset_size"] = int(len(labeled))
    report["label_value_domains"] = {
        label: sorted(float(v) for v in labeled[label].dropna().unique()) for label in LABELS
    }

    joint = cooccurrence_matrix(labeled, LABELS)
    pos_counts = {s.label: s.positive for s in stats}
    jaccard = pd.DataFrame(index=list(LABELS), columns=list(LABELS), dtype=float)
    for a in LABELS:
        for b in LABELS:
            union = pos_counts[a] + pos_counts[b] - joint.loc[a, b]
            jaccard.loc[a, b] = float(joint.loc[a, b] / union) if union else float("nan")
    report["cooccurrence_joint_positive_counts"] = {
        f"{a} | {b}": int(joint.loc[a, b])
        for i, a in enumerate(LABELS)
        for b in LABELS[i + 1 :]
        if joint.loc[a, b] > 0
    }
    report["cooccurrence_jaccard_nonzero"] = {
        f"{a} | {b}": round(float(jaccard.loc[a, b]), 4)
        for i, a in enumerate(LABELS)
        for b in LABELS[i + 1 :]
        if jaccard.loc[a, b] == jaccard.loc[a, b] and jaccard.loc[a, b] > 0
    }
    offdiag = [
        (a, b, float(jaccard.loc[a, b]))
        for i, a in enumerate(LABELS)
        for b in LABELS[i + 1 :]
    ]
    offdiag.sort(key=lambda t: -t[2])
    report["top_cooccurring_pairs_jaccard"] = [{"pair": [a, b], "jaccard": v} for a, b, v in offdiag[:8]]

    pos_per_study = (labeled[list(LABELS)] > 0).sum(axis=1)
    report["positives_per_labeled_study"] = {
        str(int(k)): int(v) for k, v in pos_per_study.value_counts().sort_index().items()
    }

    reports_text = train["Report"].dropna()
    lengths = reports_text.str.len()
    langs_hint = {
        "has_spanish_markers": int(reports_text.str.contains(r"(?i)rodilla|hallazgos|t[eé]cnica", regex=True).sum()),
        "has_german_markers": int(reports_text.str.contains(r"(?i)knie|links|rechts", regex=True).sum()),
        "has_english_markers": int(reports_text.str.contains(r"(?i)knee|findings|impression", regex=True).sum()),
    }
    report["report_text"] = {
        "non_null_reports": int(reports_text.shape[0]),
        "length_chars": {
            "min": int(lengths.min()),
            "median": float(lengths.median()),
            "max": int(lengths.max()),
        },
        "language_marker_counts_case_insensitive": langs_hint,
        "note": "markers overlap across languages; not a language ID",
    }

    ranked = sorted(stats, key=lambda s: -s.prevalence_labeled)
    report["ranked_by_prevalence"] = [
        {"label": s.label, "prevalence_labeled": round(s.prevalence_labeled, 4), "positive": s.positive}
        for s in ranked
    ]

    write_report("label_analysis.json", report)

    print(f"studies={report['n_studies']} labeled={report['n_studies_with_any_label']} unlabeled={report['n_studies_fully_unlabeled']}")
    for s in stats:
        print(f"  {s.label:18s} pos={s.positive:3d} neg={s.negative:3d} missing={s.missing:5d} prev={s.prevalence_labeled:.3f} values={sorted(set(s.positive_values))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


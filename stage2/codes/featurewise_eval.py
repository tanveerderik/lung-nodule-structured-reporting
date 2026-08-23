#!/usr/bin/env python3
import argparse, csv, json, re
from collections import defaultdict
from scipy.optimize import linear_sum_assignment

from date_normalization import canonical_date_key, is_date_field
from nodule_scoring import is_null
from prediction_normalization import normalize_prediction

KEY_FIELDS = ["Series ID", "Image ID", "Nodule ID"]
MATCH_FIELDS = [
    "Series ID", "Image ID", "Nodule ID", "Lobe", "Segment", "Type",
    "Long Axis (mm)", "Short Axis (mm)", "Average Diameter (mm)"
]

def norm(v, field_name=None):
    if is_null(v):
        return None
    if is_date_field(field_name):
        date_key = canonical_date_key(v)
        if date_key is not None:
            return date_key
    if isinstance(v, str):
        s = re.sub(r"\s+", " ", v.strip().lower())
        try:
            return round(float(s), 3)
        except Exception:
            return s
    if isinstance(v, (int, float)):
        return round(float(v), 3)
    return v

def values_equal(a, b, tol=0.01, field_name=None):
    a, b = norm(a, field_name), norm(b, field_name)
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if isinstance(a, float) and isinstance(b, float):
        return abs(a - b) <= tol
    return a == b

def safe_load(x):
    if isinstance(x, dict):
        return x
    if not isinstance(x, str):
        return {}
    try:
        return json.loads(x)
    except Exception:
        m = re.search(r"(\{.*\})", x, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                return {}
    return {}

def clean_key(k):
    return re.sub(r"\s+", " ", str(k).strip())

def canonicalize(obj):
    if isinstance(obj, dict):
        return {clean_key(k): canonicalize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [canonicalize(x) for x in obj]
    return obj

def unwrap_lungs_pleura(obj):
    obj = canonicalize(obj)
    if isinstance(obj, dict) and "Lungs Pleura" in obj and isinstance(obj["Lungs Pleura"], dict):
        return obj["Lungs Pleura"]
    return obj

def schema_leaves(schema, prefix=""):
    out = []
    schema = canonicalize(schema)

    if isinstance(schema, dict):
        if "data_type" in schema and "values" in schema:
            out.append(prefix)
        else:
            for k, v in schema.items():
                newp = f"{prefix}.{k}" if prefix else k
                out.extend(schema_leaves(v, newp))

    elif isinstance(schema, list) and schema:
        # Aggregate list fields under the same path, no [0], [1], etc.
        out.extend(schema_leaves(schema[0], prefix))

    return out

def get_path(obj, path):
    cur = obj
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur

def record_key(d):
    vals = []
    for k in KEY_FIELDS:
        v = norm(d.get(k))
        if v is not None:
            vals.append((k, v))
    return tuple(vals) if vals else None

def sim_record(g, p):
    denom = 0
    hit = 0
    for f in MATCH_FIELDS:
        if f in g and not is_null(g.get(f)):
            denom += 1
            if f in p and values_equal(g.get(f), p.get(f), field_name=f):
                hit += 1
    return hit / denom if denom else 0.0

def match_records(gold_list, pred_list, min_sim=0.20):
    matches, used_g, used_p = [], set(), set()

    pred_by_key = defaultdict(list)
    for j, p in enumerate(pred_list):
        k = record_key(p)
        if k:
            pred_by_key[k].append(j)

    for i, g in enumerate(gold_list):
        k = record_key(g)
        if k and k in pred_by_key:
            for j in pred_by_key[k]:
                if j not in used_p:
                    matches.append((i, j))
                    used_g.add(i)
                    used_p.add(j)
                    break

    rg = [i for i in range(len(gold_list)) if i not in used_g]
    rp = [j for j in range(len(pred_list)) if j not in used_p]

    if rg and rp:
        sims = [[sim_record(gold_list[i], pred_list[j]) for j in rp] for i in rg]
        rows, cols = linear_sum_assignment([[-x for x in row] for row in sims])
        for r, c in zip(rows, cols):
            if sims[r][c] >= min_sim:
                i, j = rg[r], rp[c]
                matches.append((i, j))
                used_g.add(i)
                used_p.add(j)

    return matches, set(range(len(gold_list))) - used_g, set(range(len(pred_list))) - used_p

def add_metric(stats, feature, gold_present=0, tp=0, fp=0, fn=0):
    s = stats[feature]
    s["gold_appearances"] += gold_present
    s["TP"] += tp
    s["FP"] += fp
    s["FN"] += fn

def score_scalar(stats, feature, g, p):
    g_present = not is_null(g)
    p_present = not is_null(p)

    if g_present:
        add_metric(stats, feature, gold_present=1)

    if g_present and p_present:
        if values_equal(g, p, field_name=feature):
            add_metric(stats, feature, tp=1)
        else:
            add_metric(stats, feature, fp=1, fn=1)
    elif g_present and not p_present:
        add_metric(stats, feature, fn=1)
    elif p_present and not g_present:
        add_metric(stats, feature, fp=1)

def score_dict_leaves(stats, base_feature, gold_d, pred_d, child_fields):
    gold_d = gold_d if isinstance(gold_d, dict) else {}
    pred_d = pred_d if isinstance(pred_d, dict) else {}

    for child in child_fields:
        feature = f"{base_feature}.{child}"
        score_scalar(stats, feature, gold_d.get(child), pred_d.get(child))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_cases", required=True)
    ap.add_argument("--schema_file", required=True)
    ap.add_argument("--output_csv", default="featurewise_f1.csv")
    ap.add_argument("--output_json", default="featurewise_f1.json")
    ap.add_argument("--min_similarity", type=float, default=0.20)
    args = ap.parse_args()

    cases = json.load(open(args.eval_cases, "r", encoding="utf-8"))
    schema = unwrap_lungs_pleura(json.load(open(args.schema_file, "r", encoding="utf-8")))

    # Your eval cases are flattened to the nodule/recommendation section,
    # while the schema stores them under "Nodule Findings".
    if isinstance(schema, dict) and "Nodule Findings" in schema:
        schema = schema["Nodule Findings"]

    leaves = schema_leaves(schema)

    # Find list-object schema paths, e.g. Nodule Findings.Nodules
    list_paths = {}
    def find_lists(x, prefix=""):
        x = canonicalize(x)
        if isinstance(x, dict):
            for k, v in x.items():
                p = f"{prefix}.{k}" if prefix else k
                find_lists(v, p)
        elif isinstance(x, list) and x and isinstance(x[0], dict):
            child_fields = [
                k for k, v in canonicalize(x[0]).items()
                if isinstance(v, dict) and "data_type" in v
            ]
            list_paths[prefix] = child_fields

    find_lists(schema)

    stats = defaultdict(lambda: {"gold_appearances": 0, "TP": 0, "FP": 0, "FN": 0})

    # Materialize the complete schema-defined feature universe before scoring.
    # Previously, defaultdict entries were created only after a TP/FP/FN event.
    # Consequently, a feature absent from all ground truths and never predicted
    # by one decoding arm disappeared from that arm's output, while the same
    # feature appeared in another arm if it produced even one false positive.
    # Those are not unpaired samples: the missing arm has a legitimate all-zero
    # row. Emitting every schema feature keeps comparisons paired and prevents
    # selective omission from biasing macro-F1 and feature-delta summaries.
    schema_features = set(leaves)
    for list_path, child_fields in list_paths.items():
        schema_features.update(
            f"{list_path}.{child}" for child in child_fields
        )
    for feature in sorted(schema_features):
        stats[feature]

    for case in cases:
        gold_source = (
            case.get("Ground Truth")
            if "Ground Truth" in case
            else safe_load(case.get("Raw Ground Truth"))
        )
        pred_source = (
            case.get("Prediction")
            if "Prediction" in case
            else safe_load(case.get("Raw Prediction"))
        )
        # Apply the same semantic omission policy to both sides without ever
        # falling back from a legitimate normalized empty prediction to raw text.
        gold = normalize_prediction(unwrap_lungs_pleura(gold_source))
        pred = normalize_prediction(unwrap_lungs_pleura(pred_source))

        handled_list_children = set()

        for list_path, child_fields in list_paths.items():
            gold_list = get_path(gold, list_path)
            pred_list = get_path(pred, list_path)

            gold_list = gold_list if isinstance(gold_list, list) else []
            pred_list = pred_list if isinstance(pred_list, list) else []

            matches, unmatched_g, unmatched_p = match_records(
                gold_list, pred_list, min_sim=args.min_similarity
            )

            for i, j in matches:
                score_dict_leaves(stats, list_path, gold_list[i], pred_list[j], child_fields)

            for i in unmatched_g:
                score_dict_leaves(stats, list_path, gold_list[i], {}, child_fields)

            for j in unmatched_p:
                score_dict_leaves(stats, list_path, {}, pred_list[j], child_fields)

            for child in child_fields:
                handled_list_children.add(f"{list_path}.{child}")

        for feature in leaves:
            if feature in handled_list_children:
                continue
            score_scalar(stats, feature, get_path(gold, feature), get_path(pred, feature))

    rows = []
    for feature in sorted(stats):
        s = stats[feature]
        tp, fp, fn = s["TP"], s["FP"], s["FN"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0

        rows.append({
            "feature": feature,
            "gold_appearances": s["gold_appearances"],
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.output_json}")

if __name__ == "__main__":
    main()

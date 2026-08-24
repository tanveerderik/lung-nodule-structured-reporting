from __future__ import annotations

import csv
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

DATA_DIR = Path(__file__).resolve().parents[2] / "datasets"
DEFAULT_FILES = [DATA_DIR / 'train_nodule.json', DATA_DIR / 'val_nodule.json']
OUT_DIR = Path('id_audit_outputs')
KEY_TOKEN_CHARS = r'A-Za-z0-9_'


def norm_key(s: str) -> str:
    return re.sub(r'[^a-z0-9]+', '', str(s).lower())


def collapse_ws(s: str) -> str:
    return re.sub(r'\s+', ' ', s).strip()


def strip_outer_quotes(s: str) -> str:
    s = s.strip()
    pairs = [('"', '"'), ("'", "'"), ('`', '`'), ('“', '”'), ('‘', '’')]
    changed = True
    while changed and len(s) >= 2:
        changed = False
        for a, b in pairs:
            if s.startswith(a) and s.endswith(b):
                s = s[1:-1].strip()
                changed = True
    return s


def strip_outer_punct(s: str) -> str:
    return s.strip().strip('[](){}<>.,;:')


def clean_id_text(v: Any) -> str:
    if v is None:
        return ''
    s = str(v)
    s = collapse_ws(s)
    s = strip_outer_quotes(s)
    s = strip_outer_punct(s)
    return s.strip()


def id_format(s: str) -> str:
    if not s:
        return 'empty'
    if re.fullmatch(r'\d+', s):
        return 'numeric'
    if re.fullmatch(r'[A-Za-z0-9]+', s) and re.search(r'[A-Za-z]', s) and re.search(r'\d', s):
        return 'alphanumeric'
    return 'other'


def digits_only(s: str) -> str:
    return ''.join(re.findall(r'\d+', s))


def strip_leading_zeros(num: str) -> str:
    if not num:
        return ''
    if not re.fullmatch(r'\d+', num):
        return num
    return str(int(num)) if any(ch != '0' for ch in num) else '0'


def strip_common_id_prefixes(s: str) -> str:
    s = clean_id_text(s)
    s = re.sub(r'^(?:imageid|seriesid|image|img|series|ser|id)[\s:_#-]*', '', s, flags=re.I)
    return s.strip()


def compact_separators(s: str) -> str:
    return re.sub(r'[\s\-_:/#]+', '', s)


def extract_first_balanced_json_fragment(text: str) -> Optional[str]:
    start = None
    stack: List[str] = []
    in_str = False
    escape = False
    quote = ''
    for i, ch in enumerate(text):
        if start is None:
            if ch == '{':
                start = i
                stack = ['}']
            elif ch == '[':
                start = i
                stack = [']']
            continue

        if in_str:
            if escape:
                escape = False
            elif ch == '\\':
                escape = True
            elif ch == quote:
                in_str = False
            continue

        if ch == '"':
            in_str = True
            quote = ch
            continue

        if ch == '{':
            stack.append('}')
        elif ch == '[':
            stack.append(']')
        elif stack and ch == stack[-1]:
            stack.pop()
            if not stack and start is not None:
                return text[start:i + 1]
    return None


def extract_json_candidates(raw: str) -> List[str]:
    raw = raw.strip()
    candidates: List[str] = []

    tag_match = re.search(r'<json>(.*?)</json>', raw, flags=re.I | re.S)
    if tag_match:
        candidates.append(tag_match.group(1).strip())

    fence_match = re.search(r'```(?:json)?\s*(.*?)```', raw, flags=re.I | re.S)
    if fence_match:
        candidates.append(fence_match.group(1).strip())

    candidates.append(raw)

    fragment = extract_first_balanced_json_fragment(raw)
    if fragment:
        candidates.append(fragment)

    out: List[str] = []
    seen = set()
    for cand in candidates:
        if cand and cand not in seen:
            seen.add(cand)
            out.append(cand)
    return out


def parse_json_payload(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if value is None:
        raise ValueError('output is None')
    errors = []
    for candidate in extract_json_candidates(str(value)):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as e:
            errors.append(f'{e.__class__.__name__}: {e}')
    raise ValueError('Unable to parse output as JSON. ' + ' | '.join(errors[:4]))


def iter_nodules(obj: Any, path: str = 'root') -> Iterator[Tuple[dict, str]]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            child = f'{path}.{k}'
            if norm_key(k) == 'nodules' and isinstance(v, list):
                for i, item in enumerate(v):
                    if isinstance(item, dict):
                        yield item, f'{child}[{i}]'
            yield from iter_nodules(v, child)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            yield from iter_nodules(item, f'{path}[{i}]')


def token_pattern(literal: str) -> str:
    return rf'(?<![{KEY_TOKEN_CHARS}]){re.escape(literal)}(?![{KEY_TOKEN_CHARS}])'


def token_search(text: str, literal: str, *, ignore_case: bool = False) -> Optional[re.Match]:
    if not literal:
        return None
    flags = re.I if ignore_case else 0
    return re.search(token_pattern(literal), text, flags=flags)


def compact_token_search(text: str, compact_literal: str, *, ignore_case: bool = False) -> bool:
    if not compact_literal:
        return False
    compact_text = compact_separators(text)
    flags = re.I if ignore_case else 0
    return re.search(token_pattern(compact_literal), compact_text, flags=flags) is not None


@dataclass
class MatchResult:
    strict_token: bool = False
    strict_ws_token: bool = False
    relaxed_casefold: bool = False
    relaxed_compact: bool = False
    relaxed_prefix_stripped: bool = False
    relaxed_digits: bool = False
    relaxed_digits_no_zeros: bool = False
    matched_by: str = 'none'
    span_start: Optional[int] = None
    span_end: Optional[int] = None

    @property
    def relaxed_any(self) -> bool:
        return any([
            self.strict_ws_token,
            self.relaxed_casefold,
            self.relaxed_compact,
            self.relaxed_prefix_stripped,
            self.relaxed_digits,
            self.relaxed_digits_no_zeros,
        ])

    @property
    def any_present(self) -> bool:
        return self.strict_token or self.relaxed_any


def match_id_in_text(text: str, raw_value: Any) -> MatchResult:
    txt = str(text or '')
    txt_ws = collapse_ws(txt)
    val = clean_id_text(raw_value)
    result = MatchResult()
    if not val:
        return result

    m = token_search(txt, val)
    if m:
        result.strict_token = True
        result.matched_by = 'strict_token'
        result.span_start, result.span_end = m.span()
        return result

    m = token_search(txt_ws, collapse_ws(val))
    if m:
        result.strict_ws_token = True
        result.matched_by = 'strict_ws_token'
        result.span_start, result.span_end = m.span()
        return result

    m = token_search(txt, val, ignore_case=True)
    if m:
        result.relaxed_casefold = True
        result.matched_by = 'relaxed_casefold'
        result.span_start, result.span_end = m.span()
        return result

    compact_val = compact_separators(val)
    if compact_val and compact_token_search(txt, compact_val, ignore_case=True):
        result.relaxed_compact = True
        result.matched_by = 'relaxed_compact'
        return result

    stripped = strip_common_id_prefixes(val)
    if stripped and stripped != val:
        m = token_search(txt, stripped, ignore_case=True) or token_search(txt_ws, collapse_ws(stripped), ignore_case=True)
        if m:
            result.relaxed_prefix_stripped = True
            result.matched_by = 'relaxed_prefix_stripped'
            result.span_start, result.span_end = m.span()
            return result

    digs = digits_only(val)
    if digs:
        m = token_search(txt, digs)
        if m:
            result.relaxed_digits = True
            result.matched_by = 'relaxed_digits'
            result.span_start, result.span_end = m.span()
            return result

        digs_nz = strip_leading_zeros(digs)
        if digs_nz and digs_nz != digs:
            m = token_search(txt, digs_nz)
            if m:
                result.relaxed_digits_no_zeros = True
                result.matched_by = 'relaxed_digits_no_zeros'
                result.span_start, result.span_end = m.span()
                return result

    return result


def context_excerpt(text: str, start: Optional[int], end: Optional[int], width: int = 120) -> str:
    txt = str(text or '')
    if start is None or end is None:
        return collapse_ws(txt)[: 2 * width]
    lo = max(0, start - width)
    hi = min(len(txt), end + width)
    return collapse_ws(txt[lo:hi])


def parse_dataset_file(path: Path) -> List[dict]:
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, list):
        raise ValueError(f'{path} does not contain a top-level JSON list')
    return data


def process_file(path: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    records: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    examples = parse_dataset_file(path)
    for ex_idx, ex in enumerate(examples):
        input_text = ex.get('input', '')
        output_raw = ex.get('output', '')
        try:
            payload = parse_json_payload(output_raw)
        except Exception as e:
            errors.append({'file': path.name, 'example_index': ex_idx, 'error': str(e)})
            continue

        nodule_ordinal = 0
        for nodule, nodule_path in iter_nodules(payload):
            for feature in ('Image ID', 'Series ID'):
                raw_value = nodule.get(feature)
                clean_value = clean_id_text(raw_value)
                if not clean_value:
                    continue
                match = match_id_in_text(input_text, clean_value)
                records.append({
                    'file': path.name,
                    'example_index': ex_idx,
                    'nodule_index': nodule_ordinal,
                    'nodule_path': nodule_path,
                    'feature': feature,
                    'raw_value': str(raw_value),
                    'clean_value': clean_value,
                    'id_format': id_format(clean_value),
                    'strict_present': int(match.strict_token),
                    'strict_ws_present': int(match.strict_ws_token),
                    'relaxed_present': int(match.relaxed_any),
                    'any_present': int(match.any_present),
                    'matched_by': match.matched_by,
                    'context': context_excerpt(input_text, match.span_start, match.span_end),
                })
            nodule_ordinal += 1
    return records, errors


def summarize_by_file_feature(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
    for r in records:
        key = (r['file'], r['feature'])
        c = grouped[key]
        c['total_occurrences'] += 1
        c['strict_present_n'] += r['strict_present']
        c['strict_ws_present_n'] += r['strict_ws_present']
        c['relaxed_present_n'] += r['relaxed_present']
        c['any_present_n'] += r['any_present']
        c[f"format_{r['id_format']}"] += 1
        c[f"matched_by_{r['matched_by']}"] += 1

    rows: List[Dict[str, Any]] = []
    for (file_name, feature), c in sorted(grouped.items()):
        total = c['total_occurrences'] or 1
        rows.append({
            'file': file_name,
            'feature': feature,
            'total_occurrences': c['total_occurrences'],
            'strict_present_n': c['strict_present_n'],
            'strict_present_pct': round(100.0 * c['strict_present_n'] / total, 3),
            'strict_plus_ws_n': c['strict_present_n'] + c['strict_ws_present_n'],
            'strict_plus_ws_pct': round(100.0 * (c['strict_present_n'] + c['strict_ws_present_n']) / total, 3),
            'relaxed_only_n': c['relaxed_present_n'],
            'relaxed_only_pct': round(100.0 * c['relaxed_present_n'] / total, 3),
            'any_present_n': c['any_present_n'],
            'any_present_pct': round(100.0 * c['any_present_n'] / total, 3),
            'numeric_n': c['format_numeric'],
            'numeric_pct': round(100.0 * c['format_numeric'] / total, 3),
            'alphanumeric_n': c['format_alphanumeric'],
            'alphanumeric_pct': round(100.0 * c['format_alphanumeric'] / total, 3),
            'other_n': c['format_other'],
            'other_pct': round(100.0 * c['format_other'] / total, 3),
            'matched_by_strict_token_n': c['matched_by_strict_token'],
            'matched_by_strict_ws_token_n': c['matched_by_strict_ws_token'],
            'matched_by_relaxed_casefold_n': c['matched_by_relaxed_casefold'],
            'matched_by_relaxed_compact_n': c['matched_by_relaxed_compact'],
            'matched_by_relaxed_prefix_stripped_n': c['matched_by_relaxed_prefix_stripped'],
            'matched_by_relaxed_digits_n': c['matched_by_relaxed_digits'],
            'matched_by_relaxed_digits_no_zeros_n': c['matched_by_relaxed_digits_no_zeros'],
            'matched_by_none_n': c['matched_by_none'],
        })
    return rows


def summarize_by_file(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, Counter] = defaultdict(Counter)
    for r in records:
        c = grouped[r['file']]
        c['total_occurrences'] += 1
        c['strict_present_n'] += r['strict_present']
        c['any_present_n'] += r['any_present']
        c[f"feature_{r['feature']}"] += 1

    rows: List[Dict[str, Any]] = []
    for file_name, c in sorted(grouped.items()):
        total = c['total_occurrences'] or 1
        rows.append({
            'file': file_name,
            'total_occurrences': c['total_occurrences'],
            'strict_present_n': c['strict_present_n'],
            'strict_present_pct': round(100.0 * c['strict_present_n'] / total, 3),
            'any_present_n': c['any_present_n'],
            'any_present_pct': round(100.0 * c['any_present_n'] / total, 3),
            'image_id_occurrences': c['feature_Image ID'],
            'series_id_occurrences': c['feature_Series ID'],
        })
    return rows


def format_distribution(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str], Counter] = defaultdict(Counter)
    for r in records:
        key = (r['file'], r['feature'], r['id_format'])
        grouped[key]['count'] += 1
    totals: Dict[Tuple[str, str], int] = defaultdict(int)
    for r in records:
        totals[(r['file'], r['feature'])] += 1
    rows: List[Dict[str, Any]] = []
    for (file_name, feature, fmt), c in sorted(grouped.items()):
        total = totals[(file_name, feature)] or 1
        rows.append({
            'file': file_name,
            'feature': feature,
            'id_format': fmt,
            'count': c['count'],
            'pct_within_feature': round(100.0 * c['count'] / total, 3),
        })
    return rows


def sample_mismatches(records: List[Dict[str, Any]], k: int = 20, seed: int = 7) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        if not r['any_present']:
            grouped[(r['file'], r['feature'])].append(r)
    rng = random.Random(seed)
    keys = sorted(grouped)
    for rows in grouped.values():
        rng.shuffle(rows)
    out: List[Dict[str, Any]] = []
    while len(out) < k and any(grouped.values()):
        progressed = False
        for key in keys:
            if grouped[key] and len(out) < k:
                out.append(grouped[key].pop())
                progressed = True
        if not progressed:
            break
    return out[:k]


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    fieldnames = list(rows[0].keys())
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_csv_block(title: str, rows: List[Dict[str, Any]]) -> None:
    print(f'=== {title} ===')
    if not rows:
        print('no rows')
        print()
        return
    fieldnames = list(rows[0].keys())
    print(','.join(fieldnames))
    for row in rows:
        vals = []
        for key in fieldnames:
            value = str(row.get(key, ''))
            if any(ch in value for ch in [',', '"', '\n']):
                value = '"' + value.replace('"', '""') + '"'
            vals.append(value)
        print(','.join(vals))
    print()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_records: List[Dict[str, Any]] = []
    all_errors: List[Dict[str, Any]] = []

    for path in DEFAULT_FILES:
        if not path.exists():
            all_errors.append({'file': str(path), 'example_index': '', 'error': 'File not found'})
            continue
        records, errors = process_file(path)
        all_records.extend(records)
        all_errors.extend(errors)

    ff_summary = summarize_by_file_feature(all_records)
    file_summary = summarize_by_file(all_records)
    fmt_summary = format_distribution(all_records)
    mismatch_rows = sample_mismatches(all_records, k=20, seed=7)

    write_csv(OUT_DIR / 'id_presence_records.csv', all_records)
    write_csv(OUT_DIR / 'id_presence_summary_by_file_feature.csv', ff_summary)
    write_csv(OUT_DIR / 'id_presence_summary_by_file.csv', file_summary)
    write_csv(OUT_DIR / 'id_presence_format_distribution.csv', fmt_summary)
    write_csv(OUT_DIR / 'id_presence_mismatch_samples.csv', mismatch_rows)
    write_csv(OUT_DIR / 'id_presence_errors.csv', all_errors)

    print_csv_block('id_presence_summary_by_file_feature.csv', ff_summary)
    print_csv_block('id_presence_summary_by_file.csv', file_summary)
    print_csv_block('id_presence_format_distribution.csv', fmt_summary)
    print_csv_block('id_presence_mismatch_samples.csv', mismatch_rows)
    if all_errors:
        print_csv_block('id_presence_errors.csv', all_errors)


if __name__ == '__main__':
    main()

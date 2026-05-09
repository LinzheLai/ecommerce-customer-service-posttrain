from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
CLEAN_DIR = BASE_DIR / "data" / "cleaned"
BACKUP_ROOT = BASE_DIR / "outputs" / "data_backups"

SOURCE_RAW = CLEAN_DIR / "train_clean_top23000_raw.txt"
SOURCE_PREVIEW = CLEAN_DIR / "train_clean_top23000_preview.jsonl"
MANUAL_FILE = CLEAN_DIR / "train_clean_top5000_round2_manual_dpo.json"

CANONICAL_RAW = RAW_DIR / "train_dpo_repaired.txt"
CANONICAL_REPORT = RAW_DIR / "train_dpo_repaired_report.jsonl"
CANONICAL_STATS = RAW_DIR / "train_dpo_repaired_stats.json"

TARGETS = (
    {
        "name": "top5000",
        "top_k": 5000,
        "clean_raw": CLEAN_DIR / "train_clean_top5000_raw.txt",
        "positive_only": CLEAN_DIR / "train_clean_top5000_positive_only.txt",
        "preview": CLEAN_DIR / "train_clean_top5000_preview.jsonl",
        "stats": CLEAN_DIR / "train_clean_top5000_stats.json",
        "dpo_train": BASE_DIR / "data" / "processed_5000" / "taobao_dpo_train.json",
    },
    {
        "name": "top10000",
        "top_k": 10000,
        "clean_raw": CLEAN_DIR / "train_clean_top10000_raw.txt",
        "positive_only": CLEAN_DIR / "train_clean_top10000_positive_only.txt",
        "preview": CLEAN_DIR / "train_clean_top10000_preview.jsonl",
        "stats": CLEAN_DIR / "train_clean_top10000_stats.json",
        "dpo_train": BASE_DIR / "data" / "processed_10000" / "taobao_dpo_train.json",
    },
    {
        "name": "top23000",
        "top_k": 23000,
        "clean_raw": CLEAN_DIR / "train_clean_top23000_raw.txt",
        "positive_only": CLEAN_DIR / "train_clean_top23000_positive_only.txt",
        "preview": CLEAN_DIR / "train_clean_top23000_preview.jsonl",
        "stats": CLEAN_DIR / "train_clean_top23000_stats.json",
        "dpo_train": BASE_DIR / "data" / "processed_23000" / "taobao_dpo_train.json",
    },
)

PREFERENCE_INSTRUCTION = "你是电商平台客服偏好优化数据构建器，请偏向更准确、相关、礼貌且可执行的回复。"

PHONE_RE = re.compile(r"\b1\d{10}\b")
LONG_ID_RE = re.compile(r"\b\d{8,}\b")
NUM_RE = re.compile(r"\d+")
SPACE_RE = re.compile(r"[ \t]+")
MULTI_NL_RE = re.compile(r"\n{3,}")
ROLE_PREFIX_RE = re.compile(r"^(?:用户|买家|顾客|客服|商家|店主|掌柜|小二|卖家)\s*[:：]\s*")
CLAUSE_SPLIT_RE = re.compile(r"[，。！？；!?,;]+")
PLACEHOLDER_RE = re.compile(r"(?:一个\s*)?<ID>|<ID>|<PHONE>", re.I)
LOW_INFO_REPLIES = {
    "嗯",
    "嗯嗯",
    "好的",
    "好的呢",
    "好的哦",
    "是的",
    "在的",
    "您好",
    "你好",
    "亲",
    "稍等",
    "谢谢",
    "不客气",
}
EXTERNAL_CONTACT_HINTS = {
    "微信",
    "加微",
    "加我",
    "二维码",
    "扫码",
    "qq",
    "QQ",
    "旺旺外",
}
RISKY_POLICY_HINTS = {
    "直接退款",
    "申请退款就行",
    "直接给您退款",
    "直接补发",
    "给您补发",
    "马上发出",
    "今天发出",
    "明天到",
    "一定到",
    "改价",
    "补差价",
    "联系我",
}
TIME_PROMISE_HINTS = {
    "今天发",
    "今天就发",
    "明天发",
    "明天到",
    "马上发",
    "立刻发",
    "48小时",
    "72小时",
    "当天发货",
    "次日发货",
}
STORE_SPECIFIC_HINTS = {
    "店铺首页",
    "收藏店铺",
    "关注店铺",
    "联系客服改价",
    "本店默认",
    "默认韵达",
    "默认邮政",
    "领券",
    "优惠券",
    "满减",
    "包邮",
    "活动价",
}
ADDRESS_HINTS = {"地址", "收货", "改地址", "新地址", "核对地址"}
REFUND_HINTS = {"退款", "退货", "售后", "补发", "拒收", "拦截", "退回"}
SHIPPING_HINTS = {"发货", "快递", "物流", "邮政", "韵达", "顺丰", "申通", "中通", "EMS", "百世", "汇通"}
PROMO_HINTS = {"优惠", "活动", "赠品", "礼品", "改价", "差价", "券", "满减", "包邮"}
PRODUCT_HINTS = {"尺寸", "材质", "纯棉", "无纺布", "质量", "日期", "生产日期", "规格", "口味"}
ACTION_HINTS = ADDRESS_HINTS | REFUND_HINTS | SHIPPING_HINTS | PROMO_HINTS | PRODUCT_HINTS | {"备注", "核对", "申请", "联系", "处理"}

NEGATIVE_TEMPLATES = {
    "address": [
        "您把新地址发我，我这边直接给您改一下",
        "地址我先帮您改好，您不用操作",
        "您直接把新地址发来，我这边马上给您改",
    ],
    "refund": [
        "可以直接申请退款哦",
        "我这边直接给您补发一份",
        "您不满意的话直接退货就可以",
    ],
    "shipping": [
        "今天就给您发出哦",
        "默认给您发顺丰哦",
        "明天一定给您送到哦",
    ],
    "promo": [
        "您先拍下，我这边给您改价",
        "拍下后联系我给您补差价",
        "可以加我微信给您单独优惠",
    ],
    "product": [
        "都差不多的哦",
        "您先拍下，不合适再退就行",
        "质量都一样的哦",
    ],
    "other": [
        "您先拍下，我这边给您处理",
        "这个都没问题的哦",
        "可以的，直接按您说的来就行",
    ],
}


@dataclass
class RepairGroup:
    context: List[str]
    chosen_original: str
    rejected_original: str
    chosen: str
    rejected: str
    category: str
    score: float
    source: str
    pos_line_no: Optional[int]
    neg_line_no: Optional[int]
    issues: List[str]


def clean_text(text: str) -> str:
    text = "" if text is None else str(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\ufeff", "")
    text = text.replace("\u3000", " ").replace("\xa0", " ")
    text = PHONE_RE.sub("<PHONE>", text)
    text = LONG_ID_RE.sub("<ID>", text)
    text = text.replace("\\", "")
    text = SPACE_RE.sub(" ", text)
    text = MULTI_NL_RE.sub("\n\n", text)
    lines = [ROLE_PREFIX_RE.sub("", line.strip()) for line in text.split("\n")]
    return "\n".join(line for line in lines if line).strip()


def compact(text: str) -> str:
    return clean_text(text).replace(" ", "")


def split_clauses(text: str) -> List[str]:
    seen = set()
    clauses: List[str] = []
    for clause in CLAUSE_SPLIT_RE.split(clean_text(text)):
        clause = clause.strip(" ，。！？；!?,;")
        if not clause:
            continue
        key = compact(clause)
        if key in seen:
            continue
        seen.add(key)
        clauses.append(clause)
    return clauses


def strip_reply_placeholders(text: str) -> str:
    text = clean_text(text)
    text = PLACEHOLDER_RE.sub("", text)
    text = re.sub(r"(?i)一个\s*ID", "一个账号", text)
    text = re.sub(r"(?i)不同\s*ID", "不同账号", text)
    text = re.sub(r"(?i)每个\s*ID", "每个账号", text)
    text = re.sub(r"(?i)\bID\b", "账号", text)
    text = re.sub(r"\s{2,}", " ", text).strip(" ，。！？；!?,;")
    text = text.replace("  ", " ").strip()
    return text


def token_keywords(text: str) -> List[str]:
    s = compact(text)
    hits: List[str] = []
    for word in sorted(ACTION_HINTS, key=len, reverse=True):
        if word and word in s:
            hits.append(word)
    return hits


def overlap_score(a: str, b: str) -> float:
    ta = set(token_keywords(a))
    tb = set(token_keywords(b))
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    return inter / max(1, min(len(ta), len(tb)))


def classify_category(last_user: str) -> str:
    s = compact(last_user)
    if any(word in s for word in ADDRESS_HINTS):
        return "address"
    if any(word in s for word in REFUND_HINTS):
        return "refund"
    if any(word in s for word in SHIPPING_HINTS):
        return "shipping"
    if any(word in s for word in PROMO_HINTS):
        return "promo"
    if any(word in s for word in PRODUCT_HINTS):
        return "product"
    return "other"


def contains_any(text: str, words: Iterable[str]) -> bool:
    s = compact(text)
    return any(word in s for word in words)


def is_low_info(text: str) -> bool:
    s = compact(text)
    if not s:
        return True
    if s in LOW_INFO_REPLIES:
        return True
    if len(s) <= 4:
        return True
    if len(s) <= 8 and not contains_any(s, ACTION_HINTS):
        return True
    return False


def is_bad_text(text: str) -> bool:
    s = compact(text)
    if not s or s in {"<ID>", "<PHONE>"}:
        return True
    if any(h in s for h in EXTERNAL_CONTACT_HINTS):
        return True
    return False


def select_positive_clauses(context: Sequence[str], response: str) -> Tuple[Optional[str], List[str], float]:
    last_user = context[-1]
    category = classify_category(last_user)
    issues: List[str] = []
    clauses = split_clauses(response)
    if not clauses:
        return None, ["empty_positive"], -10.0

    scored: List[Tuple[float, str]] = []
    for clause in clauses:
        clause_compact = compact(clause)
        score = 0.0
        if is_low_info(clause):
            score -= 5.0
        if any(h in clause_compact for h in EXTERNAL_CONTACT_HINTS):
            score -= 12.0
        if "<ID>" in clause_compact or "<PHONE>" in clause_compact:
            score -= 7.0
        if any(h in clause_compact for h in STORE_SPECIFIC_HINTS):
            score -= 3.5
        if any(h in clause_compact for h in TIME_PROMISE_HINTS) and not contains_any(last_user, TIME_PROMISE_HINTS | SHIPPING_HINTS):
            score -= 2.0
        if any(h in clause_compact for h in RISKY_POLICY_HINTS) and category not in {"refund", "shipping", "address"}:
            score -= 2.0
        score += overlap_score(last_user, clause) * 8.0
        if contains_any(clause, ACTION_HINTS):
            score += 2.0
        if category != "other" and contains_any(clause, {
            "address": ADDRESS_HINTS,
            "refund": REFUND_HINTS,
            "shipping": SHIPPING_HINTS,
            "promo": PROMO_HINTS,
            "product": PRODUCT_HINTS,
        }[category]):
            score += 2.0
        length = len(clause_compact)
        if 6 <= length <= 26:
            score += 2.0
        elif length <= 40:
            score += 1.0
        else:
            score -= 1.5
        scored.append((score, clause))

    scored.sort(key=lambda item: (item[0], -len(compact(item[1]))), reverse=True)
    kept: List[str] = []
    total_chars = 0
    for score, clause in scored:
        if score <= 0:
            continue
        clause_len = len(compact(clause))
        if total_chars and total_chars + clause_len > 34:
            continue
        kept.append(clause)
        total_chars += clause_len
        if len(kept) >= 2:
            break

    if not kept:
        best_score, best_clause = scored[0]
        if best_score < 0:
            return None, ["positive_not_salvageable"], best_score
        kept = [best_clause]

    cleaned = strip_reply_placeholders("，".join(kept))
    if cleaned != clean_text(response):
        issues.append("trimmed_positive")
    if any(h in compact(cleaned) for h in STORE_SPECIFIC_HINTS):
        issues.append("store_specific_positive")
    if any(h in compact(cleaned) for h in TIME_PROMISE_HINTS) and not contains_any(last_user, SHIPPING_HINTS):
        issues.append("time_promise_positive")
    if is_low_info(cleaned):
        return None, issues + ["positive_low_info"], -5.0
    if is_bad_text(cleaned):
        return None, issues + ["positive_bad_text"], -6.0
    return cleaned, issues, scored[0][0] + len(compact(cleaned)) / 10.0


def negative_needs_rewrite(last_user: str, rejected: str) -> bool:
    s = compact(rejected)
    if is_low_info(rejected):
        return True
    if not s:
        return True
    if "<ID>" in s or "<PHONE>" in s:
        return True
    if overlap_score(last_user, rejected) < 0.12 and not contains_any(rejected, EXTERNAL_CONTACT_HINTS | RISKY_POLICY_HINTS):
        return True
    if len(s) <= 6:
        return True
    if any(h in s for h in {"中关村", "满99包邮", "满159送", "优惠券", "地址哦"}) and not contains_any(last_user, PROMO_HINTS | ADDRESS_HINTS):
        return True
    return False


def choose_negative_template(last_user: str, category: str) -> str:
    templates = NEGATIVE_TEMPLATES.get(category) or NEGATIVE_TEMPLATES["other"]
    seed = hashlib.md5(compact(last_user).encode("utf-8")).hexdigest()
    index = int(seed[:8], 16) % len(templates)
    return templates[index]


def repair_negative(context: Sequence[str], rejected: str, chosen: str) -> Tuple[Optional[str], List[str], float]:
    last_user = context[-1]
    category = classify_category(last_user)
    issues: List[str] = []

    cleaned = strip_reply_placeholders(rejected)
    clauses = split_clauses(cleaned)
    if clauses:
        scored: List[Tuple[float, str]] = []
        for clause in clauses:
            score = 0.0
            clause_compact = compact(clause)
            if is_low_info(clause):
                score -= 5.0
            if contains_any(clause, EXTERNAL_CONTACT_HINTS):
                score += 4.5
            if contains_any(clause, RISKY_POLICY_HINTS | TIME_PROMISE_HINTS):
                score += 3.0
            score += overlap_score(last_user, clause) * 6.0
            if category != "other" and contains_any(clause, {
                "address": ADDRESS_HINTS,
                "refund": REFUND_HINTS,
                "shipping": SHIPPING_HINTS,
                "promo": PROMO_HINTS,
                "product": PRODUCT_HINTS,
            }[category]):
                score += 1.5
            if len(clause_compact) < 6:
                score -= 1.0
            elif len(clause_compact) <= 28:
                score += 1.0
            scored.append((score, clause))

        scored.sort(key=lambda item: (item[0], -len(compact(item[1]))), reverse=True)
        if scored and scored[0][0] > 0:
            cleaned = strip_reply_placeholders(scored[0][1])
            if cleaned != clean_text(rejected):
                issues.append("trimmed_negative")

    if negative_needs_rewrite(last_user, cleaned) or compact(cleaned) == compact(chosen):
        cleaned = strip_reply_placeholders(choose_negative_template(last_user, category))
        issues.append("rewritten_negative")

    if compact(cleaned) == compact(chosen):
        return None, issues + ["negative_same_as_positive"], -10.0
    if is_low_info(cleaned):
        return None, issues + ["negative_low_info"], -10.0
    if len(compact(cleaned)) < 6:
        return None, issues + ["negative_too_short"], -10.0

    score = overlap_score(last_user, cleaned) * 6.0
    if contains_any(cleaned, EXTERNAL_CONTACT_HINTS):
        score += 4.0
    if contains_any(cleaned, RISKY_POLICY_HINTS | TIME_PROMISE_HINTS):
        score += 3.0
    if contains_any(cleaned, ACTION_HINTS):
        score += 1.0
    if "rewritten_negative" in issues:
        score += 2.0
    return cleaned, issues, score


def normalize_for_template(text: str) -> str:
    s = compact(text)
    s = NUM_RE.sub("<NUM>", s)
    return s


def context_key(context: Sequence[str]) -> str:
    return "\n".join(clean_text(x) for x in context)


def build_raw_line(label: int, context: Sequence[str], response: str) -> str:
    return "\t".join([str(label), *context, response])


def build_prompt(context: Sequence[str]) -> str:
    lines: List[str] = []
    for idx, utt in enumerate(context):
        role = "用户" if idx % 2 == 0 else "客服"
        lines.append(f"{role}：{utt}")
    return "\n".join(lines)


def parse_raw_line(line: str) -> Tuple[int, List[str], str]:
    parts = line.rstrip("\n").split("\t")
    if len(parts) < 3:
        raise ValueError(f"bad raw line: {line!r}")
    label = int(parts[0])
    context = [clean_text(x) for x in parts[1:-1] if x.strip()]
    response = clean_text(parts[-1])
    return label, context, response


def load_preview_meta(path: Path) -> Dict[str, Dict[str, int]]:
    meta: Dict[str, Dict[str, int]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            key = context_key(row["context"])
            meta[key] = {
                "pos_line_no": row.get("pos_line_no"),
                "neg_line_no": row.get("neg_line_no"),
                "source_score": row.get("score", 0.0),
            }
    return meta


def load_raw_groups(path: Path, preview_meta: Dict[str, Dict[str, int]]) -> List[RepairGroup]:
    groups: List[RepairGroup] = []
    current: Dict[str, object] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            label, context, response = parse_raw_line(line)
            key = context_key(context)
            if not current or current["key"] != key:
                if current:
                    groups.append(
                        RepairGroup(
                            context=current["context"],  # type: ignore[arg-type]
                            chosen_original=current["chosen_original"],  # type: ignore[arg-type]
                            rejected_original=current["rejected_original"],  # type: ignore[arg-type]
                            chosen="",
                            rejected="",
                            category=classify_category(current["context"][-1]),  # type: ignore[index]
                            score=preview_meta.get(key, {}).get("source_score", 0.0),
                            source="top23000",
                            pos_line_no=preview_meta.get(key, {}).get("pos_line_no"),
                            neg_line_no=preview_meta.get(key, {}).get("neg_line_no"),
                            issues=[],
                        )
                    )
                current = {
                    "key": key,
                    "context": context,
                    "chosen_original": "",
                    "rejected_original": "",
                }
            if label == 1:
                current["chosen_original"] = response
            else:
                current["rejected_original"] = response
    if current:
        key = current["key"]  # type: ignore[assignment]
        groups.append(
            RepairGroup(
                context=current["context"],  # type: ignore[arg-type]
                chosen_original=current["chosen_original"],  # type: ignore[arg-type]
                rejected_original=current["rejected_original"],  # type: ignore[arg-type]
                chosen="",
                rejected="",
                category=classify_category(current["context"][-1]),  # type: ignore[index]
                score=preview_meta.get(key, {}).get("source_score", 0.0),
                source="top23000",
                pos_line_no=preview_meta.get(key, {}).get("pos_line_no"),
                neg_line_no=preview_meta.get(key, {}).get("neg_line_no"),
                issues=[],
            )
        )
    return groups


def parse_manual_prompt(prompt: str) -> List[str]:
    context: List[str] = []
    for raw_line in clean_text(prompt).split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        if "：" in line:
            _, content = line.split("：", 1)
        elif ":" in line:
            _, content = line.split(":", 1)
        else:
            content = line
        content = clean_text(content)
        if content:
            context.append(content)
    return context


def load_manual_groups(path: Path) -> List[RepairGroup]:
    if not path.exists():
        return []
    groups: List[RepairGroup] = []
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    for row in data:
        context = parse_manual_prompt(row.get("input", ""))
        chosen = clean_text(row.get("chosen", ""))
        rejected = clean_text(row.get("rejected", ""))
        if not context or not chosen or not rejected:
            continue
        groups.append(
            RepairGroup(
                context=context,
                chosen_original=chosen,
                rejected_original=rejected,
                chosen="",
                rejected="",
                category=classify_category(context[-1]),
                score=19.0,
                source="manual_round2",
                pos_line_no=None,
                neg_line_no=None,
                issues=["manual_source"],
            )
        )
    return groups


def repair_group(group: RepairGroup) -> Optional[RepairGroup]:
    chosen, pos_issues, pos_score = select_positive_clauses(group.context, group.chosen_original)
    if chosen is None:
        return None

    rejected, neg_issues, neg_score = repair_negative(group.context, group.rejected_original, chosen)
    if rejected is None:
        return None

    group.chosen = chosen
    group.rejected = rejected
    group.issues = group.issues + pos_issues + neg_issues
    group.score = group.score + pos_score + neg_score
    return group


def backup_paths(paths: Sequence[Path]) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = BACKUP_ROOT / f"dpo_dataset_rebuild_{stamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        if path.exists():
            dest = backup_dir / path.relative_to(BASE_DIR)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
    return backup_dir


def select_diverse(groups: Sequence[RepairGroup], top_k: int) -> List[RepairGroup]:
    groups = sorted(
        groups,
        key=lambda g: (g.score, 1 if g.source == "manual_round2" else 0, -len(compact(g.chosen))),
        reverse=True,
    )
    selected: List[RepairGroup] = []
    template_counter: Counter = Counter()
    last_user_counter: Counter = Counter()
    category_counter: Counter = Counter()
    caps = {
        "address": int(top_k * 0.14),
        "refund": int(top_k * 0.23),
        "shipping": int(top_k * 0.20),
        "promo": int(top_k * 0.14),
        "product": int(top_k * 0.16),
        "other": int(top_k * 0.25),
    }

    def max_template_repeat() -> int:
        if top_k <= 5000:
            return 3
        if top_k <= 10000:
            return 4
        return 5

    max_repeat = max_template_repeat()
    for group in groups:
        if len(selected) >= top_k:
            break
        template_key = normalize_for_template(group.chosen)
        last_user_key = normalize_for_template(group.context[-1])
        if template_counter[template_key] >= max_repeat:
            continue
        if last_user_counter[last_user_key] >= 2:
            continue
        if category_counter[group.category] >= caps.get(group.category, top_k):
            continue
        selected.append(group)
        template_counter[template_key] += 1
        last_user_counter[last_user_key] += 1
        category_counter[group.category] += 1

    if len(selected) < top_k:
        selected_keys = {
            (context_key(g.context), normalize_for_template(g.chosen), normalize_for_template(g.rejected))
            for g in selected
        }
        for group in groups:
            if len(selected) >= top_k:
                break
            key = (context_key(group.context), normalize_for_template(group.chosen), normalize_for_template(group.rejected))
            if key in selected_keys:
                continue
            selected.append(group)
            selected_keys.add(key)
    return selected[:top_k]


def write_raw(groups: Sequence[RepairGroup], raw_path: Path, positive_path: Optional[Path] = None) -> None:
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_path.open("w", encoding="utf-8") as f:
        for group in groups:
            f.write(build_raw_line(1, group.context, group.chosen) + "\n")
            f.write(build_raw_line(0, group.context, group.rejected) + "\n")
    if positive_path is not None:
        positive_path.parent.mkdir(parents=True, exist_ok=True)
        with positive_path.open("w", encoding="utf-8") as f:
            for group in groups:
                f.write(build_raw_line(1, group.context, group.chosen) + "\n")


def write_preview(groups: Sequence[RepairGroup], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for group in groups:
            row = {
                "score": round(group.score, 4),
                "category": group.category,
                "source": group.source,
                "context": group.context,
                "chosen": group.chosen,
                "rejected": group.rejected,
                "chosen_original": group.chosen_original,
                "rejected_original": group.rejected_original,
                "issues": group.issues,
                "pos_line_no": group.pos_line_no,
                "neg_line_no": group.neg_line_no,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_dpo_train(groups: Sequence[RepairGroup], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "instruction": PREFERENCE_INSTRUCTION,
            "input": build_prompt(group.context),
            "chosen": group.chosen,
            "rejected": group.rejected,
        }
        for group in groups
    ]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_stats(path: Path, meta: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def build_report(groups: Sequence[RepairGroup], backup_dir: Path, dropped: int, repaired_total: int) -> Dict[str, object]:
    category_counter = Counter(group.category for group in groups)
    issue_counter = Counter(issue for group in groups for issue in group.issues)
    return {
        "backup_dir": str(backup_dir),
        "canonical_raw": str(CANONICAL_RAW),
        "canonical_report": str(CANONICAL_REPORT),
        "groups_kept": len(groups),
        "groups_dropped": dropped,
        "repaired_total": repaired_total,
        "category_distribution": dict(category_counter),
        "issue_distribution": dict(issue_counter.most_common(100)),
        "source_distribution": dict(Counter(group.source for group in groups)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair DPO source pairs and rebuild train sets.")
    parser.add_argument("--source-raw", type=str, default=str(SOURCE_RAW))
    parser.add_argument("--source-preview", type=str, default=str(SOURCE_PREVIEW))
    parser.add_argument("--manual-file", type=str, default=str(MANUAL_FILE))
    args = parser.parse_args()

    source_raw = Path(args.source_raw)
    source_preview = Path(args.source_preview)
    manual_file = Path(args.manual_file)

    if not source_raw.exists():
        raise FileNotFoundError(source_raw)
    if not source_preview.exists():
        raise FileNotFoundError(source_preview)

    preview_meta = load_preview_meta(source_preview)
    raw_groups = load_raw_groups(source_raw, preview_meta)
    manual_groups = load_manual_groups(manual_file)

    repaired: List[RepairGroup] = []
    dropped = 0
    for group in raw_groups + manual_groups:
        result = repair_group(group)
        if result is None:
            dropped += 1
            continue
        repaired.append(result)

    backup_dir = backup_paths(
        [
            CANONICAL_RAW,
            CANONICAL_REPORT,
            CANONICAL_STATS,
            *(target["clean_raw"] for target in TARGETS),
            *(target["positive_only"] for target in TARGETS),
            *(target["preview"] for target in TARGETS),
            *(target["stats"] for target in TARGETS),
            *(target["dpo_train"] for target in TARGETS),
        ]
    )

    repaired.sort(
        key=lambda g: (g.score, 1 if g.source == "manual_round2" else 0, -len(compact(g.chosen))),
        reverse=True,
    )

    write_raw(repaired, CANONICAL_RAW)
    write_preview(repaired, CANONICAL_REPORT)
    write_stats(CANONICAL_STATS, build_report(repaired, backup_dir, dropped, len(raw_groups) + len(manual_groups)))

    for target in TARGETS:
        if len(repaired) < target["top_k"]:
            raise ValueError(
                f"not enough repaired groups for {target['name']}: have {len(repaired)}, need {target['top_k']}"
            )
        selected = select_diverse(repaired, target["top_k"])
        write_raw(selected, target["clean_raw"], target["positive_only"])
        write_preview(selected, target["preview"])
        write_dpo_train(selected, target["dpo_train"])
        write_stats(
            target["stats"],
            {
                "target": target["name"],
                "top_k": target["top_k"],
                "clean_raw": str(target["clean_raw"]),
                "positive_only": str(target["positive_only"]),
                "preview": str(target["preview"]),
                "dpo_train": str(target["dpo_train"]),
                "category_distribution": dict(Counter(group.category for group in selected)),
                "issue_distribution": dict(Counter(issue for group in selected for issue in group.issues).most_common(100)),
                "source_distribution": dict(Counter(group.source for group in selected)),
            },
        )

    summary = {
        "canonical_groups": len(repaired),
        "dropped_groups": dropped,
        "targets": {target["name"]: target["top_k"] for target in TARGETS},
        "backup_dir": str(backup_dir),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

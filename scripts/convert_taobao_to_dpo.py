from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

# =========================
# 固定配置
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "data" / "processed"

PROCESS_SPLITS = ("train", "dev")

INPUT_FILES = {
    "train": RAW_DIR / "train.txt",
    "dev": RAW_DIR / "dev.txt",
}

OUTPUT_FILES = {
    "train": OUTPUT_DIR / "taobao_dpo_train.json",
    "dev": OUTPUT_DIR / "taobao_dpo_dev.json",
}

MAX_NEGATIVES_PER_POSITIVE = 1
SEED = 42
MIN_SCORE_GAP = 3

DIGIT_ID_RE = re.compile(r"\b\d{8,}\b")
PHONE_RE = re.compile(r"\b1\d{10}\b")
SPACES_RE = re.compile(r"\s+")
CHINESE_RE = re.compile(r"[\u4e00-\u9fff]")
LATIN_RE = re.compile(r"[A-Za-z]{2,}")
MULTI_DIGIT_RE = re.compile(r"\d+")

BAD_SUBSTRINGS = {
    "SHOPNAME",
    "hdne",
    "iqn",
    "打错",
    "微信号",
    "薇信",
    "weixin",
    "返现",
    "好评",
    "加微信",
    "加微",
    "么么哒",
    "QQ",
    "qq",
}

LOW_INFO_REPLIES = {
    "嗯", "恩", "嗯嗯", "恩恩",
    "好", "好的", "好的呢", "好的哦", "好的亲",
    "是的", "您好", "您好亲", "你好",
    "在的", "在的哦", "在的呢",
    "有的", "可以的", "亲亲", "稍等", "稍等哦",
    "谢谢", "谢谢亲", "不客气", "客气啦"
}

ACTION_HINT_WORDS = {
    "发货", "退款", "退货", "补发", "拒收", "拦截",
    "下单", "拍下", "备注", "链接", "改价", "运费", "地址",
    "快递", "优惠", "活动", "库存", "有货", "没货",
    "纯棉", "无纺布", "尺寸", "质量", "售后", "检测",
    "申请", "查看", "核对", "包邮", "客服", "联系"
}


def normalize_text(text: str) -> str:
    text = text.strip()
    text = SPACES_RE.sub(" ", text)
    text = PHONE_RE.sub("<PHONE>", text)
    text = DIGIT_ID_RE.sub("<ID>", text)
    return text


def detokenize_zh(text: str) -> str:
    text = text.strip()
    if not text:
        return text

    parts = text.split()
    if not parts:
        return text

    out = [parts[0]]
    for cur in parts[1:]:
        prev = out[-1]

        prev_has_zh = bool(CHINESE_RE.search(prev))
        cur_has_zh = bool(CHINESE_RE.search(cur))
        prev_is_num = prev.isdigit() or bool(MULTI_DIGIT_RE.fullmatch(prev))
        cur_is_num = cur.isdigit() or bool(MULTI_DIGIT_RE.fullmatch(cur))
        prev_is_ascii = prev.isascii() and prev.replace("_", "").replace("-", "").isalnum()
        cur_is_ascii = cur.isascii() and cur.replace("_", "").replace("-", "").isalnum()

        if (prev_has_zh and cur_has_zh) or \
           (prev_has_zh and cur_is_num) or \
           (prev_is_num and cur_has_zh) or \
           (prev_is_ascii and cur_is_ascii) or \
           (prev_is_ascii and cur_has_zh) or \
           (prev_has_zh and cur_is_ascii):
            out[-1] = prev + cur
        else:
            out.append(cur)

    return "".join(out)


def clean_text(text: str) -> str:
    text = text.strip()
    text = text.replace("\\", "")
    text = SPACES_RE.sub(" ", text).strip()
    text = detokenize_zh(text)
    return text


def has_bad_placeholder(text: str) -> bool:
    if text in {"", "<ID>", "<PHONE>"}:
        return True
    for bad in BAD_SUBSTRINGS:
        if bad in text:
            return True
    return False


def is_mostly_garbage(text: str) -> bool:
    if not CHINESE_RE.search(text) and LATIN_RE.search(text):
        return True
    return False


def is_low_info_reply(text: str) -> bool:
    compact = text.replace(" ", "")
    if text in LOW_INFO_REPLIES:
        return True
    if len(compact) <= 4:
        return True
    if len(compact) <= 8 and not any(w in text for w in ACTION_HINT_WORDS):
        return True
    return False


def is_bad_reply(text: str) -> bool:
    if has_bad_placeholder(text):
        return True
    if is_mostly_garbage(text):
        return True
    return False


def quality_score(text: str) -> int:
    if is_bad_reply(text):
        return -10

    score = 0
    compact = text.replace(" ", "")

    if CHINESE_RE.search(text):
        score += 1

    score += min(len(compact) // 6, 4)

    for w in ACTION_HINT_WORDS:
        if w in text:
            score += 2

    if is_low_info_reply(text):
        score -= 4

    return score


def parse_line(line: str) -> Tuple[int, Tuple[str, ...], str] | None:
    parts = line.rstrip("\n").split("\t")
    if len(parts) < 3:
        return None

    try:
        label = int(parts[0])
    except ValueError:
        return None

    context = tuple(clean_text(normalize_text(x)) for x in parts[1:-1] if x.strip())
    response = clean_text(normalize_text(parts[-1]))
    return label, context, response


def build_prompt(context: Tuple[str, ...]) -> str:
    lines = ["请根据下面的电商客服历史对话，选择更合适的客服回复。", ""]
    for idx, utt in enumerate(context):
        role = "用户" if idx % 2 == 0 else "客服"
        lines.append(f"{role}：{utt}")
    lines.append("客服：")
    return "\n".join(lines)


def convert_one_file(input_path: Path, output_path: Path) -> None:
    if not input_path.exists():
        print(f"[DPO] 跳过，不存在: {input_path}")
        return

    random.seed(SEED)
    groups: Dict[Tuple[str, ...], Dict[str, List[str]]] = {}

    total = 0
    parse_failed = 0

    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            total += 1
            parsed = parse_line(line)
            if parsed is None:
                parse_failed += 1
                continue

            label, context, response = parsed
            if not context or not response:
                continue

            bucket = groups.setdefault(context, {"pos": [], "neg": []})
            if label == 1:
                bucket["pos"].append(response)
            else:
                bucket["neg"].append(response)

    samples = []
    valid_groups = 0
    stats = {
        "parse_failed": parse_failed,
        "bad_pos": 0,
        "low_info_pos": 0,
        "bad_neg": 0,
        "same_pair": 0,
        "weak_preference": 0,
        "saved": 0,
    }

    for context, bucket in groups.items():
        pos_list = list(dict.fromkeys(bucket["pos"]))
        neg_list = list(dict.fromkeys(bucket["neg"]))

        if not pos_list or not neg_list:
            continue

        valid_groups += 1
        prompt = build_prompt(context)

        for pos in pos_list:
            if is_bad_reply(pos):
                stats["bad_pos"] += 1
                continue

            if is_low_info_reply(pos):
                stats["low_info_pos"] += 1
                continue

            chosen_negs = neg_list[:]
            random.shuffle(chosen_negs)
            chosen_negs = chosen_negs[:MAX_NEGATIVES_PER_POSITIVE]

            for neg in chosen_negs:
                if neg == pos:
                    stats["same_pair"] += 1
                    continue

                if is_bad_reply(neg):
                    stats["bad_neg"] += 1
                    continue

                pos_score = quality_score(pos)
                neg_score = quality_score(neg)

                if pos_score <= neg_score or (pos_score - neg_score) < MIN_SCORE_GAP:
                    stats["weak_preference"] += 1
                    continue

                sample = {
                    "instruction": "你是电商平台客服偏好优化数据构建器，请偏向更准确、相关、礼貌且可执行的回复。",
                    "input": prompt,
                    "chosen": pos,
                    "rejected": neg,
                }
                samples.append(sample)
                stats["saved"] += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    print(f"[DPO] input: {input_path}")
    print(f"[DPO] total lines: {total}")
    print(f"[DPO] total grouped contexts: {len(groups)}")
    print(f"[DPO] valid groups(pos+neg): {valid_groups}")
    print(f"[DPO] saved pairs: {len(samples)}")
    print(f"[DPO] stats: {stats}")
    print(f"[DPO] output: {output_path}")
    print("-" * 60)


def main():
    print("[DPO] 开始处理淘宝数据...")
    for split in PROCESS_SPLITS:
        convert_one_file(INPUT_FILES[split], OUTPUT_FILES[split])
    print("[DPO] 全部处理完成。")


if __name__ == "__main__":
    main()
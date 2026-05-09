from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List, Tuple

# =========================
# 固定配置（加入 test）
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
CLEAN_DIR = BASE_DIR / "data" / "cleaned"
OUTPUT_DIR = BASE_DIR / "data" / "processed_5000"

PROCESS_SPLITS = ("train", "dev", "test")

INPUT_FILES = {
    "train": CLEAN_DIR / "train_clean_top5000_raw.txt",
    "dev": RAW_DIR / "dev.txt",
    "test": RAW_DIR / "test.txt",
}

OUTPUT_FILES = {
    "train": OUTPUT_DIR / "taobao_sft_train.json",
    "dev": OUTPUT_DIR / "taobao_sft_dev.json",
    "test": OUTPUT_DIR / "taobao_sft_test.json",
}

MIN_TURNS = 1
MIN_RESPONSE_CHARS = 6  # 过滤太短回复

DIGIT_ID_RE = re.compile(r"\b\d{8,}\b")
PHONE_RE = re.compile(r"\b1\d{10}\b")
SPACES_RE = re.compile(r"[ \t]+")
BLANK_LINES_RE = re.compile(r"\n{3,}")
CHINESE_RE = re.compile(r"[\u4e00-\u9fff]")
LATIN_RE = re.compile(r"[A-Za-z]{2,}")
MULTI_DIGIT_RE = re.compile(r"\d+")
HALFWIDTH_COLON_AFTER_ROLE_RE = re.compile(r"^(用户|客服|买家|顾客|商家|店主|掌柜|小二|卖家)\s*:\s*")
ROLE_PREFIX_RE = re.compile(r"^(用户|客服|买家|顾客|商家|店主|掌柜|小二|卖家)\s*[：:]\s*")
TASK_PREFIX_RE = re.compile(
    r"^请根据下面的电商客服历史对话，生成下一句合适的客服回复。\s*",
    re.S,
)

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

SYSTEM_INSTRUCTION = "你是电商客服。只回答当前最后一句用户问题，不要扩展其他信息。如果历史对话里没有明确信息，只能给出简短、保守回复"
TASK_INPUT_PREFIX = "请根据下面的电商客服历史对话，生成下一句合适的客服回复。"


def normalize_text(text: str) -> str:
    text = text.strip()
    if not text:
        return text

    text = text.replace("\u3000", " ")
    text = text.replace("\xa0", " ")

    text = PHONE_RE.sub("<PHONE>", text)
    text = DIGIT_ID_RE.sub("<ID>", text)

    text = (
        text.replace("“", '"')
            .replace("”", '"')
            .replace("‘", "'")
            .replace("’", "'")
            .replace("—", "-")
            .replace("–", "-")
            .replace("…", "...")
    )

    text = text.replace("\\", "")
    return text.strip()


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

        if (
            (prev_has_zh and cur_has_zh)
            or (prev_has_zh and cur_is_num)
            or (prev_is_num and cur_has_zh)
        ):
            out[-1] = prev + cur
        else:
            out.append(cur)

    return " ".join(out)


def normalize_role_prefix(text: str) -> str:
    text = text.strip()
    if not text:
        return text

    m = ROLE_PREFIX_RE.match(text)
    if not m:
        return text

    role = m.group(1)
    content = ROLE_PREFIX_RE.sub("", text, count=1).strip()

    if role in {"用户", "买家", "顾客"}:
        role_std = "用户"
    else:
        role_std = "客服"

    return f"{role_std}：{content}" if content else f"{role_std}："


def normalize_punctuation_spacing(text: str) -> str:
    text = text.strip()
    if not text:
        return text

    text = HALFWIDTH_COLON_AFTER_ROLE_RE.sub(lambda m: m.group(1) + "：", text)
    text = re.sub(r"\s*:\s*", "：", text)
    text = re.sub(r"\s+([，。！？；：、）】》〉])", r"\1", text)
    text = re.sub(r"([（【《〈])\s+", r"\1", text)
    text = re.sub(r"(\d)\s*-\s*(\d)", r"\1-\2", text)
    text = SPACES_RE.sub(" ", text)

    return text.strip()


def clean_text(text: str) -> str:
    text = normalize_text(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = BLANK_LINES_RE.sub("\n\n", text)

    lines = []
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        line = normalize_punctuation_spacing(line)
        line = detokenize_zh(line)
        line = normalize_role_prefix(line)
        lines.append(line)

    text = "\n".join(lines).strip()
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
    if compact in LOW_INFO_REPLIES:
        return True
    if len(compact) <= 4:
        return True
    if len(compact) <= 8 and not any(w in compact for w in ACTION_HINT_WORDS):
        return True
    return False


def is_bad_reply(text: str) -> bool:
    if has_bad_placeholder(text):
        return True
    if is_mostly_garbage(text):
        return True
    return False


def strip_task_prefix(text: str) -> str:
    return TASK_PREFIX_RE.sub("", text).strip()


def remove_leading_role(text: str) -> str:
    text = text.strip()
    if text.startswith("客服："):
        return text[3:].strip()
    if text.startswith("客服:"):
        return text[3:].strip()
    return text


def parse_line(line: str) -> Tuple[int, List[str], str] | None:
    parts = line.rstrip("\n").split("\t")
    if len(parts) < 3:
        return None

    try:
        label = int(parts[0])
    except ValueError:
        return None

    context = [clean_text(x) for x in parts[1:-1] if x.strip()]
    response = clean_text(parts[-1])
    response = remove_leading_role(response)
    return label, context, response


def build_prompt(context: List[str]) -> str:
    lines = [TASK_INPUT_PREFIX, ""]

    for idx, utt in enumerate(context):
        utt = strip_task_prefix(utt)
        utt = utt.strip()
        utt = normalize_role_prefix(utt)

        if ROLE_PREFIX_RE.match(utt):
            lines.append(utt)
        else:
            role = "用户" if idx % 2 == 0 else "客服"
            lines.append(f"{role}：{utt}")

    if not lines[-1].startswith("客服："):
        lines.append("客服：")
    elif lines[-1] != "客服：":
        lines.append("客服：")

    return "\n".join(lines).strip()


def should_keep(context: List[str], response: str) -> Tuple[bool, str]:
    if len(context) < MIN_TURNS:
        return False, "too_few_turns"

    if len(response.replace(" ", "")) < MIN_RESPONSE_CHARS:
        return False, "too_short"

    if is_bad_reply(response):
        return False, "bad_reply"

    if is_low_info_reply(response):
        return False, "low_info"

    return True, "ok"


def convert_one_file(input_path: Path, output_path: Path) -> None:
    if not input_path.exists():
        print(f"[SFT] 跳过，不存在: {input_path}")
        return

    samples = []
    total = 0
    positive = 0

    stats = {
        "parse_failed": 0,
        "not_positive": 0,
        "too_few_turns": 0,
        "too_short": 0,
        "bad_reply": 0,
        "low_info": 0,
        "saved": 0,
    }

    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            total += 1
            parsed = parse_line(line)
            if parsed is None:
                stats["parse_failed"] += 1
                continue

            label, context, response = parsed
            if label != 1:
                stats["not_positive"] += 1
                continue

            positive += 1
            keep, reason = should_keep(context, response)
            if not keep:
                stats[reason] += 1
                continue

            sample = {
                "instruction": SYSTEM_INSTRUCTION,
                "input": build_prompt(context),
                "output": response,
            }
            samples.append(sample)
            stats["saved"] += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    print(f"[SFT] input: {input_path}")
    print(f"[SFT] total lines: {total}")
    print(f"[SFT] positive lines: {positive}")
    print(f"[SFT] saved samples: {len(samples)}")
    print(f"[SFT] stats: {stats}")
    print(f"[SFT] output: {output_path}")
    print("-" * 60)


def main():
    parser = argparse.ArgumentParser(description="将原始/cleaned 淘宝对话转为 SFT JSON")
    parser.add_argument("--input", type=str, default=None, help="单文件输入路径")
    parser.add_argument("--output", type=str, default=None, help="单文件输出路径")
    args = parser.parse_args()

    if args.input or args.output:
        if not (args.input and args.output):
            raise ValueError("单文件模式下必须同时提供 --input 和 --output")
        convert_one_file(Path(args.input), Path(args.output))
        return

    print("[SFT] 开始处理淘宝数据...")
    for split in PROCESS_SPLITS:
        convert_one_file(INPUT_FILES[split], OUTPUT_FILES[split])
    print("[SFT] 全部处理完成。")


if __name__ == "__main__":
    main()

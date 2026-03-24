#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
将淘宝 E-commerce Dialogue Corpus 的检索式数据
(label \t utterance1 \t ... \t utteranceN \t response)
转换为 LLaMA-Factory 可用的 SFT 数据格式。

直接运行：
python scripts/convert_taobao_to_sft.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Tuple

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
    "train": OUTPUT_DIR / "taobao_sft_train.json",
    "dev": OUTPUT_DIR / "taobao_sft_dev.json",
}

MIN_TURNS = 1
MIN_RESPONSE_CHARS = 2

DIGIT_ID_RE = re.compile(r"\b\d{8,}\b")
PHONE_RE = re.compile(r"\b1\d{10}\b")
SPACES_RE = re.compile(r"\s+")
CHINESE_RE = re.compile(r"[\u4e00-\u9fff]")
LATIN_RE = re.compile(r"[A-Za-z]{2,}")

BAD_SUBSTRINGS = {
    "SHOPNAME",
    "hdne",
    "iqn",
}

LOW_INFO_REPLIES = {
    "嗯", "恩", "嗯嗯", "恩恩",
    "好", "好的", "好 的", "好 的 呢", "好 的 哦", "好 的 亲",
    "嗯 嗯", "嗯 嗯 亲", "嗯 嗯 好 的", "恩 呢",
    "在 的", "在 的 哦", "在 的 呢",
    "您好", "您好 亲", "你好", "客气 啦"
}


def normalize_text(text: str) -> str:
    text = text.strip()
    text = SPACES_RE.sub(" ", text)
    text = PHONE_RE.sub("<PHONE>", text)
    text = DIGIT_ID_RE.sub("<ID>", text)
    return text


def clean_reply_text(text: str) -> str:
    text = text.strip()
    text = text.replace("\\", "")
    text = SPACES_RE.sub(" ", text).strip()
    return text


def has_bad_placeholder(text: str) -> bool:
    if text in {"", "<ID>", "<PHONE>"}:
        return True
    for bad in BAD_SUBSTRINGS:
        if bad in text:
            return True
    return False


def is_mostly_garbage(text: str) -> bool:
    # 没中文但有异常英文 token，通常是脏样本
    if not CHINESE_RE.search(text) and LATIN_RE.search(text):
        return True
    return False


def is_low_info_reply(text: str) -> bool:
    text_norm = text.strip()
    compact = text.replace(" ", "")
    if text_norm in LOW_INFO_REPLIES:
        return True
    if len(compact) <= 4:
        return True
    return False


def is_bad_reply(text: str) -> bool:
    text = clean_reply_text(text)
    if has_bad_placeholder(text):
        return True
    if is_mostly_garbage(text):
        return True
    return False


def parse_line(line: str) -> Tuple[int, List[str], str] | None:
    parts = line.rstrip("\n").split("\t")
    if len(parts) < 3:
        return None
    try:
        label = int(parts[0])
    except ValueError:
        return None

    context = [normalize_text(x) for x in parts[1:-1] if x.strip()]
    response = clean_reply_text(normalize_text(parts[-1]))
    return label, context, response


def build_prompt(context: List[str]) -> str:
    lines = ["请根据下面的电商客服历史对话，生成下一句合适的客服回复。", ""]
    for idx, utt in enumerate(context):
        role = "用户" if idx % 2 == 0 else "客服"
        lines.append(f"{role}：{utt}")
    lines.append("客服：")
    return "\n".join(lines)


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
                "instruction": "你是电商平台客服，请基于历史对话给出准确、礼貌、可执行的回复。",
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
    print("[SFT] 开始处理淘宝数据...")
    for split in PROCESS_SPLITS:
        convert_one_file(INPUT_FILES[split], OUTPUT_FILES[split])
    print("[SFT] 全部处理完成。")


if __name__ == "__main__":
    main()
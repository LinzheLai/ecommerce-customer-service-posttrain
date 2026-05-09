from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ============================================================
# 目标：
# 1) 从原始 train.txt 中筛 1w 条更适合 SFT 的高质量对话
# 2) 优先保留“短、准、贴最后一句用户问题”的回答
# 3) 强压物流/优惠券/退款等高频模板污染
# 4) 强过滤幻觉式承诺、流程堆砌、模板尾巴
# 5) 新增：硬过滤弱问题、片段式 chosen、软模板客服腔、历史活动话术
# 6) 输出保持 raw txt 兼容 SFT / DPO
# ============================================================

DEFAULT_BASE_DIR = Path("/opt/data/llz/ecommerce-customer-service-posttrain")
DEFAULT_INPUT = DEFAULT_BASE_DIR / "data" / "raw" / "train.txt"
DEFAULT_OUT_DIR = DEFAULT_BASE_DIR / "data" / "cleaned"

PHONE_RE = re.compile(r"\b1\d{10}\b")
LONG_DIGIT_RE = re.compile(r"\b\d{8,}\b")
NUM_RE = re.compile(r"\d+")
YEAR_RE = re.compile(r"20(1\d|2\d)年?")
CHINESE_RE = re.compile(r"[\u4e00-\u9fff]")
LATIN_RE = re.compile(r"[A-Za-z]{2,}")
MULTI_SPACE_RE = re.compile(r"[ \t]+")
PUNC_ONLY_RE = re.compile(r"^[\W_]+$", re.UNICODE)
MULTI_PUNC_RE = re.compile(r"([，。！？；、,.!?;:]){2,}")

# -----------------------------
# 强过滤项
# -----------------------------
BAD_SUBSTRINGS = {
    "SHOPNAME",
    "微信", "薇信", "weixin", "vx", "Vx", "加微", "加微信",
    "QQ", "qq", "qq群", "进群",
    "返现", "好评", "五星", "晒图", "公众号",
}

LOW_INFO_REPLIES = {
    "嗯", "恩", "嗯嗯", "恩恩", "好的", "好", "好的呢", "好的哦", "好哒",
    "是的", "是的呢", "您好", "你好", "在的", "在的哦", "在的呢",
    "可以的", "有的", "稍等", "稍等哦", "谢谢", "不客气", "客气啦",
    "嗯呢", "恩呢", "收到", "好的亲", "好的哦亲", "好的哈",
    "好的呢亲", "好滴", "嗯呐", "恩好的", "好的额", "嗯好的",
}

QUESTION_HINT_WORDS = {
    "吗", "么", "呢", "多少", "几", "多久", "什么时候", "能不能",
    "有没有", "怎么", "为何", "为啥", "什么", "是否", "咋", "哪", "怎么弄",
}

ACTION_HINT_WORDS = {
    "发货", "退款", "退货", "补发", "拒收", "拦截", "售后", "退回", "签收",
    "下单", "拍下", "备注", "改价", "运费", "地址",
    "快递", "物流", "催件",
    "优惠", "活动", "优惠券", "券", "满减", "包邮",
    "库存", "有货", "没货", "到货",
    "纯棉", "无纺布", "尺寸", "规格", "材质", "质量", "日期", "生产日期", "保质期",
    "申请", "核对", "联系", "处理", "安排", "补偿",
}

TOPIC_WORDS = {
    "发货", "退款", "退货", "补发", "拒收", "运费", "地址", "快递", "物流",
    "库存", "有货", "没货", "日期", "质量", "尺寸", "材质", "活动", "优惠",
    "包邮", "链接", "备注", "改价", "售后", "签收", "拦截",
    "顺丰", "邮政", "韵达", "百世", "汇通", "中通", "申通", "ems",
    "纯棉", "无纺布", "湿巾", "棉柔巾", "茶刀", "茶针", "核桃", "榴莲", "粽子", "青团",
}

STOPWORDS = {
    "的", "了", "呢", "啊", "呀", "哦", "亲", "亲亲", "一下", "这个", "那个", "现在",
    "已经", "还是", "一个", "这款", "那款", "您好", "你好", "在吗", "在", "谢谢", "好吧",
    "可以", "好的", "嗯嗯", "恩恩", "请问", "麻烦", "实在", "不好意思",
}

# -----------------------------
# 强模板与低价值尾巴
# -----------------------------
ADDRESS_TAIL_RE = re.compile(
    r"(?:亲|亲亲|您好|好的|嗯嗯|恩恩|哦|呢|哈|呀|哟|啦|呐|客官|小二)*"
    r"(?:请核对一下收货地址|核对一下收货地址)"
    r"(?:哦|呢|哈|呀|哟|啦|呐|亲|亲亲|客官|吧|。|！|!|~|～|'|\")*\s*$"
)
REPEATED_ADDRESS_RE = re.compile(r"(请核对一下收货地址[哦呢哈呀哟啦呐亲亲]*){2,}")

PURE_ADDRESS_REPLY_KEYS = {
    "请核对一下收货地址",
    "好的请核对一下收货地址",
    "不客气请核对一下收货地址",
    "稍等请核对一下收货地址",
    "备注了哦请核对一下收货地址",
}

HARD_DROP_TEMPLATE_KEYS = {
    "请核对一下收货地址",
    "好的请核对一下收货地址",
    "不客气请核对一下收货地址",
    "稍等请核对一下收货地址",
    "拍下后<num>小时内发货",
    "拍下后<num>小时内发货会尽快的",
    "<num>小时内发货",
    "按付款时间顺序<num>小时内发货的",
    "按付款时间顺序<num>小时内发货的请核对一下收货地址",
}

# 高频模板：强降权
HIGH_RISK_TEMPLATE_HINTS = [
    "默认", "优惠券", "满减", "包邮", "运费险", "七天无理由", "退款", "拒收", "拦截",
    "72小时内发货", "48小时内发货", "随机发", "请核对一下收货地址",
]

# 幻觉/承诺型表达：强降权
HALLUCINATION_RISK_HINTS = [
    "主动联系您", "自动退款", "无条件补发", "无条件退款", "一定给您", "保证到",
    "24小时内处理", "48小时内处理", "7天无理由", "不满意就拒收", "收到后拒收",
    "我们会通知快递", "会有专人", "有权限", "没有权限", "系统会自动",
]

# 历史活动 / 时效敏感话术：直接强降权甚至过滤
TEMPORAL_ACTIVITY_HINTS = [
    "双11", "双12", "618", "520", "聚划算", "预售", "店庆", "秒杀",
    "明天开始", "后天开始", "今天开始", "几点开始", "10点开始", "9点开始",
    "22号以后发货", "16号", "7号", "3号", "4号", "5月4号",
    "前100名", "前4000名", "定金", "尾款", "换购",
]

SFT_HARD_RISK_HINTS = {
    "48小时", "72小时", "当天发货", "次日发货", "今天16点前", "早拍早发货",
    "工厂直销", "放心购买", "质量保证", "绝不做陈货",
    "前6000名", "前4000名", "前100名", "聚划算", "秒杀", "年前",
    "拍下确认收货后找客服返",
}
SFT_REFUND_PROMISE_HINTS = {
    "会给您退款", "会处理您的退款", "自动退款",
}
SFT_SHIPPING_TEMPLATE_HINTS = {
    "随机", "默认", "不指定",
}

GENERIC_TEMPLATE_PATTERNS = [
    re.compile(r"^拍下后<NUM>小时内发货.*$", re.I),
    re.compile(r"^活动期间.*小时内发货.*$", re.I),
    re.compile(r"^本店默认.*快递.*$", re.I),
    re.compile(r"^快递随机发.*$", re.I),
    re.compile(r"^按付款时间顺序.*发货.*$", re.I),
    re.compile(r"^亲亲可以领取.*优惠券.*$", re.I),
    re.compile(r"^亲亲.*申请退款.*$", re.I),
]

SFT_MASKED_TIME_PROMISE_PATTERNS = [
    re.compile(r".*<NUM>-<NUM>天发货.*", re.I),
    re.compile(r".*<NUM>天发货.*", re.I),
    re.compile(r".*<NUM>天左右发货.*", re.I),
    re.compile(r".*发货后<NUM>-<NUM>天.*", re.I),
    re.compile(r".*发货后<NUM>天.*", re.I),
]

# 更高质量的业务样本偏好
GOOD_ACTION_PATTERNS = [
    "不能折现", "稍等我帮您问下", "可以备注", "拍的时候备注", "这个不能确定",
    "需要提供照片", "帮您联系", "给您改价", "可以申请退款", "可以补发",
]

# 分类词
CATEGORY_HINTS = {
    "after_sale": {"退款", "退货", "拒收", "补发", "售后", "运费险", "退回", "拦截", "退款原因"},
    "product_info": {"尺寸", "材质", "纯棉", "无纺布", "规格", "质量", "生产日期", "保质期", "口味", "日期"},
    "price_activity": {"优惠", "活动", "优惠券", "券", "满减", "改价", "折现", "便宜", "包邮"},
    "courier_info": {"快递", "物流", "邮政", "韵达", "百世", "汇通", "中通", "申通", "ems", "顺丰"},
    "shipping_sla": {"发货", "到货", "几天", "多久", "什么时候到", "发出"},
}

# 软模板客服腔：不一定绝对删除，但默认强降权
SOFT_TEMPLATE_HINTS = [
    "质量保证的亲",
    "质量不错的亲",
    "质量不错的呢",
    "都是近期的生产日期的",
    "近期的生产日期的亲",
    "纯棉的更好一些",
    "对的亲",
    "可以的亲",
    "好的亲",
]

PROMO_HINT_WORDS = {
    "优惠", "优惠券", "领券", "满减", "折现", "改价", "包邮", "活动",
}
ADDRESS_HINT_WORDS = {
    "地址", "收货", "核对", "改地址", "收货地址",
}
SHIPPING_HINT_WORDS = {
    "快递", "发货", "到货", "邮政", "韵达", "申通", "中通", "百世", "天天", "顺丰",
}
REFUND_HINT_WORDS = {
    "退款", "退货", "售后", "补发", "补偿", "拒收", "拦截",
}
SALES_PITCH_HINTS = [
    "质量保证", "放心购买", "工厂直销", "一分价钱一分货",
    "可以先拍下", "您看可以吗", "店铺首页", "领取优惠券",
]
NON_CJK_NOISE_RE = re.compile(r"[\u0600-\u06FF\u0E00-\u0E7F\u0400-\u04FF\uAC00-\uD7AF]")
ROLE_CHAT_LEAK_RE = re.compile(r"(^|\n)\s*(?:user|assistant|system)\b", re.I)
CHINESE_ROLE_LEAK_RE = re.compile(r"(^|\n)\s*(?:用户|客服|买家|卖家)\s*[：:]", re.I)
CLAUSE_SPLIT_RE = re.compile(r"[，。！？!?；;]+")
TOPIC_EXPANSION_RULES = {
    "promo": (PROMO_HINT_WORDS, PROMO_HINT_WORDS | {"便宜", "划算"}),
    "address": (ADDRESS_HINT_WORDS, ADDRESS_HINT_WORDS | {"填写"}),
    "shipping": (SHIPPING_HINT_WORDS, SHIPPING_HINT_WORDS | {"什么时候", "几天", "时效"}),
    "refund": (REFUND_HINT_WORDS, REFUND_HINT_WORDS | {"质量问题"}),
}

STRICT_MODE = False
GENERIC_MODE = False
SFT_MODE = False

SHOP_NAME_HINTS = {
    "爱哒", "爱哒家", "爱哒小店",
    "味立家",
    "九都",
    "屹茗堂",
}

STORE_SPECIFIC_OPERATION_HINTS = {
    "店铺首页", "收藏店铺", "关注店铺", "收藏本店", "关注本店",
    "晒一下收藏店铺截图", "收藏截图", "进店", "进小店", "进入小店",
    "拍下后联系我", "联系我", "联系我们", "联系客服", "联系我改价", "联系客服改价", "联系掌柜", "掌柜",
    "小二家", "本店", "小店", "店铺", "链接发下", "发下链接",
    "微信", "加微", "二维码", "旺旺名", "QQ", "qq",
}

STORE_SPECIFIC_POLICY_HINTS = {
    "优惠券", "领券", "活动", "满减", "包邮", "拍下自动减价",
    "默认", "改价", "前100名", "前1000名", "店庆", "秒杀", "预售", "定金",
}

GENERIC_RESPONSE_HARD_HINTS = {
    "微信", "二维码", "旺旺名", "QQ", "qq",
    "店铺首页", "收藏店铺", "关注店铺", "晒一下收藏店铺截图",
    "拍下后联系我", "联系我改价", "联系客服改价",
    "联系我", "联系我们", "联系客服",
    "仓库", "72小时", "安全放心",
}

MISSING_CONTEXT_START_PREFIXES = (
    "那", "那我", "那这个", "那就", "那还", "那能", "那是不是",
    "然后", "所以", "所以说", "这样", "那么",
)

WEAK_LAST_USER_RE = re.compile(r"^(?:嗯|好|好的|哦|恩|是的|在吗|在|<ID>|<NUM>|\d+)$")


@dataclass
class Candidate:
    label: int
    context: List[str]
    response: str
    raw_line: str
    line_no: int
    context_key: str
    response_key: str
    response_template_key: str
    pair_key: str
    category: str
    score: float
    issues: List[str]


@dataclass
class GroupSelection:
    context_key: str
    pos: Candidate
    neg: Candidate
    score: float
    diversity_key: str
    pair_key: str
    last_user_key: str
    category: str


def normalize_text(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    text = text.replace("\u3000", " ").replace("\xa0", " ")
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    text = text.replace("—", "-").replace("–", "-").replace("…", "...")
    text = text.replace("\\", "")
    text = PHONE_RE.sub("<PHONE>", text)
    text = LONG_DIGIT_RE.sub("<ID>", text)
    text = MULTI_SPACE_RE.sub(" ", text)
    text = MULTI_PUNC_RE.sub(lambda m: m.group(1), text)
    return text.strip()


def detokenize_zh(text: str) -> str:
    parts = text.split()
    if not parts:
        return text
    out = [parts[0]]
    for cur in parts[1:]:
        prev = out[-1]
        prev_has_zh = bool(CHINESE_RE.search(prev))
        cur_has_zh = bool(CHINESE_RE.search(cur))
        prev_is_num = prev.isdigit() or bool(NUM_RE.fullmatch(prev))
        cur_is_num = cur.isdigit() or bool(NUM_RE.fullmatch(cur))
        if (prev_has_zh and cur_has_zh) or (prev_has_zh and cur_is_num) or (prev_is_num and cur_has_zh):
            out[-1] = prev + cur
        else:
            out.append(cur)
    return " ".join(out)


def clean_utt(text: str) -> str:
    text = normalize_text(text)
    text = detokenize_zh(text)
    text = MULTI_SPACE_RE.sub(" ", text)
    return text.strip()


def mask_for_key(text: str) -> str:
    text = clean_utt(text)
    text = YEAR_RE.sub("<YEAR>", text)
    text = NUM_RE.sub("<NUM>", text)
    return text


def normalize_for_template(text: str) -> str:
    text = mask_for_key(text)
    for word in ["亲亲", "亲", "哦", "呢", "呀", "哈", "啦", "哟", "呐", "客官", "小二", "您好", "你好"]:
        text = text.replace(word, "")
    text = text.replace(" ", "")
    text = re.sub(r"[，。！？；：、,.!?;:'\"~～]", "", text)
    return text.strip()


def tokenize_for_overlap(text: str) -> List[str]:
    text = mask_for_key(text).lower()
    tokens = set()
    for w in TOPIC_WORDS:
        if w.lower() in text:
            tokens.add(w.lower())
    for frag in re.findall(r"[\u4e00-\u9fff]{2,}|[a-z]{2,}|<id>|<phone>|<num>|<year>", text):
        if frag in STOPWORDS:
            continue
        if len(frag) == 1:
            continue
        tokens.add(frag)
    return sorted(tokens)


def question_strength(text: str) -> int:
    s = text.replace(" ", "")
    score = 0
    if "?" in s or "？" in s:
        score += 1
    for w in QUESTION_HINT_WORDS:
        if w in s:
            score += 1
    return score


def keyword_overlap(a: str, b: str) -> float:
    ta = set(tokenize_for_overlap(a))
    tb = set(tokenize_for_overlap(b))
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    denom = max(1, min(len(ta), len(tb)))
    return inter / denom


def contains_bad_substring(text: str) -> bool:
    lower = text.lower()
    return any(b.lower() in lower for b in BAD_SUBSTRINGS)


def is_garbage_text(text: str) -> bool:
    compact = text.replace(" ", "")
    if not compact:
        return True
    if compact in {"<ID>", "<PHONE>", "<NUM>", "<YEAR>"}:
        return True
    if PUNC_ONLY_RE.fullmatch(compact):
        return True
    if not CHINESE_RE.search(compact) and LATIN_RE.search(compact):
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


def contains_temporal_activity_hint(text: str) -> bool:
    s = text.replace(" ", "")
    return any(h in s for h in TEMPORAL_ACTIVITY_HINTS)


def contains_soft_template_hint(text: str) -> bool:
    s = text.replace(" ", "")
    return any(h in s for h in SOFT_TEMPLATE_HINTS)


def contains_role_or_chat_leak(text: str) -> bool:
    lower = text.lower()
    if "<|im_start|>" in lower or "<|im_end|>" in lower:
        return True
    if ROLE_CHAT_LEAK_RE.search(lower):
        return True
    return bool(CHINESE_ROLE_LEAK_RE.search(text))


def contains_non_cjk_noise(text: str) -> bool:
    return bool(NON_CJK_NOISE_RE.search(text))


def clause_count(text: str) -> int:
    compact = text.replace(" ", "")
    if not compact:
        return 0
    return len([x for x in CLAUSE_SPLIT_RE.split(compact) if x])


def count_unasked_topic_expansions(last_user: str, response: str) -> Tuple[int, List[str]]:
    last_user = last_user.replace(" ", "")
    response = response.replace(" ", "")
    hits: List[str] = []
    for name, (resp_words, user_words) in TOPIC_EXPANSION_RULES.items():
        if any(w in response for w in resp_words) and not any(w in last_user for w in user_words):
            hits.append(name)
    return len(hits), hits


def looks_sales_pitch(text: str) -> bool:
    compact = text.replace(" ", "")
    return any(h in compact for h in SALES_PITCH_HINTS)


def contains_shop_name(text: str) -> bool:
    compact = text.replace(" ", "")
    return any(h in compact for h in SHOP_NAME_HINTS)


def count_store_specific_hits(text: str) -> Tuple[int, List[str]]:
    compact = text.replace(" ", "")
    hits: List[str] = []
    if contains_shop_name(compact):
        hits.append("shop_name")
    if any(h in compact for h in STORE_SPECIFIC_OPERATION_HINTS):
        hits.append("store_operation")
    if any(h in compact for h in STORE_SPECIFIC_POLICY_HINTS):
        hits.append("store_policy")
    return len(hits), hits


def has_incomplete_context_start(context: List[str]) -> bool:
    if len(context) < 2:
        return False
    first = context[0].replace(" ", "")
    if not first:
        return True
    if any(first.startswith(prefix) for prefix in MISSING_CONTEXT_START_PREFIXES):
        return True
    return is_low_info_reply(first) or len(first) <= 4


def is_fragmented_response(text: str) -> bool:
    """
    过滤片段式回答：
    - 纯数字/数字+尺寸/纯参数片段
    - 没有完整句意，只有“纯棉1020”“1520的尺寸”“近期的生产日期的”
    """
    s = text.replace(" ", "")
    if len(s) <= 3:
        return True

    # 纯参数片段
    fragment_patterns = [
        r"^(纯棉|无纺布|植物纤维)?\d{3,4}$",
        r"^\d{2,4}的尺寸$",
        r"^(纯棉|无纺布|植物纤维)(\d{2,4})?的?$",
        r"^近期的生产日期的?$",
        r"^保质期\d+年(的)?$",
        r"^混纺的材质$",
        r"^质量(不错|保证)(的)?(亲)?$",
        r"^就是尺寸不同(哦亲)?$",
    ]
    for p in fragment_patterns:
        if re.fullmatch(p, s):
            return True

    # 非完整句：太短且不含明显动作/判断/解释词
    if len(s) <= 8:
        good_complete_markers = ["可以", "不是", "是的", "对的", "不能", "需要", "建议", "没有", "有的", "不送", "不支持"]
        if not any(m in s for m in good_complete_markers):
            return True

    return False


def trim_response_tail(text: str) -> Tuple[str, List[str]]:
    issues: List[str] = []
    original = text.strip()
    s = original

    s2 = REPEATED_ADDRESS_RE.sub("请核对一下收货地址", s)
    if s2 != s:
        s = s2
        issues.append("dedup_address_tail")

    before = s
    s = ADDRESS_TAIL_RE.sub("", s).strip(" ，。！？；、,.!?;:'\"~～")
    if s != before:
        issues.append("trim_address_tail")

    s = s.strip(" '\"")
    s = re.sub(r"(好的|嗯嗯|恩恩|哦哦|亲亲){2,}$", "", s).strip(" ，。！？；、,.!?;:'\"~～")
    s = MULTI_SPACE_RE.sub(" ", s).strip()

    if not s:
        s = original
    return s, issues


def looks_generic_template(text: str) -> bool:
    s = mask_for_key(text)
    return any(p.match(s) for p in GENERIC_TEMPLATE_PATTERNS)


def contains_hallucination_risk(text: str) -> bool:
    s = text.replace(" ", "")
    return any(h in s for h in HALLUCINATION_RISK_HINTS)


def contains_sft_hard_risk(text: str) -> bool:
    s = text.replace(" ", "")
    return any(h in s for h in SFT_HARD_RISK_HINTS)


def contains_sft_refund_promise(text: str) -> bool:
    s = text.replace(" ", "")
    return any(h in s for h in SFT_REFUND_PROMISE_HINTS)


def contains_sft_time_promise(text: str) -> bool:
    s = mask_for_key(text)
    return any(p.match(s) for p in SFT_MASKED_TIME_PROMISE_PATTERNS)


def count_high_risk_template_hits(text: str) -> int:
    s = text.replace(" ", "")
    return sum(1 for h in HIGH_RISK_TEMPLATE_HINTS if h in s)


def classify_response(context: List[str], response: str) -> str:
    s = response.replace(" ", "")
    last_user = context[-1].replace(" ", "") if context else ""

    for cat, words in CATEGORY_HINTS.items():
        if any(w in s or w in last_user for w in words):
            return cat
    return "other"


def last_turn_relevance(last_user: str, response: str) -> float:
    overlap = keyword_overlap(last_user, response)
    action_hits = sum(1 for w in ACTION_HINT_WORDS if w in response.replace(" ", ""))
    bonus = 0.0
    if any(p in response for p in GOOD_ACTION_PATTERNS):
        bonus += 0.6
    if question_strength(last_user) >= 2 and action_hits > 0:
        bonus += 0.4
    return overlap * 5.0 + min(action_hits, 4) * 0.35 + bonus


def score_context(context: List[str]) -> Tuple[float, List[str]]:
    issues: List[str] = []
    score = 0.0
    turn_n = len(context)
    if turn_n == 0:
        return -999.0, ["empty_context"]

    if GENERIC_MODE:
        for utt in context:
            store_hit_count, store_hits = count_store_specific_hits(utt)
            if "shop_name" in store_hits:
                return -999.0, ["shop_specific_context"]
            if "store_operation" in store_hits:
                return -999.0, ["store_operation_context"]
            if "store_policy" in store_hits:
                return -999.0, ["store_policy_context"]
        if has_incomplete_context_start(context):
            return -999.0, ["incomplete_context_start"]

    score += min(turn_n, 8) * 0.5
    if turn_n % 2 == 1:
        score += 1.0
    else:
        score -= 0.8
        issues.append("even_turn_context")

    last_user = context[-1]
    qs = question_strength(last_user)
    if qs > 0:
        score += 1.2
    else:
        score -= 1.8
        issues.append("weak_last_user_query")

    last_user_compact = last_user.replace(" ", "")
    if GENERIC_MODE:
        store_hit_count, store_hits = count_store_specific_hits(last_user)
        if "shop_name" in store_hits:
            return -999.0, ["shop_specific_last_user"]
        if "store_operation" in store_hits:
            return -999.0, ["store_operation_last_user"]
        if "store_policy" in store_hits:
            return -999.0, ["store_policy_last_user"]
        if WEAK_LAST_USER_RE.fullmatch(last_user_compact):
            return -999.0, ["weak_last_user_query"]

    # 单字/纯数字/极弱关键词末句，直接强惩罚
    if len(last_user_compact) <= 3 or re.fullmatch(r"<ID>|<NUM>|\d+|日期|保质期|纯棉|地址|质量", last_user_compact):
        score -= 2.2
        issues.append("very_weak_last_user_query")

    unique_ratio = len(set(context)) / max(1, len(context))
    score += unique_ratio * 0.8
    if unique_ratio < 0.6:
        score -= 0.8
        issues.append("repetitive_context")

    return score, issues


def score_positive_response(context: List[str], response: str) -> Tuple[float, List[str], str]:
    issues: List[str] = []
    score = 0.0
    last_user = context[-1] if context else ""

    response, trim_issues = trim_response_tail(response)
    issues.extend(trim_issues)
    compact = response.replace(" ", "")

    if is_garbage_text(response):
        return -999.0, ["garbage_response"], response

    if contains_role_or_chat_leak(response):
        return -999.0, ["role_or_chat_leak"], response

    if contains_non_cjk_noise(response):
        return -999.0, ["non_cjk_noise"], response

    if contains_bad_substring(response):
        return -999.0, ["bad_substring"], response

    if GENERIC_MODE:
        store_hit_count, store_hits = count_store_specific_hits(response)
        if "shop_name" in store_hits:
            return -999.0, ["shop_specific_response"], response
        if any(h in compact for h in GENERIC_RESPONSE_HARD_HINTS):
            return -999.0, ["store_operation_response"], response
        if "store_operation" in store_hits:
            return -999.0, ["store_operation_response"], response
        if "store_policy" in store_hits:
            return -999.0, ["store_policy_response"], response

    template_key = normalize_for_template(response)
    if template_key in HARD_DROP_TEMPLATE_KEYS:
        return -999.0, ["hard_drop_template"], response

    if normalize_for_template(response) in PURE_ADDRESS_REPLY_KEYS:
        return -999.0, ["pure_address_reply"], response

    if is_low_info_reply(response):
        return -999.0, ["low_info_reply"], response

    if response in context:
        return -999.0, ["response_copied_from_context"], response

    if is_fragmented_response(response):
        return -999.0, ["fragmented_response"], response

    if SFT_MODE and looks_generic_template(response):
        return -999.0, ["sft_generic_template"], response

    if SFT_MODE and contains_hallucination_risk(response):
        return -999.0, ["sft_hallucination_risk"], response

    if SFT_MODE and contains_sft_hard_risk(response):
        return -999.0, ["sft_hard_risk"], response

    if SFT_MODE and contains_sft_refund_promise(response):
        return -999.0, ["sft_refund_promise"], response

    if SFT_MODE and contains_sft_time_promise(response):
        return -999.0, ["sft_time_promise"], response

    strict_penalty_scale = 1.0 if STRICT_MODE else 0.0

    if contains_temporal_activity_hint(response):
        if SFT_MODE:
            return -999.0, ["sft_temporal_activity_hint"], response
        issues.append("temporal_activity_hint")
        score -= 2.5 + 0.7 * strict_penalty_scale

    if contains_soft_template_hint(response):
        issues.append("soft_template")
        score -= 1.8 + 0.8 * strict_penalty_scale

    n = len(compact)
    if n < 6:
        return -999.0, ["too_short"], response
    else:
        if STRICT_MODE:
            if n <= 12:
                score += 2.2
            elif n <= 22:
                score += 3.8
            elif n <= 32:
                score += 3.0
            elif n <= 40:
                score += 1.5
            elif n <= 52:
                score += 0.2
                issues.append("longish_response")
            elif n <= 70:
                score -= 1.8
                issues.append("too_long")
            else:
                score -= 4.0
                issues.append("too_long")
        elif n <= 14:
            score += 1.8
        elif n <= 28:
            score += 3.4
        elif n <= 42:
            score += 3.0
        elif n <= 60:
            score += 1.8
        elif n <= 80:
            score += 0.5
            issues.append("a_bit_long")
        else:
            score -= 2.5
            issues.append("too_long")

    relevance = last_turn_relevance(last_user, response)
    score += relevance
    if relevance < 0.6:
        score -= 3.0
        issues.append("weak_last_turn_relevance")
    elif relevance < 1.2:
        score -= 1.3
        issues.append("medium_last_turn_relevance")

    category = classify_response(context, response)
    clause_n = clause_count(response)

    if SFT_MODE and category in {"courier_info", "shipping_sla"}:
        if any(h in compact for h in SFT_SHIPPING_TEMPLATE_HINTS):
            return -999.0, ["sft_shipping_template"], response

    if STRICT_MODE:
        if clause_n >= 4 and n > 28:
            score -= 1.8
            issues.append("too_many_clauses")
        if clause_n >= 6:
            score -= 2.6
            issues.append("too_many_clauses")
    elif clause_n >= 6 and n > 40:
        score -= 1.2
        issues.append("too_many_clauses")

    if looks_generic_template(response):
        score -= 4.0 if STRICT_MODE else 3.0
        issues.append("generic_template")

    risk_hits = count_high_risk_template_hits(response)
    score -= risk_hits * (1.0 if STRICT_MODE else 0.7)
    if risk_hits >= (2 if STRICT_MODE else 3):
        issues.append("template_heavy")

    if contains_hallucination_risk(response):
        score -= 4.0 if STRICT_MODE else 3.0
        issues.append("hallucination_risk")

    if "请核对一下收货地址" in compact:
        score -= 3.0 if STRICT_MODE else 2.5
        issues.append("contains_address_tail")

    if STRICT_MODE:
        if category == "product_info":
            score -= 0.6
        elif category == "after_sale":
            score += 0.8
        elif category == "price_activity":
            score -= 1.4
        elif category == "courier_info":
            score -= 1.6
        elif category == "shipping_sla":
            score -= 1.6
        elif category == "other":
            score += 0.4
    elif category == "product_info":
        score -= 0.2
    elif category == "after_sale":
        score += 0.8
    elif category == "price_activity":
        score -= 0.8
    elif category == "courier_info":
        score -= 1.0
    elif category == "shipping_sla":
        score -= 1.0
    elif category == "other":
        score += 0.5

    unasked_count, _ = count_unasked_topic_expansions(last_user, response)
    if STRICT_MODE and unasked_count >= 3 and n > 20:
        return -999.0, issues + ["aggressive_unasked_topic_expansion"], response
    if unasked_count >= 2:
        score -= 3.6 if STRICT_MODE else 2.2
        issues.append("unasked_topic_expansion")
    elif STRICT_MODE and unasked_count == 1 and n > 24:
        score -= 1.2
        issues.append("minor_unasked_topic_expansion")

    if looks_sales_pitch(response):
        if SFT_MODE:
            return -999.0, issues + ["sft_sales_pitch"], response
        if STRICT_MODE and unasked_count >= 1:
            score -= 2.6
            issues.append("sales_pitch_expansion")
        elif STRICT_MODE and n > 28:
            score -= 1.2
            issues.append("sales_pitch_expansion")
        elif unasked_count >= 1:
            score -= 1.3
            issues.append("sales_pitch_expansion")

    if STRICT_MODE and clause_n >= 6 and (unasked_count >= 1 or "template_heavy" in issues):
        return -999.0, issues + ["template_like_multi_clause"], response

    if any(x in compact for x in ["不能", "可以的", "可以哦", "稍等我帮您", "这个不能确定", "拍的时候备注"]):
        score += 0.4

    return score, issues, response


def score_negative_response(context: List[str], response: str, pos_response: str) -> Tuple[float, List[str], str]:
    issues: List[str] = []
    response, trim_issues = trim_response_tail(response)
    issues.extend(trim_issues)
    compact = response.replace(" ", "")

    if is_garbage_text(response):
        return -999.0, ["garbage_negative"], response
    if contains_role_or_chat_leak(response):
        return -999.0, ["role_or_chat_leak_negative"], response
    if contains_non_cjk_noise(response):
        return -999.0, ["non_cjk_noise_negative"], response
    if compact == pos_response.replace(" ", ""):
        return -999.0, ["same_as_positive"], response
    if len(compact) < 2:
        return -999.0, ["too_short_negative"], response

    # 过滤太弱的负样本
    if compact in LOW_INFO_REPLIES or len(compact) <= 4:
        return -999.0, ["weak_negative"], response

    score = 1.0
    if response in context:
        score -= 1.0
        issues.append("negative_copied_from_context")
    if contains_bad_substring(response):
        score -= 1.0
        issues.append("bad_substring_negative")

    # 更偏好“像客服但答偏”的负样本
    overlap = keyword_overlap(context[-1] if context else "", response)
    if overlap > 0.15:
        score += 0.8
    if looks_generic_template(response):
        score += 0.5
    if contains_soft_template_hint(response):
        score += 0.5
    if contains_temporal_activity_hint(response):
        score += 0.3
    unasked_count, _ = count_unasked_topic_expansions(context[-1] if context else "", response)
    if unasked_count >= 1:
        score += 0.6 if STRICT_MODE else 0.4
    if looks_sales_pitch(response):
        score += 0.4

    # 太片段的负样本也不理想
    if is_fragmented_response(response):
        score -= 0.5
        issues.append("fragmented_negative")

    return score, issues, response


def parse_raw_line(line: str) -> Optional[Tuple[int, List[str], str]]:
    parts = line.rstrip("\n").split("\t")
    if len(parts) < 3:
        return None
    try:
        label = int(parts[0])
    except ValueError:
        return None
    if label not in (0, 1):
        return None
    context = [clean_utt(x) for x in parts[1:-1] if x.strip()]
    response = clean_utt(parts[-1])
    if not context or not response:
        return None
    return label, context, response


def build_raw_line(label: int, context: List[str], response: str) -> str:
    return "\t".join([str(label)] + context + [response])


def make_candidate(label: int, context: List[str], response: str, line_no: int) -> Candidate:
    context_score, ctx_issues = score_context(context)
    if label == 1:
        resp_score, resp_issues, resp_clean = score_positive_response(context, response)
    else:
        resp_score, resp_issues, resp_clean = 0.0, [], response

    context_key = " || ".join(mask_for_key(x) for x in context)
    response_key = mask_for_key(resp_clean)
    response_template_key = normalize_for_template(resp_clean)
    last_user_key = mask_for_key(context[-1]) if context else ""
    pair_key = normalize_for_template(last_user_key + " || " + resp_clean)
    category = classify_response(context, resp_clean)

    return Candidate(
        label=label,
        context=context,
        response=resp_clean,
        raw_line=build_raw_line(label, context, resp_clean),
        line_no=line_no,
        context_key=context_key,
        response_key=response_key,
        response_template_key=response_template_key,
        pair_key=pair_key,
        category=category,
        score=context_score + resp_score,
        issues=ctx_issues + resp_issues,
    )


def pick_best_positive(cands: List[Candidate]) -> Optional[Candidate]:
    if not cands:
        return None
    cands = sorted(cands, key=lambda x: (x.score, -len(x.issues), -len(x.response)), reverse=True)
    best = cands[0]
    if best.score < 0:
        return None

    hard_block_issues = {
        "hard_drop_template", "pure_address_reply", "garbage_response", "bad_substring",
        "low_info_reply", "too_short", "response_copied_from_context", "fragmented_response",
        "very_weak_last_user_query", "role_or_chat_leak", "non_cjk_noise",
        "aggressive_unasked_topic_expansion", "template_like_multi_clause",
        "shop_specific_context", "store_operation_context", "incomplete_context_start",
        "store_policy_context", "shop_specific_last_user", "store_operation_last_user",
        "store_policy_last_user", "shop_specific_response", "store_operation_response",
        "store_policy_response",
    }
    if any(i in hard_block_issues for i in best.issues):
        return None

    # 对弱问题样本进一步硬过滤
    if "weak_last_user_query" in best.issues and best.score < 4.8:
        return None

    return best


def pick_best_negative(cands: List[Candidate], pos: Candidate) -> Optional[Candidate]:
    scored: List[Candidate] = []
    for c in cands:
        neg_score, neg_issues, neg_clean = score_negative_response(pos.context, c.response, pos.response)
        if neg_score <= -999:
            continue
        cc = Candidate(
            label=0,
            context=c.context,
            response=neg_clean,
            raw_line=build_raw_line(0, c.context, neg_clean),
            line_no=c.line_no,
            context_key=c.context_key,
            response_key=mask_for_key(neg_clean),
            response_template_key=normalize_for_template(neg_clean),
            pair_key=normalize_for_template((mask_for_key(c.context[-1]) if c.context else "") + " || " + neg_clean),
            category=classify_response(c.context, neg_clean),
            score=neg_score,
            issues=c.issues + neg_issues,
        )
        scored.append(cc)
    if not scored:
        return None
    scored.sort(key=lambda x: (x.score, -len(x.issues), -len(x.response)), reverse=True)
    return scored[0]


def build_group_selection(context_key: str, pos_cands: List[Candidate], neg_cands: List[Candidate]) -> Optional[GroupSelection]:
    pos = pick_best_positive(pos_cands)
    if pos is None:
        return None
    neg = pick_best_negative(neg_cands, pos)
    if neg is None:
        return None

    group_score = pos.score + max(0.0, neg.score * 0.15)
    last_user_key = mask_for_key(pos.context[-1]) if pos.context else ""
    return GroupSelection(
        context_key=context_key,
        pos=pos,
        neg=neg,
        score=group_score,
        diversity_key=pos.response_template_key,
        pair_key=pos.pair_key,
        last_user_key=last_user_key,
        category=pos.category,
    )


def read_candidates(input_path: Path) -> Tuple[List[Candidate], Dict[str, int]]:
    stats = Counter()
    candidates: List[Candidate] = []
    with input_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stats["total_lines"] += 1
            parsed = parse_raw_line(line)
            if parsed is None:
                stats["parse_failed"] += 1
                continue
            label, context, response = parsed
            cand = make_candidate(label, context, response, line_no)
            candidates.append(cand)
            stats[f"label_{label}"] += 1
    return candidates, dict(stats)


def group_candidates(candidates: List[Candidate]) -> Tuple[List[GroupSelection], Dict[str, int]]:
    stats = Counter()
    by_context: Dict[str, Dict[int, List[Candidate]]] = defaultdict(lambda: defaultdict(list))
    for c in candidates:
        by_context[c.context_key][c.label].append(c)

    groups: List[GroupSelection] = []
    for context_key, bucket in by_context.items():
        stats["context_groups"] += 1
        pos_cands = bucket.get(1, [])
        neg_cands = bucket.get(0, [])
        if not pos_cands:
            stats["group_no_positive"] += 1
            continue
        if not neg_cands:
            stats["group_no_negative"] += 1
            continue
        g = build_group_selection(context_key, pos_cands, neg_cands)
        if g is None:
            stats["group_rejected"] += 1
            continue
        groups.append(g)
        stats["group_kept_as_candidate"] += 1

    groups.sort(key=lambda g: (g.score, len(g.pos.context), -len(g.pos.response)), reverse=True)
    return groups, dict(stats)


def build_category_caps(top_k: int) -> Dict[str, int]:
    if SFT_MODE:
        return {
            "product_info": int(top_k * 0.26),
            "after_sale": int(top_k * 0.26),
            "price_activity": int(top_k * 0.08),
            "courier_info": int(top_k * 0.06),
            "shipping_sla": int(top_k * 0.06),
            "other": int(top_k * 0.28),
        }

    if GENERIC_MODE:
        return {
            "product_info": int(top_k * 0.28),
            "after_sale": int(top_k * 0.32),
            "price_activity": int(top_k * 0.04),
            "courier_info": int(top_k * 0.03),
            "shipping_sla": int(top_k * 0.03),
            "other": int(top_k * 0.30),
        }

    if STRICT_MODE:
        # 严格模式下进一步压低高频模板类目，给更保守、更贴问题的样本更多配额。
        return {
            "product_info": int(top_k * 0.20),
            "after_sale": int(top_k * 0.28),
            "price_activity": int(top_k * 0.08),
            "courier_info": int(top_k * 0.06),
            "shipping_sla": int(top_k * 0.06),
            "other": int(top_k * 0.32),
        }

    # 默认模式更平衡：压低 product_info，增加 other / after_sale
    return {
        "product_info": int(top_k * 0.26),
        "after_sale": int(top_k * 0.24),
        "price_activity": int(top_k * 0.10),
        "courier_info": int(top_k * 0.08),
        "shipping_sla": int(top_k * 0.08),
        "other": int(top_k * 0.24),
    }


def select_diverse_topk(
    groups: List[GroupSelection],
    top_k: int,
    max_same_response_template: int = 5,
    max_same_last_user: int = 2,
    max_same_pair: int = 1,
) -> Tuple[List[GroupSelection], Dict[str, int], Counter, Counter, Counter]:
    selected: List[GroupSelection] = []
    stats = Counter()
    tpl_counter: Counter = Counter()
    user_counter: Counter = Counter()
    pair_counter: Counter = Counter()
    category_counter: Counter = Counter()
    category_caps = build_category_caps(top_k)

    leftovers: List[GroupSelection] = []
    for g in groups:
        if len(selected) >= top_k:
            break

        tpl_key = g.diversity_key or "<EMPTY>"
        user_key = g.last_user_key or "<EMPTY>"
        pair_key = g.pair_key or "<EMPTY>"
        category = g.category or "other"

        if category_counter[category] >= category_caps.get(category, top_k):
            stats["drop_category_cap"] += 1
            leftovers.append(g)
            continue
        if tpl_counter[tpl_key] >= max_same_response_template:
            stats["drop_same_response_template_cap"] += 1
            leftovers.append(g)
            continue
        if user_counter[user_key] >= max_same_last_user:
            stats["drop_same_last_user_cap"] += 1
            leftovers.append(g)
            continue
        if pair_counter[pair_key] >= max_same_pair:
            stats["drop_same_pair_cap"] += 1
            leftovers.append(g)
            continue

        selected.append(g)
        tpl_counter[tpl_key] += 1
        user_counter[user_key] += 1
        pair_counter[pair_key] += 1
        category_counter[category] += 1
        stats["selected_first_pass"] += 1

    if len(selected) < top_k:
        for g in leftovers:
            if len(selected) >= top_k:
                break
            tpl_key = g.diversity_key or "<EMPTY>"
            user_key = g.last_user_key or "<EMPTY>"
            pair_key = g.pair_key or "<EMPTY>"
            category = g.category or "other"

            if tpl_counter[tpl_key] >= max_same_response_template + 1:
                stats["drop_same_response_template_cap_relaxed"] += 1
                continue
            if user_counter[user_key] >= max_same_last_user + 1:
                stats["drop_same_last_user_cap_relaxed"] += 1
                continue
            if pair_counter[pair_key] >= max_same_pair + 1:
                stats["drop_same_pair_cap_relaxed"] += 1
                continue

            selected.append(g)
            tpl_counter[tpl_key] += 1
            user_counter[user_key] += 1
            pair_counter[pair_key] += 1
            category_counter[category] += 1
            stats["selected_second_pass"] += 1

    stats["requested_top_k"] = top_k
    stats["final_selected_groups"] = len(selected)
    return selected, dict(stats), tpl_counter, user_counter, category_counter


def write_outputs(
    selected: List[GroupSelection],
    out_dir: Path,
    prefix: str,
    source_path: Path,
    read_stats: Dict[str, int],
    group_stats: Dict[str, int],
    select_stats: Dict[str, int],
    tpl_counter: Counter,
    user_counter: Counter,
    category_counter: Counter,
) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_out = out_dir / f"{prefix}_raw.txt"
    pos_out = out_dir / f"{prefix}_positive_only.txt"
    preview_out = out_dir / f"{prefix}_preview.jsonl"
    stats_out = out_dir / f"{prefix}_stats.json"

    with raw_out.open("w", encoding="utf-8") as f_raw, \
         pos_out.open("w", encoding="utf-8") as f_pos, \
         preview_out.open("w", encoding="utf-8") as f_prev:

        for g in selected:
            f_raw.write(g.pos.raw_line + "\n")
            f_raw.write(g.neg.raw_line + "\n")
            f_pos.write(g.pos.raw_line + "\n")
            preview = {
                "score": round(g.score, 4),
                "category": g.category,
                "context": g.pos.context,
                "chosen": g.pos.response,
                "rejected": g.neg.response,
                "chosen_issues": g.pos.issues,
                "rejected_issues": g.neg.issues,
                "pos_line_no": g.pos.line_no,
                "neg_line_no": g.neg.line_no,
            }
            f_prev.write(json.dumps(preview, ensure_ascii=False) + "\n")

    stats = {
        "source_path": str(source_path),
        "output_raw_path": str(raw_out),
        "output_positive_only_path": str(pos_out),
        "output_preview_path": str(preview_out),
        "strict_mode": STRICT_MODE,
        "generic_mode": GENERIC_MODE,
        "read_stats": read_stats,
        "group_stats": group_stats,
        "select_stats": select_stats,
        "category_distribution": dict(category_counter),
        "top_response_templates": tpl_counter.most_common(80),
        "top_last_user_patterns": user_counter.most_common(80),
        "notes": {
            "raw_file_format": "保持原始 train.txt 格式；每个入选 context 输出两行：label=1 的 chosen + label=0 的 rejected。",
            "positive_only_file": "只保留 label=1，可直接给 SFT 脚本使用。",
            "dpo_compatibility": "raw 文件中保留了 1/0 配对，更方便后续 DPO 转换。",
            "sft_mode": SFT_MODE,
            "objective": [
                "优先短、准、贴最后一句用户问题",
                "强压高频模板回复",
                "强过滤幻觉式承诺和无关扩展",
                "减少快递/优惠券/退款模板污染",
                "硬过滤弱问题、片段式 chosen、软模板客服腔、历史活动话术",
                "SFT 模式下额外剔除时效承诺、营销口径、店铺化物流模板与退款承诺句",
            ],
        },
    }
    with stats_out.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    return {
        "raw": raw_out,
        "positive_only": pos_out,
        "preview": preview_out,
        "stats": stats_out,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="从淘宝原始 train.txt 中筛选更高质量 top-k 对话组")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT), help="原始 train.txt 路径")
    parser.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT_DIR), help="输出目录")
    parser.add_argument("--top_k", type=int, default=10000, help="要保留的对话组数量")
    parser.add_argument("--prefix", type=str, default="train_clean_top10000", help="输出文件名前缀")
    parser.add_argument("--max_same_response_template", type=int, default=5, help="相同回复模板最大保留数")
    parser.add_argument("--max_same_last_user", type=int, default=2, help="相同最后一句用户问题最大保留数")
    parser.add_argument("--max_same_pair", type=int, default=1, help="相同 last_user + chosen 组合最大保留数")
    parser.add_argument("--strict_mode", action="store_true", help="启用更严格的高质量筛选规则")
    parser.add_argument("--generic_mode", action="store_true", help="启用更强调泛化能力与去店铺化的筛选规则")
    parser.add_argument("--sft_mode", action="store_true", help="启用更适合 SFT 的硬过滤规则，进一步去除时效承诺和营销模板")
    args = parser.parse_args()

    global STRICT_MODE, GENERIC_MODE, SFT_MODE
    STRICT_MODE = args.strict_mode
    GENERIC_MODE = args.generic_mode
    SFT_MODE = args.sft_mode

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在：{input_path}")

    print(f"[INFO] 读取原始数据：{input_path}")
    print(f"[INFO] strict_mode: {STRICT_MODE}")
    print(f"[INFO] generic_mode: {GENERIC_MODE}")
    print(f"[INFO] sft_mode: {SFT_MODE}")
    candidates, read_stats = read_candidates(input_path)
    print(f"[INFO] 解析完成：{read_stats}")

    groups, group_stats = group_candidates(candidates)
    print(f"[INFO] 可参与排序的候选对话组数：{len(groups)}")
    print(f"[INFO] 分组统计：{group_stats}")

    selected, select_stats, tpl_counter, user_counter, category_counter = select_diverse_topk(
        groups=groups,
        top_k=args.top_k,
        max_same_response_template=args.max_same_response_template,
        max_same_last_user=args.max_same_last_user,
        max_same_pair=args.max_same_pair,
    )
    print(f"[INFO] 最终选中对话组数：{len(selected)}")
    print(f"[INFO] 选择统计：{select_stats}")
    print(f"[INFO] 类别分布：{dict(category_counter)}")

    outputs = write_outputs(
        selected=selected,
        out_dir=out_dir,
        prefix=args.prefix,
        source_path=input_path,
        read_stats=read_stats,
        group_stats=group_stats,
        select_stats=select_stats,
        tpl_counter=tpl_counter,
        user_counter=user_counter,
        category_counter=category_counter,
    )

    print("[INFO] 输出完成：")
    for k, v in outputs.items():
        print(f"  - {k}: {v}")

    print("\n[INFO] 建议：")
    print(f"1) 先抽查 {outputs['preview']} 前 200 条")
    print(f"2) 若过严，可把 --max_same_response_template 调到 6 或 7")
    print(f"3) SFT 用 {outputs['positive_only']}；DPO 用 {outputs['raw']}")


if __name__ == "__main__":
    main()

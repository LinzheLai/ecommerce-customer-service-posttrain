from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CLEAN_DIR = BASE_DIR / "data" / "cleaned"
BACKUP_ROOT = BASE_DIR / "outputs" / "data_backups"

RAW_PATH = CLEAN_DIR / "train_clean_top5000_raw.txt"
REPLACE_GROUP_COUNT = 96

Turn = Tuple[str, str]


@dataclass(frozen=True)
class RawScenario:
    family: str
    pre_turns: Sequence[Turn]
    last_user: str
    positive: str
    negative: str


WRAPPERS: Sequence[Sequence[Turn]] = (
    (("user", "你好"), ("assistant", "您好")),
    (("user", "在吗"), ("assistant", "在的")),
    (("user", "问一下"), ("assistant", "您说")),
    (("user", "麻烦看下"), ("assistant", "好的")),
)


SCENARIOS: Sequence[RawScenario] = (
    RawScenario(
        family="anti_contact",
        pre_turns=(),
        last_user="是不是得扫一扫",
        positive="不用额外操作哦，按页面操作就可以",
        negative="扫二维码加我微信说会更快哦",
    ),
    RawScenario(
        family="anti_contact",
        pre_turns=(),
        last_user="可以私下联系吗",
        positive="直接在这里留言就可以哦",
        negative="可以加我微信单独联系哦",
    ),
    RawScenario(
        family="anti_contact",
        pre_turns=(),
        last_user="怎么领取优惠",
        positive="有优惠会在页面直接显示哦",
        negative="拍下后联系我给您改价哦",
    ),
    RawScenario(
        family="anti_contact",
        pre_turns=(),
        last_user="可以发邮政吗",
        positive="可以的亲，拍的时候备注下哦",
        negative="加我微信发地址我给您单独安排哦",
    ),
    RawScenario(
        family="anti_contact",
        pre_turns=(("user", "地址写错了"), ("assistant", "您把新地址发来哦")),
        last_user="请给这个地址 <ID> 请改地址",
        positive="好的亲，请核对一下收货地址哦",
        negative="把新地址发我微信我帮您改哦",
    ),
    RawScenario(
        family="anti_contact",
        pre_turns=(("user", "我怕你们忘了"), ("assistant", "您说下您的要求")),
        last_user="好别发申通快递啊",
        positive="好的亲，这边备注一下哦",
        negative="您加我微信我单独帮您安排快递哦",
    ),
    RawScenario(
        family="anti_contact",
        pre_turns=(("user", "我查不到物流"), ("assistant", "您稍等哦")),
        last_user="亲请问快递送到哪里了",
        positive="我帮您问下哦稍等哦",
        negative="您扫二维码查一下物流哦",
    ),
    RawScenario(
        family="anti_contact",
        pre_turns=(("user", "有赠品吗"), ("assistant", "有的哦")),
        last_user="是什么礼品",
        positive="这个是随机的哦",
        negative="加我微信我发礼品图片给您哦",
    ),
    RawScenario(
        family="anti_refund",
        pre_turns=(("user", "怕当天没有余额"), ("assistant", "您看下支付方式哦")),
        last_user="怕当天没有余额怎么办",
        positive="有绑定快捷支付会从银行卡里扣的哦",
        negative="那您先退款了重新拍就行哦",
    ),
    RawScenario(
        family="anti_refund",
        pre_turns=(),
        last_user="商品已经拆封了还能退吗",
        positive="要看是否影响二次销售哦",
        negative="可以直接申请退款哦",
    ),
    RawScenario(
        family="anti_refund",
        pre_turns=(("user", "想多买一点"), ("assistant", "您看下页面活动哦")),
        last_user="怎么优惠那能不能提前给我优惠啊",
        positive="这个不能提前优惠哦",
        negative="可以的，先拍下我给您退差价哦",
    ),
    RawScenario(
        family="anti_refund",
        pre_turns=(("user", "买一也送湿巾吧"), ("assistant", "没有写赠品的话就是没有哦")),
        last_user="之前买的都有啊",
        positive="每次活动不一样哦",
        negative="那您退掉重拍就会有哦",
    ),
    RawScenario(
        family="anti_refund",
        pre_turns=(("user", "我喜欢肉粽"), ("assistant", "有几个口味哦")),
        last_user="我喜欢肉粽还有你家别的宝贝呀",
        positive="有几个口味哦，您看下需要哪种",
        negative="不喜欢的话可以退款重新拍哦",
    ),
    RawScenario(
        family="anti_refund",
        pre_turns=(("user", "赠品会一起发吗"), ("assistant", "会一起发的哦")),
        last_user="那具体是什么礼品",
        positive="这个随机的哦",
        negative="先拍下，不满意的话可以退款哦",
    ),
    RawScenario(
        family="anti_refund",
        pre_turns=(("user", "质量怎么样"), ("assistant", "您说下想了解哪点")),
        last_user="送夹子吗",
        positive="不送夹子哦",
        negative="不合适的话可以退款哦",
    ),
    RawScenario(
        family="anti_refund",
        pre_turns=(("user", "洗过会不会有问题"), ("assistant", "您说下具体情况哦")),
        last_user="如果洗过之后起球变形对不齐怎么处理",
        positive="如果确实有问题，这边会处理哦",
        negative="您先申请退款就可以哦",
    ),
    RawScenario(
        family="anti_resend",
        pre_turns=(("user", "默认发什么快递"), ("assistant", "会按地址安排哦")),
        last_user="那可以发邮政吗",
        positive="可以的亲，拍的时候备注下哦",
        negative="我给您补发一个邮政件哦",
    ),
    RawScenario(
        family="anti_resend",
        pre_turns=(("user", "快递别弄错了"), ("assistant", "您说下您的要求")),
        last_user="好别发申通快递啊",
        positive="好的亲，这边备注一下哦",
        negative="我这边给您重新补发一单别的快递哦",
    ),
    RawScenario(
        family="anti_resend",
        pre_turns=(("user", "地址好像写错了"), ("assistant", "您把正确地址发来哦")),
        last_user="请给这个地址 <ID> 请改地址",
        positive="好的亲，请核对一下收货地址哦",
        negative="我先给您补发到新地址哦",
    ),
    RawScenario(
        family="anti_resend",
        pre_turns=(("user", "差价怎么算"), ("assistant", "您说下具体情况哦")),
        last_user="算了你们给我们补差价吧十个粽子减去二十个咸鸭蛋",
        positive="我这边给您看下差价哦",
        negative="我给您补发点赠品补差价哦",
    ),
    RawScenario(
        family="anti_resend",
        pre_turns=(
            ("user", "我想多买一些"),
            ("assistant", "好的哦，您说下数量"),
            ("user", "十几份都有货吗"),
            ("assistant", "有的哦"),
        ),
        last_user="那申请领导的结果呢这样可以吗我也不要优惠券就买17份另送我一份可以吗",
        positive="这个要按活动规则来哦",
        negative="可以的，我给您多补发一份哦",
    ),
    RawScenario(
        family="anti_resend",
        pre_turns=(("user", "洗后会不会变形"), ("assistant", "您说下具体情况哦")),
        last_user="如果洗过之后起球变形对不齐怎么处理",
        positive="如果确实有问题，这边会处理哦",
        negative="我直接给您补发一件哦",
    ),
    RawScenario(
        family="anti_resend",
        pre_turns=(("user", "快递一直不动"), ("assistant", "我帮您看下")),
        last_user="亲请问快递送到哪里了",
        positive="我帮您问下哦稍等哦",
        negative="我给您重新补发一个快递哦",
    ),
    RawScenario(
        family="anti_resend",
        pre_turns=(("user", "发什么快递"), ("assistant", "随机安排哦")),
        last_user="发什么快递今天发货不",
        positive="快递随机的哦，发货时间以页面为准",
        negative="今天一定给您补发出去哦",
    ),
)


def parse_raw_line(line: str) -> Tuple[int, List[str], str]:
    parts = line.rstrip("\n").split("\t")
    if len(parts) < 3:
        raise ValueError(f"bad raw line: {line!r}")
    label = int(parts[0])
    context = parts[1:-1]
    response = parts[-1]
    return label, context, response


def build_raw_line(label: int, context: Sequence[str], response: str) -> str:
    return "\t".join([str(label), *context, response])


def backup_file(path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = BACKUP_ROOT / f"cleaned_antibias_{stamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, backup_dir / path.name)
    return backup_dir


def risk_score(context: Sequence[str], response: str) -> float:
    score = 0.0
    text = " ".join(context) + " " + response
    weighted_terms = {
        "微信": 7.0,
        "二维码": 6.0,
        "联系我": 4.5,
        "加我": 4.0,
        "申请退款": 6.0,
        "退款": 4.5,
        "退货": 3.0,
        "补发": 5.0,
        "换货": 4.0,
        "重发": 4.0,
        "24小时": 3.0,
        "48小时": 3.0,
        "今天发货": 2.5,
        "明天发货": 2.5,
        "厂家直销": 4.0,
        "质量有保证": 3.0,
        "放心购买": 2.5,
        "仓库": 1.5,
    }
    for term, weight in weighted_terms.items():
        if term in text:
            score += weight
    if len(response) > 24:
        score += 0.8
    if response.count("亲") >= 4:
        score += 0.5
    return score


def build_context(wrapper: Sequence[Turn], pre_turns: Sequence[Turn], last_user: str) -> List[str]:
    turns = list(wrapper) + list(pre_turns) + [("user", last_user)]
    if turns[-1][0] != "user":
        raise ValueError("last turn must be user")
    return [text for _, text in turns]


def build_template_pairs() -> Tuple[List[Tuple[str, str]], Dict[str, int]]:
    pairs: List[Tuple[str, str]] = []
    family_counter: Dict[str, int] = {}
    for scenario in SCENARIOS:
        for wrapper in WRAPPERS:
            context = build_context(wrapper, scenario.pre_turns, scenario.last_user)
            pos = build_raw_line(1, context, scenario.positive)
            neg = build_raw_line(0, context, scenario.negative)
            pairs.append((pos, neg))
            family_counter[scenario.family] = family_counter.get(scenario.family, 0) + 1
    return pairs, family_counter


def group_lines(lines: Iterable[str]) -> List[Tuple[Tuple[str, ...], List[str]]]:
    buckets: Dict[Tuple[str, ...], List[str]] = {}
    order: List[Tuple[str, ...]] = []
    for line in lines:
        _, context, _ = parse_raw_line(line)
        key = tuple(context)
        if key not in buckets:
            buckets[key] = []
            order.append(key)
        buckets[key].append(line.rstrip("\n"))
    return [(key, buckets[key]) for key in order]


def positive_response(group_lines: Sequence[str]) -> str:
    for line in group_lines:
        label, _, response = parse_raw_line(line)
        if label == 1:
            return response
    return ""


def main() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(RAW_PATH)

    original_lines = RAW_PATH.read_text(encoding="utf-8").splitlines()
    groups = group_lines(original_lines)
    if len(groups) < REPLACE_GROUP_COUNT:
        raise ValueError(f"not enough groups to replace: {len(groups)}")

    template_pairs, family_counter = build_template_pairs()
    if len(template_pairs) != REPLACE_GROUP_COUNT:
        raise ValueError(f"unexpected template pair count: {len(template_pairs)}")

    scored_groups: List[Tuple[float, int]] = []
    for idx, (context_key, raw_group_lines) in enumerate(groups):
        pos = positive_response(raw_group_lines)
        scored_groups.append((risk_score(list(context_key), pos), idx))
    scored_groups.sort(key=lambda item: (item[0], item[1]), reverse=True)

    replace_indices = {idx for _, idx in scored_groups[:REPLACE_GROUP_COUNT]}
    kept_groups = [raw_group_lines for idx, (_, raw_group_lines) in enumerate(groups) if idx not in replace_indices]

    new_lines: List[str] = []
    for raw_group_lines in kept_groups:
        new_lines.extend(raw_group_lines)
    for pos, neg in template_pairs:
        new_lines.append(pos)
        new_lines.append(neg)

    backup_dir = backup_file(RAW_PATH)
    RAW_PATH.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

    summary = {
        "backup_dir": str(backup_dir),
        "raw_path": str(RAW_PATH),
        "old_line_count": len(original_lines),
        "new_line_count": len(new_lines),
        "old_group_count": len(groups),
        "new_group_count": len(group_lines(new_lines)),
        "replaced_group_count": REPLACE_GROUP_COUNT,
        "template_family_counts": family_counter,
        "top_removed_scores": [score for score, _ in scored_groups[:10]],
    }
    summary_path = backup_dir / "cleaned_antibias_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

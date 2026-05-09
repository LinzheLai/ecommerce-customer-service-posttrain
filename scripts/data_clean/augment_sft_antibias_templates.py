from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed_5000"
BACKUP_ROOT = BASE_DIR / "outputs" / "data_backups"

SYSTEM_PROMPT = "你是电商客服。只回答当前最后一句用户问题，不要扩展其他信息。如果历史对话里没有明确信息，只能给出简短、保守回复"
TASK_PREFIX = "请根据下面的电商客服历史对话，生成下一句合适的客服回复。"

TRAIN_SFT_PATH = PROCESSED_DIR / "taobao_sft_train.json"
DEV_SFT_PATH = PROCESSED_DIR / "taobao_sft_dev.json"
TEST_SFT_PATH = PROCESSED_DIR / "taobao_sft_test.json"
TRAIN_MSG_PATH = PROCESSED_DIR / "taobao_messages_train.json"
DEV_MSG_PATH = PROCESSED_DIR / "taobao_messages_dev.json"
TEST_MSG_PATH = PROCESSED_DIR / "taobao_messages_test.json"

GENERIC_DEV_SFT_PATH = PROCESSED_DIR / "taobao_sft_generic_dev.json"
GENERIC_TEST_SFT_PATH = PROCESSED_DIR / "taobao_sft_generic_test.json"
GENERIC_DEV_MSG_PATH = PROCESSED_DIR / "taobao_messages_generic_dev.json"
GENERIC_TEST_MSG_PATH = PROCESSED_DIR / "taobao_messages_generic_test.json"

TRAIN_REPLACE_COUNT = 96

Turn = Tuple[str, str]


@dataclass(frozen=True)
class Scenario:
    family: str
    pre_turns: Sequence[Turn]
    train_user: str
    dev_user: str
    test_user: str
    train_output: str
    dev_output: str
    test_output: str


TRAIN_WRAPPERS: Sequence[Sequence[Turn]] = (
    (("user", "你好"), ("assistant", "您好")),
    (("user", "在吗"), ("assistant", "在的")),
    (("user", "问一下"), ("assistant", "您说")),
    (("user", "麻烦看下"), ("assistant", "好的")),
)

DEV_WRAPPERS: Sequence[Sequence[Turn]] = (
    (("user", "你好"), ("assistant", "您好")),
    (("user", "我再确认一下"), ("assistant", "好的")),
)

TEST_WRAPPERS: Sequence[Sequence[Turn]] = (
    (("user", "亲我想问一下"), ("assistant", "您说")),
    (("user", "问下"), ("assistant", "好的亲")),
)


SCENARIOS: Sequence[Scenario] = (
    Scenario(
        family="anti_contact",
        pre_turns=(),
        train_user="要加微信吗",
        dev_user="需要加微信吗",
        test_user="是不是得加微信",
        train_output="不用加微信哦，直接在这里说就可以",
        dev_output="不用加微信哦，直接在这里沟通就行",
        test_output="不用加微信哦，直接在这里留言就可以",
    ),
    Scenario(
        family="anti_contact",
        pre_turns=(),
        train_user="是不是要扫二维码",
        dev_user="需要扫二维码吗",
        test_user="是不是得扫一扫",
        train_output="不用扫二维码哦，按页面操作就可以",
        dev_output="不用扫二维码哦，直接按页面提示操作就行",
        test_output="不用扫一扫哦，直接在这里下单就可以",
    ),
    Scenario(
        family="anti_contact",
        pre_turns=(),
        train_user="可以私下联系吗",
        dev_user="能不能私下联系",
        test_user="要单独联系吗",
        train_output="直接在这里留言就可以哦",
        dev_output="直接在这里沟通就可以哦",
        test_output="不用单独联系哦，直接在这里说就行",
    ),
    Scenario(
        family="anti_contact",
        pre_turns=(),
        train_user="怎么领取优惠",
        dev_user="优惠是怎么领",
        test_user="优惠怎么拿",
        train_output="有优惠会在页面直接显示哦",
        dev_output="有优惠会直接在页面显示哦",
        test_output="页面有优惠会直接显示哦",
    ),
    Scenario(
        family="anti_contact",
        pre_turns=(),
        train_user="可以发邮政吗",
        dev_user="邮政能发吗",
        test_user="可以走邮政吗",
        train_output="可以的亲，拍的时候备注下哦",
        dev_output="可以的哦，下单时备注一下就行",
        test_output="可以的亲，拍下时备注一下哦",
    ),
    Scenario(
        family="anti_contact",
        pre_turns=(("user", "地址写错了"), ("assistant", "您把新地址发来哦")),
        train_user="请给这个地址 <ID> 请改地址",
        dev_user="地址改成 <ID> 可以吗",
        test_user="麻烦改到这个地址 <ID>",
        train_output="好的亲，请核对一下收货地址哦",
        dev_output="可以的亲，请核对一下新地址哦",
        test_output="好的亲，这边按新地址备注哦",
    ),
    Scenario(
        family="anti_contact",
        pre_turns=(("user", "我怕你们忘了"), ("assistant", "您说下您的要求")),
        train_user="好别发申通快递啊",
        dev_user="别发申通可以吗",
        test_user="申通不要发哦",
        train_output="好的亲，这边给您备注一下哦",
        dev_output="可以的亲，这边备注下哦",
        test_output="好的亲，这边帮您备注不要申通哦",
    ),
    Scenario(
        family="anti_contact",
        pre_turns=(("user", "有小礼物吗"), ("assistant", "有的哦")),
        train_user="是什么礼品",
        dev_user="礼品是什么",
        test_user="送的是什么呀",
        train_output="礼品以实际收到为准哦",
        dev_output="这个以实际收到为准哦",
        test_output="赠品以实际收到为准哦",
    ),
    Scenario(
        family="anti_refund",
        pre_turns=(),
        train_user="亲请问快递送到哪里了",
        dev_user="快递到哪里了",
        test_user="物流现在到哪了",
        train_output="我帮您问下哦，稍等",
        dev_output="我帮您查一下哦，稍等",
        test_output="我帮您问一下哦，稍等下",
    ),
    Scenario(
        family="anti_refund",
        pre_turns=(("user", "我账户里余额可能不够"), ("assistant", "您先按正常流程付款就行")),
        train_user="怕当天没有余额怎么办",
        dev_user="如果当天余额不够怎么办",
        test_user="万一那天余额不够呢",
        train_output="有绑定快捷支付会从银行卡里扣的哦",
        dev_output="如果绑定了快捷支付，会从银行卡里扣哦",
        test_output="有快捷支付的话会从银行卡扣款哦",
    ),
    Scenario(
        family="anti_refund",
        pre_turns=(),
        train_user="商品已经拆封了还能退吗",
        dev_user="拆封了还能退吗",
        test_user="打开过了还能退吗",
        train_output="要看是否影响二次销售哦",
        dev_output="要看是否影响二次销售呢",
        test_output="这个要看是否影响二次销售哦",
    ),
    Scenario(
        family="anti_refund",
        pre_turns=(("user", "想多买一点"), ("assistant", "您看下页面活动哦")),
        train_user="怎么优惠那能不能提前给我优惠啊",
        dev_user="能不能先给我点优惠",
        test_user="可以提前优惠吗",
        train_output="这个不能提前优惠哦",
        dev_output="这个暂时不能提前优惠哦",
        test_output="不好意思哦，这个不能提前优惠",
    ),
    Scenario(
        family="anti_refund",
        pre_turns=(("user", "买一也送湿巾吧"), ("assistant", "没有写赠品的话就是没有哦")),
        train_user="之前买的都有啊",
        dev_user="我之前买的时候都有",
        test_user="以前买都有的呀",
        train_output="每次活动不一样哦",
        dev_output="每次活动都不一样哦",
        test_output="这个每次活动不一样哦",
    ),
    Scenario(
        family="anti_refund",
        pre_turns=(("user", "我喜欢肉粽"), ("assistant", "有几个口味哦")),
        train_user="我喜欢肉粽还有你家别的宝贝呀",
        dev_user="我喜欢肉粽还有别的口味吗",
        test_user="我喜欢肉的还有别的吗",
        train_output="有几个口味哦，您看下需要哪种",
        dev_output="有几个口味的，您看下喜欢哪种哦",
        test_output="有几种口味哦，您看下要哪种",
    ),
    Scenario(
        family="anti_refund",
        pre_turns=(("user", "有赠品吗"), ("assistant", "有的哦")),
        train_user="是什么礼品",
        dev_user="礼品具体是什么",
        test_user="送的礼品是什么",
        train_output="这个是随机的哦",
        dev_output="这个是随机赠送的哦",
        test_output="赠品是随机的哦",
    ),
    Scenario(
        family="anti_refund",
        pre_turns=(("user", "不下水我也看不出来"), ("assistant", "您说下您的担心点")),
        train_user="如果洗过之后起球变形对不齐怎么处理",
        dev_user="要是洗后起球变形怎么办",
        test_user="如果洗后变形起球怎么处理",
        train_output="如果确实有这种情况，提供照片这边会处理哦",
        dev_output="如果有这种情况，提供照片这边会处理哦",
        test_output="如果确实有这种情况，拍照给这边处理哦",
    ),
    Scenario(
        family="anti_resend",
        pre_turns=(("user", "我想确认一下"), ("assistant", "您说下具体问题")),
        train_user="发什么快递今天发货不",
        dev_user="发什么快递今天能发吗",
        test_user="今天发不发，走什么快递",
        train_output="快递随机发哦，今天不一定能发",
        dev_output="快递是随机安排的哦，今天不一定能发",
        test_output="快递随机安排哦，今天不一定能发出",
    ),
    Scenario(
        family="anti_resend",
        pre_turns=(),
        train_user="可以发邮政吗",
        dev_user="邮政能发吗",
        test_user="能发邮政吗",
        train_output="可以的亲，拍的时候备注下哦",
        dev_output="可以的哦，下单备注一下就行",
        test_output="可以的亲，备注一下就可以",
    ),
    Scenario(
        family="anti_resend",
        pre_turns=(("user", "快递别弄错了"), ("assistant", "您说下您的要求")),
        train_user="好别发申通快递啊",
        dev_user="别发申通可以吗",
        test_user="申通不要发哈",
        train_output="好的亲，这边备注一下哦",
        dev_output="可以的亲，这边帮您备注哦",
        test_output="好的亲，给您备注不要申通哦",
    ),
    Scenario(
        family="anti_resend",
        pre_turns=(("user", "地址好像写错了"), ("assistant", "您把正确地址发来哦")),
        train_user="请给这个地址 <ID> 请改地址",
        dev_user="帮我改成这个地址 <ID>",
        test_user="地址改到 <ID> 哦",
        train_output="好的亲，请核对一下收货地址哦",
        dev_output="可以的亲，请核对一下新地址哦",
        test_output="好的亲，这边按新地址给您备注哦",
    ),
    Scenario(
        family="anti_resend",
        pre_turns=(("user", "差价怎么算"), ("assistant", "您说下具体情况哦")),
        train_user="算了你们给我们补差价吧十个粽子减去二十个咸鸭蛋",
        dev_user="那你们给我补下差价吧",
        test_user="这个差价怎么补",
        train_output="我这边给您看下差价哦",
        dev_output="我这边给您核一下差价哦",
        test_output="我帮您看下差价怎么处理哦",
    ),
    Scenario(
        family="anti_resend",
        pre_turns=(
            ("user", "我想多买一些"),
            ("assistant", "好的哦，您说下数量"),
            ("user", "十几份都有货吗"),
            ("assistant", "有的哦"),
        ),
        train_user="那申请领导的结果呢这样可以吗我也不要优惠券就买17份另送我一份可以吗",
        dev_user="那领导那边怎么说我买17份能不能多送一份",
        test_user="那申请的结果出来了吗十几份能不能多送一份",
        train_output="这个要按活动规则来哦",
        dev_output="这个要按活动规则处理哦",
        test_output="这个要按活动规则看哦",
    ),
    Scenario(
        family="anti_resend",
        pre_turns=(("user", "多送点零食吧"), ("assistant", "赠品以实际情况为准哦")),
        train_user="嗯家里小孩子多送点别的小吃孩子爱吃的话以后会经常光顾咱家小店的",
        dev_user="孩子多能不能多送点别的小吃",
        test_user="能不能多送点别的零食给孩子",
        train_output="赠品是随机的哦，这个不能指定",
        dev_output="赠品是随机的哦，不能指定哈",
        test_output="赠品随机发哦，这个没法指定",
    ),
    Scenario(
        family="anti_resend",
        pre_turns=(("user", "质量怎样"), ("assistant", "您说下想了解哪点")),
        train_user="送夹子吗",
        dev_user="会送夹子吗",
        test_user="有送夹子吗",
        train_output="不送夹子哦",
        dev_output="没有夹子哦",
        test_output="这个不送夹子哦",
    ),
)


def load_json(path: Path) -> List[Dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: List[Dict]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def build_input(turns: Sequence[Turn]) -> str:
    lines = [TASK_PREFIX, ""]
    for role, text in turns:
        prefix = "用户" if role == "user" else "客服"
        lines.append(f"{prefix}：{text}")
    lines.append("客服：")
    return "\n".join(lines)


def build_sft_sample(turns: Sequence[Turn], output: str) -> Dict:
    return {
        "instruction": SYSTEM_PROMPT,
        "input": build_input(turns),
        "output": output,
    }


def build_messages_sample(turns: Sequence[Turn], output: str) -> Dict:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for role, text in turns:
        messages.append({"role": role, "content": text})
    messages.append({"role": "assistant", "content": output})
    return {"messages": messages}


def build_turns(wrapper: Sequence[Turn], pre_turns: Sequence[Turn], last_user: str) -> List[Turn]:
    turns = list(wrapper) + list(pre_turns) + [("user", last_user)]
    if turns[-1][0] != "user":
        raise ValueError("last turn must be user")
    return turns


def generate_split_templates(split: str) -> Tuple[List[Dict], List[Dict], Dict[str, int]]:
    if split == "train":
        wrappers = TRAIN_WRAPPERS
    elif split == "dev":
        wrappers = DEV_WRAPPERS
    elif split == "test":
        wrappers = TEST_WRAPPERS
    else:
        raise ValueError(split)

    sft_records: List[Dict] = []
    msg_records: List[Dict] = []
    family_counter: Dict[str, int] = {}

    for scenario in SCENARIOS:
        last_user = getattr(scenario, f"{split}_user")
        output = getattr(scenario, f"{split}_output")
        for wrapper in wrappers:
            turns = build_turns(wrapper, scenario.pre_turns, last_user)
            sft_records.append(build_sft_sample(turns, output))
            msg_records.append(build_messages_sample(turns, output))
            family_counter[scenario.family] = family_counter.get(scenario.family, 0) + 1

    return sft_records, msg_records, family_counter


def risk_score(record: Dict) -> float:
    output = record.get("output", "")
    input_text = record.get("input", "")
    score = 0.0
    output_weights = {
        "申请退款": 6.0,
        "退款": 4.0,
        "补发": 4.5,
        "换货": 4.0,
        "24小时": 3.5,
        "48小时": 3.5,
        "今天发货": 3.0,
        "明天发货": 3.0,
        "厂家直销": 4.0,
        "质量有保证": 3.0,
        "放心购买": 2.5,
    }
    input_weights = {
        "仓库": 1.5,
        "安全放心": 1.0,
        "工厂直销": 1.0,
    }
    for term, weight in output_weights.items():
        if term in output:
            score += weight
    for term, weight in input_weights.items():
        if term in input_text:
            score += weight
    if len(output) > 24:
        score += 0.8
    if output.count("亲") >= 4:
        score += 0.5
    return score


def backup_files(paths: Sequence[Path]) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = BACKUP_ROOT / f"antibias_templates_{stamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        shutil.copy2(path, backup_dir / path.name)
    return backup_dir


def main() -> None:
    required = [
        TRAIN_SFT_PATH,
        DEV_SFT_PATH,
        TEST_SFT_PATH,
        TRAIN_MSG_PATH,
        DEV_MSG_PATH,
        TEST_MSG_PATH,
        GENERIC_DEV_SFT_PATH,
        GENERIC_TEST_SFT_PATH,
        GENERIC_DEV_MSG_PATH,
        GENERIC_TEST_MSG_PATH,
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"missing required file: {path}")

    train_sft = load_json(TRAIN_SFT_PATH)
    train_msg = load_json(TRAIN_MSG_PATH)
    generic_dev_sft = load_json(GENERIC_DEV_SFT_PATH)
    generic_test_sft = load_json(GENERIC_TEST_SFT_PATH)
    generic_dev_msg = load_json(GENERIC_DEV_MSG_PATH)
    generic_test_msg = load_json(GENERIC_TEST_MSG_PATH)

    if len(train_sft) != len(train_msg):
        raise ValueError("train sft/messages count mismatch")

    train_tpl_sft, train_tpl_msg, train_tpl_stats = generate_split_templates("train")
    dev_tpl_sft, dev_tpl_msg, dev_tpl_stats = generate_split_templates("dev")
    test_tpl_sft, test_tpl_msg, test_tpl_stats = generate_split_templates("test")

    if len(train_tpl_sft) != TRAIN_REPLACE_COUNT:
        raise ValueError(f"unexpected train template count: {len(train_tpl_sft)}")

    scored = [(risk_score(rec), idx) for idx, rec in enumerate(train_sft)]
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    replace_indices = {idx for _, idx in scored[:TRAIN_REPLACE_COUNT]}

    kept_train_sft = [rec for idx, rec in enumerate(train_sft) if idx not in replace_indices]
    kept_train_msg = [rec for idx, rec in enumerate(train_msg) if idx not in replace_indices]

    new_train_sft = kept_train_sft + train_tpl_sft
    new_train_msg = kept_train_msg + train_tpl_msg
    new_dev_sft = generic_dev_sft + dev_tpl_sft
    new_dev_msg = generic_dev_msg + dev_tpl_msg
    new_test_sft = generic_test_sft + test_tpl_sft
    new_test_msg = generic_test_msg + test_tpl_msg

    backup_dir = backup_files([
        TRAIN_SFT_PATH,
        DEV_SFT_PATH,
        TEST_SFT_PATH,
        TRAIN_MSG_PATH,
        DEV_MSG_PATH,
        TEST_MSG_PATH,
    ])

    save_json(TRAIN_SFT_PATH, new_train_sft)
    save_json(DEV_SFT_PATH, new_dev_sft)
    save_json(TEST_SFT_PATH, new_test_sft)
    save_json(TRAIN_MSG_PATH, new_train_msg)
    save_json(DEV_MSG_PATH, new_dev_msg)
    save_json(TEST_MSG_PATH, new_test_msg)

    summary = {
        "backup_dir": str(backup_dir),
        "train": {
            "old_count": len(train_sft),
            "new_count": len(new_train_sft),
            "replaced_count": TRAIN_REPLACE_COUNT,
            "template_family_counts": train_tpl_stats,
        },
        "dev": {
            "old_count": len(load_json(backup_dir / DEV_SFT_PATH.name)),
            "generic_base_count": len(generic_dev_sft),
            "template_added": len(dev_tpl_sft),
            "new_count": len(new_dev_sft),
            "template_family_counts": dev_tpl_stats,
        },
        "test": {
            "old_count": len(load_json(backup_dir / TEST_SFT_PATH.name)),
            "generic_base_count": len(generic_test_sft),
            "template_added": len(test_tpl_sft),
            "new_count": len(new_test_sft),
            "template_family_counts": test_tpl_stats,
        },
    }

    summary_path = backup_dir / "antibias_template_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

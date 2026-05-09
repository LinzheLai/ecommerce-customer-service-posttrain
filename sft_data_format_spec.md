# 电商客服 SFT 稳定数据格式规范

## 目标

降低 LLaMA Factory / TRL 在 tokenization 阶段出现的：

- `Mismatch between tokenized prompt and the start of tokenized prompt+completion`

## 推荐的 alpaca-SFT 规范

每条样本固定为：

```json
{
  "instruction": "你是电商平台客服，请基于历史对话给出准确、礼貌、可执行的回复。",
  "input": "请根据下面的电商客服历史对话，生成下一句合适的客服回复。\n\n用户：……\n客服：……\n用户：……\n客服：",
  "output": "……"
}
```

## 强约束

1. `instruction` 固定，不要在不同样本间频繁变化。
2. `input` 必须以 **`客服：`** 结尾，作为 assistant 回复起始边界。
3. `output` 只保留客服回复正文，不再重复 `客服：`。
4. `output` 不允许以空格、换行、制表符开头。
5. 统一中文标点与空白：
   - `:` → `：`
   - 连续空白压缩为一个
   - 删除行尾空格
   - 统一全角/半角兼容字符（NFKC）
6. 占位符统一：
   - `<ID>` 保留成统一形式，不要混成 `< id >`、`<Id>` 等多个版本。
7. 对话体内角色名固定：
   - 只使用 `用户：`
   - 只使用 `客服：`
   - 不混用 `user:` / `assistant:` / `买家：` / `卖家：`

## 为什么这样更稳

tokenizer 最容易在 “prompt 最后几个字符 + answer 第一个字符” 的边界重切分。
把 `input` 明确停在 `客服：`，再让 `output` 仅包含回复正文，能显著减少边界漂移。

---

## 推荐的 messages 规范

```json
{
  "messages": [
    {"role": "system", "content": "你是电商平台客服，请基于历史对话给出准确、礼貌、可执行的回复。"},
    {"role": "user", "content": "……"},
    {"role": "assistant", "content": "……"},
    {"role": "user", "content": "……"},
    {"role": "assistant", "content": "目标回复"}
  ]
}
```

## messages 版本约束

1. 第一条固定为 `system`。
2. 之后必须严格交替：
   - `user`
   - `assistant`
   - `user`
   - `assistant`
3. 最后一条 `assistant` 就是监督目标。
4. 不要把整段历史对话原封不动塞进一条 `user` 里；要拆成多轮。

## 当前数据中已观察到的高风险点

- 存在半角冒号 `:`
- 存在 `3 - 5天` 这种带空格连接符
- 存在 `<ID>` 占位符
- 存在中英混排与英文缩写（如 `EMS`）

这些不一定非法，但建议统一清洗以减少分词边界波动。
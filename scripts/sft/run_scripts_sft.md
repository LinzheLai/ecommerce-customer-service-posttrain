1. 先跑已有脚本生成同一批 query 的 base 和 SFT 输出：

bash runs/infer/run_compare_batch_infer_qwen3_8b.sh

2. 然后生成 Markdown 对比表：

python scripts/sft/export_before_after_compare.py \
  --base_json outputs/base/batch_infer/taobao_messages_test_random20_base.json \
  --sft_json outputs/sft/batch_infer/taobao_messages_test_random20_sft.json \
  --output_md outputs/compare/sft_before_after.md \
  --keyword 退货


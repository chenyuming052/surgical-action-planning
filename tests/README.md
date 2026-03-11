建议最先写的测试：
1. registry split 无泄漏
2. Cholec80 phase 对齐后与 1fps 帧数一致
3. Cholec80-CVS 畸形区间和越界区间统计为 3 / 63
4. Endoscapes test 通过文件系统补齐 228 帧
5. one batch 的 collate 字段 shape 正确
6. hazard loss 对 censor / uncensor 都数值稳定

"""
eval/t_eval.py —— 检索评测（离线、不花 token）
============================================
衡量"检索"质量：把每个问题丢给 HybridRAG，看期望的文档是否进了 top-5。
指标：R@1 / R@3 / R@5（命中率）和 MRR@5（平均倒数排名，上限 1.0）。
每条题只计一次倒数排名，避免重复累加（这是 RAG 评测常见 bug，这里已修正）。

用法：python -m eval.t_eval
注意：CASES 里的 gold 要改成你 corpus/ 里文件名的真实关键词。
"""

import os
from collections import defaultdict

from rag.retriever import HybridRAG
from rag.ingest import ingest_corpus

CORPUS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "corpus")
KS = (1, 3, 5)

# 样例 24 题（按 5 个部门分组）。gold = 期望命中文件名的包含关键词。
# 接入你自己的语料后改 gold；按相同格式补 8 条即可凑满 32 题。
CASES = [
    # HR
    {"dept": "HR", "query": "年假有几天？", "gold": "hr"},
    {"dept": "HR", "query": "怎么请事假？", "gold": "hr"},
    {"dept": "HR", "query": "试用期多长？", "gold": "hr"},
    {"dept": "HR", "query": "五险一金怎么交？", "gold": "hr"},
    {"dept": "HR", "query": "加班费怎么算？", "gold": "hr"},
    {"dept": "HR", "query": "转正流程是什么？", "gold": "hr"},
    # IT
    {"dept": "IT", "query": "WiFi 连不上怎么办？", "gold": "it"},
    {"dept": "IT", "query": "怎么重置邮箱密码？", "gold": "it"},
    {"dept": "IT", "query": "VPN 怎么配置？", "gold": "it"},
    {"dept": "IT", "query": "电脑蓝屏了怎么处理？", "gold": "it"},
    {"dept": "IT", "query": "打印机无法打印怎么办？", "gold": "it"},
    {"dept": "IT", "query": "内网访问不了怎么办？", "gold": "it"},
    # 合规-PIPL
    {"dept": "合规-PIPL", "query": "处理个人信息要遵守什么原则？", "gold": "pip"},
    {"dept": "合规-PIPL", "query": "发生个人信息泄露怎么处理？", "gold": "pip"},
    {"dept": "合规-PIPL", "query": "员工隐私数据怎么保护？", "gold": "pip"},
    {"dept": "合规-PIPL", "query": "谁负责个人信息保护？", "gold": "pip"},
    # 合规-内部
    {"dept": "合规-内部", "query": "报销流程怎么走？", "gold": "policy"},
    {"dept": "合规-内部", "query": "出差标准是多少？", "gold": "policy"},
    {"dept": "合规-内部", "query": "合同审批找谁？", "gold": "policy"},
    {"dept": "合规-内部", "query": "报销多久到账？", "gold": "policy"},
    # 产品-Python
    {"dept": "产品-Python", "query": "怎么用 pandas 读 csv？", "gold": "python"},
    {"dept": "产品-Python", "query": "Python 怎么发 HTTP 请求？", "gold": "python"},
    {"dept": "产品-Python", "query": "怎么写单元测试？", "gold": "python"},
    {"dept": "产品-Python", "query": "虚拟环境怎么建？", "gold": "python"},
]


def run_eval():
    rag = HybridRAG()
    ingest_corpus(rag, CORPUS_DIR)
    if not rag.docs:
        print("[eval] corpus 为空，无法评测。请先放文档到 corpus/")
        return

    hit = defaultdict(lambda: {k: 0 for k in KS})
    overall_hit = {k: 0 for k in KS}
    mrr = defaultdict(float)
    overall_mrr = 0.0
    n_dept = defaultdict(int)
    misses = []

    for c in CASES:
        n_dept[c["dept"]] += 1
        sources = [h["source"].lower() for h in rag.retrieve(c["query"], top_k=5)]
        gold = c["gold"].lower()
        first_rank = None
        for rank, s in enumerate(sources, start=1):
            if gold in s:
                first_rank = rank
                break
        found = first_rank is not None
        for k in KS:
            if found and first_rank <= k:
                hit[c["dept"]][k] += 1
                overall_hit[k] += 1
        if found:
            mrr[c["dept"]] += 1.0 / first_rank
            overall_mrr += 1.0 / first_rank
        else:
            misses.append((c["dept"], c["query"], sources[:3]))

    n = len(CASES)
    print("\n=== 检索评测结果（R@k / MRR@5）===")
    print(f"{'部门':<12}{'R@1':>8}{'R@3':>8}{'R@5':>8}{'MRR@5':>9}")
    for dept in n_dept:
        print(f"{dept:<12}"
              f"{hit[dept][1]/n_dept[dept]:>8.2f}"
              f"{hit[dept][3]/n_dept[dept]:>8.2f}"
              f"{hit[dept][5]/n_dept[dept]:>8.2f}"
              f"{mrr[dept]/n_dept[dept]:>9.2f}")
    print(f"{'整体':<12}"
          f"{overall_hit[1]/n:>8.2f}"
          f"{overall_hit[3]/n:>8.2f}"
          f"{overall_hit[5]/n:>8.2f}"
          f"{overall_mrr/n:>9.2f}")

    if misses:
        print(f"\n=== 错题本（{len(misses)} 题 top-5 未命中 gold）===")
        for dept, q, top in misses:
            print(f"  [{dept}] {q}  -> 误命中: {top}")


if __name__ == "__main__":
    run_eval()

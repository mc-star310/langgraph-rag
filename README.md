# langgraph-rag · 企业知识库问答系统（LangGraph 版）

> 基于 **LangGraph** 编排的企业级 RAG 知识库。把 **HybridRAG 混合检索器**（TF-IDF + bge-m3 语义 + RRF 融合）封装成工具节点，由 LLM 自主决定何时检索，支持可选 **MCP** 工具接入、FastAPI 流式服务、Docker 一键部署与检索评测闭环。

> 本项目聚焦**LangGraph 编排层**：用 StateGraph 显式描述 ReAct 的"思考→调用工具→观察"循环，检索器作为工具节点接入，便于后续加路由、人工介入和多 Agent 扩展。

---

## 一、架构

```
用户问题
   │
   ▼
┌───────────────────────── LangGraph StateGraph ─────────────────────────┐
│  START → [chatbot] →（模型要调工具?）─ 是 → [tools] ─┐                  │
│                 ↑                                   │                  │
│                 └────────────── 否 → END ◄──────────┘                  │
└────────────────────────────────────────────────────────────────────────┘
        [tools] 节点包含：
          • search_knowledge_base  →  HybridRAG 混合检索（TF-IDF + bge-m3 + RRF）
          • （可选）MCP 工具        →  filesystem 等外部能力
                        │
                        ▼
                  HybridRAG
           ┌──────────┴──────────┐
      TF-IDF 关键词路          bge-m3 语义路（Ollama 本地）
           └──────────┬──────────┘
                 RRF 融合 → top-k 片段
```

**技术栈**：LangGraph · LangChain · DeepSeek（生成）· Ollama bge-m3（嵌入）· FastAPI · scikit-learn · Docker

---

## 二、快速开始（本机）

```powershell
# 1. 准备嵌入模型（需先装 Ollama: https://ollama.com）
ollama pull bge-m3

# 2. 建环境
cd langgraph-rag
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# 3. 配置密钥
copy .env.example .env
#   用记事本打开 .env，填入你的 DEEPSEEK_API_KEY

# 4. 放语料：把 .md/.txt/.pdf/.docx 丢进 corpus/ 目录

# 5. 跑起来
python -m graph.agent        # 命令行单轮问答（真 RAG：建索引 + 工具调用，验证用）
python -m eval.t_eval        # 跑检索评测（不花 token）
uvicorn app:app --port 8000  # 启动 Web 服务，浏览器开 http://localhost:8000
```

> 嫌麻烦可直接双击 `start.bat`（Windows），它会拉起 Ollama + 服务并打开浏览器。

---

## 三、检索评测（离线、免费）

```powershell
python -m eval.t_eval
```

输出每个部门与整体的 **R@1 / R@3 / R@5 / MRR@5**，并打印错题本（top-5 没找对的题）。
脚本内置 24 条样例，覆盖 HR / IT / 合规-PIPL / 合规-内部 / 产品-Python 五个部门；接入自有语料后按相同格式扩充题集，`CASES` 里的 `gold` 关键词需改成 `corpus/` 中文件的真实名称。

**关于评测数字**：仓库不含原始语料（涉及企业内部文档），因此 `R@5 = 0.94` 的结果在公开仓库中**无法直接复现**。该数字来自同一套混合检索方法在完整语料（跨 5 部门、约 10 万字、32 题评测集）上的实测：R@5 从纯 TF-IDF 的 0.81 提升至混合检索的 0.94，其中 IT 部门 0.50 → 1.00；叠加上 MQE/HyDE 反而略降至 0.91（强基线下增益递减，已诚实记录）。公开仓库提供完整评测脚本与样例题，便于复现方法而非原始分数。

---

## 四、可选：接入 MCP

编辑 `.env`：

```ini
MCP_ENABLED=true
MCP_FS_ENABLED=true
MCP_FS_PATH=./corpus
```

重启服务后，Agent 会自动加载 filesystem MCP server 的工具，无需改任何业务代码。
要接别的 MCP server，在 `mcp/client.py` 的 `servers` 字典里加一项即可。

---

## 五、Docker 部署

```powershell
docker compose up --build
# 首次需拉取嵌入模型：
docker exec -it langgraph-rag-ollama-1 ollama pull bge-m3
# 打开 http://localhost:8000
```

---

## 六、目录结构

```
langgraph-rag/
├── app.py              # FastAPI 服务（/chat JSON + /chat/stream SSE + 前端）
├── requirements.txt
├── .env.example        # 配置模板（复制为 .env 并填 key）
├── Dockerfile / docker-compose.yml
├── start.bat           # Windows 一键启动
├── graph/
│   └── agent.py        # LangGraph 编排（核心）：StateGraph + 工具节点
├── rag/
│   ├── retriever.py    # HybridRAG 混合检索器（TF-IDF + bge-m3 + RRF）
│   └── ingest.py       # 多格式语料接入（md/txt/pdf/docx）
├── mcp/
│   └── client.py       # 可选 MCP 工具接入
├── eval/
│   └── t_eval.py       # 32 题检索评测
├── web/
│   └── index.html      # 简易聊天前端
└── corpus/             # 放你的文档（.gitkeep 占位）
```

---

## 七、局限与展望（答辩/面试可主动讲）

- 同源多主题文档会"互相抢答"（如内部指引 vs 法律全文），下一步可用**元数据路由**按部门/文档类型预筛。
- 高级检索增强（MQE/HyDE）在强 baseline 上不一定带来提升，以评测数据为准，默认关、可作可选项。
- 当前为单轮检索 + 生成；多轮对话记忆、多智能体协作可后续扩展。

"""
app.py —— FastAPI 服务（Web 演示 + 接口）
=========================================
端点：
  GET  /            -> 简易聊天前端 (web/index.html)
  POST /chat        -> {"message": "..."} 返回 {"answer", "sources"}（JSON，稳定）
  POST /chat/stream -> 同上，但用 SSE 流式返回模型 token（演示用）

注意：首次请求会懒加载 build_agent()（建索引 + 拉起检索），会比较慢，请耐心等几秒。
"""

import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel

from graph.agent import build_agent

load_dotenv()

app = FastAPI(title="LangGraph RAG Knowledge Base")

_AGENT = None


async def get_agent():
    global _AGENT
    if _AGENT is None:
        _AGENT = await build_agent()
    return _AGENT


class ChatReq(BaseModel):
    message: str


@app.post("/chat")
async def chat(req: ChatReq):
    graph, _ = await get_agent()
    result = await graph.ainvoke({"messages": [("user", req.message)]})
    answer = result["messages"][-1].content
    sources = [m.content for m in result["messages"] if getattr(m, "type", "") == "tool"]
    return {"answer": answer, "sources": sources}


@app.post("/chat/stream")
async def chat_stream(req: ChatReq, request: Request):
    graph, _ = await get_agent()

    async def event_gen():
        async for ev in graph.astream_events(
            {"messages": [("user", req.message)]}, version="v2"
        ):
            if ev.get("event") == "on_chat_model_stream":
                chunk = ev["data"]["chunk"]
                if chunk.content:
                    yield f"data: {chunk.content}\n\n"
            if await request.is_disconnected():
                return
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@app.get("/", response_class=HTMLResponse)
async def index():
    p = os.path.join(os.path.dirname(__file__), "web", "index.html")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return f.read()
    return "<h1>LangGraph RAG</h1><p>前端文件 web/index.html 缺失</p>"

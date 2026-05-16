"""
SASRec 番剧推荐系统 - FastAPI Web 后端
模型在启动时加载一次，所有推断通过 RecommenderEngine 完成。
"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from recommender_engine import RecommenderEngine

app = FastAPI(title="SASRec 番剧推荐系统")

# 启动时只加载一次模型和字典
engine = RecommenderEngine()

# 挂载静态文件目录
app.mount("/static", StaticFiles(directory="static"), name="static")


# ========== 请求模型 ==========

class FeedbackRequest(BaseModel):
    session_id: str
    item_id: int
    feedback: str  # "like" 或 "dislike"


class RecommendRequest(BaseModel):
    session_id: str


# ========== API 端点 ==========

@app.get("/")
def serve_index():
    """返回前端首页。"""
    return FileResponse("static/index.html")


@app.post("/api/session/start")
def start_session():
    """启动新推荐 session。"""
    try:
        return engine.start_session()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/session/feedback")
def submit_feedback(req: FeedbackRequest):
    """提交用户反馈（like/dislike），获取下一个候选。"""
    if req.feedback not in ("like", "dislike"):
        raise HTTPException(
            status_code=400,
            detail=f"非法反馈类型: {req.feedback}，仅支持 'like' 或 'dislike'",
        )
    try:
        return engine.submit_feedback(req.session_id, req.item_id, req.feedback)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/session/recommend")
def get_recommendations(req: RecommendRequest):
    """生成最终推荐（Top-10 + I2I 关联）。"""
    try:
        return engine.recommend(req.session_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
def health_check():
    """健康检查接口。"""
    return {
        "status": "ok",
        "device": str(engine.device),
        "item_num": engine.item_num,
    }

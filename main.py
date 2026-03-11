"""
智能学习诊断系统 - FastAPI后端服务
============================
提供学生诊断数据API和AI学习建议服务

接口列表:
- GET /api/students: 获取所有学生列表
- GET /api/diagnose/{student_id}: 获取学生诊断结果
- GET /api/advice/{student_id}: 获取AI学习建议
"""

import os
from typing import Dict, List, Optional
from dataclasses import dataclass

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import httpx

from dina_diagnostic import DataLoader, DINADiagnosticEngine, DiagnosticResult, IRTModel


# ============ 配置常量 ============

DB_PATH = "smart_profile.db"

# LLM服务配置（请填入您的API密钥和Base URL）
LLM_API_KEY = os.getenv("OPENAI_API_KEY", "sk-c5af6455e731484087cc2062d4ffa513")
LLM_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen-turbo")

# 知识点依赖关系（硬编码）
KNOWLEDGE_GRAPH = {
    "K1": {"name": "基础代数", "prerequisite": None},
    "K2": {"name": "一元方程", "prerequisite": "K1"},
    "K3": {"name": "函数概念", "prerequisite": "K2"},
    "K4": {"name": "平面几何", "prerequisite": "K1"},
    "K5": {"name": "立体几何", "prerequisite": "K4"},
}


# ============ 响应模型定义 ============

class StudentInfo(BaseModel):
    """学生信息模型"""
    student_id: str
    student_name: str


class DiagnosisResponse(BaseModel):
    """诊断结果响应模型"""
    student_id: str
    student_name: str
    knowledge_mastery: Dict[str, float]
    most_likely_pattern: List[int]
    pattern_posterior: float
    concept_names: Dict[str, str]
    irt_score: int  # IRT综合能力分（0-100）


class AdviceResponse(BaseModel):
    """学习建议响应模型"""
    student_id: str
    student_name: str
    advice: str
    source: str  # "llm" 或 "mock"


# ============ LLM服务类 ============

class LLMService:
    """
    大语言模型服务类
    支持OpenAI接口标准，提供学习建议生成功能
    """
    
    def __init__(
        self,
        api_key: str = LLM_API_KEY,
        base_url: str = LLM_BASE_URL,
        model: str = LLM_MODEL
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
    
    def build_prompt(
        self,
        student_name: str,
        knowledge_mastery: Dict[str, float]
    ) -> str:
        """
        构建发送给LLM的Prompt
        
        Args:
            student_name: 学生姓名
            knowledge_mastery: 知识点掌握概率字典
            
        Returns:
            str: 完整的Prompt
        """
        # 构建知识点掌握情况描述
        mastery_desc = []
        for kid, prob in knowledge_mastery.items():
            kname = KNOWLEDGE_GRAPH[kid]["name"]
            prereq = KNOWLEDGE_GRAPH[kid]["prerequisite"]
            prereq_desc = f"（前置知识: {KNOWLEDGE_GRAPH[prereq]['name']}" if prereq else "（基础知识点，无前置要求）"
            mastery_desc.append(f"- {kid} {kname}: 掌握概率 {prob*100:.1f}% {prereq_desc}")
        
        mastery_text = "\n".join(mastery_desc)
        
        prompt = f"""你是一位资深的AI教育专家，擅长根据认知诊断结果为学生提供个性化的学习建议。

## 学生信息
学生姓名：{student_name}

## 认知诊断结果
该学生在各知识点上的掌握概率如下：
{mastery_text}

## 知识图谱依赖关系
- K1（基础代数）是基础知识点，无前置要求
- K2（一元方程）需要先掌握K1
- K3（函数概念）需要先掌握K2
- K4（平面几何）需要先掌握K1
- K5（立体几何）需要先掌握K4

## 任务要求
请根据以上诊断结果和知识图谱依赖关系，为该学生生成一份个性化的学习建议，要求：
1. 分析学生当前的知识掌握状况，指出优势和薄弱环节
2. 根据知识图谱的前后置关系，给出合理的学习路径建议
3. 针对薄弱知识点，提供具体的学习策略建议
4. 语言要亲切、鼓励性强，适合学生阅读

请直接输出学习建议，不要有多余的开场白。"""
        
        return prompt
    
    async def generate_advice(
        self,
        student_name: str,
        knowledge_mastery: Dict[str, float]
    ) -> tuple[str, str]:
        """
        调用LLM生成学习建议
        
        Args:
            student_name: 学生姓名
            knowledge_mastery: 知识点掌握概率字典
            
        Returns:
            tuple[str, str]: (建议内容, 来源标识 "llm"或"mock")
        """
        prompt = self.build_prompt(student_name, knowledge_mastery)
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": "你是一位专业的AI教育顾问。"},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.7,
                        "max_tokens": 1000
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    advice = data["choices"][0]["message"]["content"]
                    return advice, "llm"
                else:
                    return self._generate_mock_advice(student_name, knowledge_mastery), "mock"
                    
        except Exception as e:
            print(f"LLM调用失败: {e}")
            return self._generate_mock_advice(student_name, knowledge_mastery), "mock"
    
    def _generate_mock_advice(
        self,
        student_name: str,
        knowledge_mastery: Dict[str, float]
    ) -> str:
        """
        基于规则的Mock建议生成（兜底方案）
        
        Args:
            student_name: 学生姓名
            knowledge_mastery: 知识点掌握概率字典
            
        Returns:
            str: 模拟的学习建议
        """
        # 分析掌握情况
        strong_points = []
        weak_points = []
        
        for kid, prob in knowledge_mastery.items():
            kname = KNOWLEDGE_GRAPH[kid]["name"]
            if prob >= 0.6:
                strong_points.append(f"{kname}({prob*100:.0f}%)")
            elif prob < 0.4:
                weak_points.append((kid, kname, prob))
        
        # 构建建议
        advice_parts = [f"亲爱的{student_name}同学，以下是为你定制的个性化学习建议：\n"]
        
        # 优势分析
        if strong_points:
            advice_parts.append(f"📊 **优势分析**\n你在以下知识点表现出色：{', '.join(strong_points)}。继续保持，这些是你后续学习的坚实基础！\n")
        else:
            advice_parts.append("📊 **现状分析**\n目前各知识点掌握程度有待提升，但别担心，我们有明确的学习路径！\n")
        
        # 薄弱环节分析
        if weak_points:
            weak_points.sort(key=lambda x: x[2])
            advice_parts.append("🎯 **重点突破建议**\n")
            for kid, kname, prob in weak_points:
                prereq = KNOWLEDGE_GRAPH[kid]["prerequisite"]
                if prereq:
                    prereq_prob = knowledge_mastery.get(prereq, 0)
                    if prereq_prob < 0.6:
                        prereq_name = KNOWLEDGE_GRAPH[prereq]["name"]
                        advice_parts.append(
                            f"- **{kname}**（掌握率{prob*100:.0f}%）：建议先巩固前置知识「{prereq_name}」，"
                            f"因为掌握{prereq_name}是学好{kname}的关键基础。\n"
                        )
                    else:
                        advice_parts.append(
                            f"- **{kname}**（掌握率{prob*100:.0f}%）：前置知识已掌握，建议多做相关练习题，"
                            f"重点理解核心概念和解题方法。\n"
                        )
                else:
                    advice_parts.append(
                        f"- **{kname}**（掌握率{prob*100:.0f}%）：这是基础知识点，建议回归教材，"
                        f"夯实基础概念和基本运算能力。\n"
                    )
        
        # 学习路径建议
        advice_parts.append("\n🚀 **推荐学习路径**\n")
        advice_parts.append("根据知识图谱依赖关系，建议按以下顺序学习：\n")
        advice_parts.append("1. 先确保K1（基础代数）掌握牢固\n")
        advice_parts.append("2. 在K1基础上学习K2（一元方程）和K4（平面几何）\n")
        advice_parts.append("3. 掌握K2后进阶学习K3（函数概念）\n")
        advice_parts.append("4. 掌握K4后进阶学习K5（立体几何）\n")
        
        advice_parts.append("\n💪 **加油！** 学习是一个循序渐进的过程，相信通过有针对性的练习，你一定能取得进步！")
        
        return "".join(advice_parts)


# ============ 全局变量（单例模式） ============

_data_loader: Optional[DataLoader] = None
_diagnostic_engine: Optional[DINADiagnosticEngine] = None
_diagnostic_results: Optional[List[DiagnosticResult]] = None
_irt_model: Optional[IRTModel] = None
_llm_service: Optional[LLMService] = None


def get_data_loader() -> DataLoader:
    """获取数据加载器单例"""
    global _data_loader
    if _data_loader is None:
        _data_loader = DataLoader(DB_PATH)
        _data_loader.connect()
    return _data_loader


def get_diagnostic_engine() -> DINADiagnosticEngine:
    """获取诊断引擎单例"""
    global _diagnostic_engine, _data_loader
    if _diagnostic_engine is None:
        loader = get_data_loader()
        Q = loader.build_q_matrix()
        _diagnostic_engine = DINADiagnosticEngine(q_matrix=Q)
    return _diagnostic_engine


def get_diagnostic_results() -> List[DiagnosticResult]:
    """获取所有学生的诊断结果"""
    global _diagnostic_results, _data_loader, _diagnostic_engine
    if _diagnostic_results is None:
        loader = get_data_loader()
        engine = get_diagnostic_engine()
        X = loader.build_x_matrix()
        _diagnostic_results = engine.diagnose_all_students(X, loader)
    return _diagnostic_results


def get_irt_model() -> IRTModel:
    """获取IRT模型单例"""
    global _irt_model, _data_loader
    if _irt_model is None:
        loader = get_data_loader()
        X = loader.build_x_matrix()
        _irt_model = IRTModel(x_matrix=X)
    return _irt_model


def get_llm_service() -> LLMService:
    """获取LLM服务单例"""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service


# ============ FastAPI应用 ============

app = FastAPI(
    title="智能学习诊断系统",
    description="基于DINA模型的认知诊断与AI学习建议服务",
    version="1.0.0"
)


@app.get("/api/students", response_model=List[StudentInfo])
async def get_students():
    """
    获取所有学生列表
    
    Returns:
        List[StudentInfo]: 学生信息列表
    """
    loader = get_data_loader()
    students = []
    for i, student_id in enumerate(loader.student_ids):
        students.append(StudentInfo(
            student_id=student_id,
            student_name=loader.student_names[i]
        ))
    return students


@app.get("/api/diagnose/{student_id}", response_model=DiagnosisResponse)
async def diagnose_student(student_id: str):
    """
    获取学生诊断结果（双引擎：DINA + IRT）
    
    Args:
        student_id: 学生ID
        
    Returns:
        DiagnosisResponse: 诊断结果（包含DINA知识点掌握和IRT综合能力分）
    """
    results = get_diagnostic_results()
    loader = get_data_loader()
    irt_model = get_irt_model()
    
    for i, result in enumerate(results):
        if result.student_id == student_id:
            # 获取IRT综合能力分
            irt_score = irt_model.get_student_irt_score(i)
            
            return DiagnosisResponse(
                student_id=result.student_id,
                student_name=result.student_name,
                knowledge_mastery=result.knowledge_mastery,
                most_likely_pattern=result.most_likely_pattern.tolist(),
                pattern_posterior=result.pattern_posterior,
                concept_names={kid: name for kid, name in zip(loader.concept_ids, loader.concept_names)},
                irt_score=irt_score
            )
    
    raise HTTPException(status_code=404, detail=f"学生 {student_id} 不存在")


@app.get("/api/advice/{student_id}", response_model=AdviceResponse)
async def get_advice(student_id: str):
    """
    获取AI学习建议
    
    Args:
        student_id: 学生ID
        
    Returns:
        AdviceResponse: 学习建议
    """
    results = get_diagnostic_results()
    
    for result in results:
        if result.student_id == student_id:
            llm_service = get_llm_service()
            advice, source = await llm_service.generate_advice(
                result.student_name,
                result.knowledge_mastery
            )
            return AdviceResponse(
                student_id=result.student_id,
                student_name=result.student_name,
                advice=advice,
                source=source
            )
    
    raise HTTPException(status_code=404, detail=f"学生 {student_id} 不存在")


@app.get("/", response_class=HTMLResponse)
async def index():
    """返回前端页面"""
    html_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return HTMLResponse(content="<h1>请创建 templates/index.html 文件</h1>", status_code=404)


# 挂载静态文件目录
templates_dir = os.path.join(os.path.dirname(__file__), "templates")
if os.path.exists(templates_dir):
    app.mount("/static", StaticFiles(directory=templates_dir), name="static")


# ============ 应用生命周期事件 ============

@app.on_event("startup")
async def startup_event():
    """应用启动时初始化"""
    print("=" * 50)
    print("智能学习诊断系统启动中...")
    print("=" * 50)
    
    # 预加载数据
    get_data_loader()
    get_diagnostic_engine()
    get_diagnostic_results()
    get_irt_model()
    
    print("✓ 数据加载完成")
    print(f"✓ 学生数: {len(get_diagnostic_results())}")
    print("✓ 双引擎诊断系统就绪（DINA + IRT）")
    print("=" * 50)


@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时清理资源"""
    global _data_loader
    if _data_loader:
        _data_loader.disconnect()
    print("数据库连接已关闭")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8051)

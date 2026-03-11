smart_profile_mvp/

├── data_visual/

│   └── data_visualization.html   # 初始化数据可视化

│   └── data_server.py          # 数据后端

├── templates/

│   └── index.html          # 前端可视化大屏

├── main.py                 # FastAPI 后端服务与核心 API

├── dina_model.py           # 核心算法类 (DINA + IRT)

├── init_db.py              # 数据库初始化与数据生成脚本

├── smart_profile.db        # SQLite 数据库文件 


├── requirements.txt        # 核心依赖包声明

├── .gitignore              # Git 忽略文件配置

├── README.md               # 核心说明与运行文档

# SmartProfile - 智能化认知诊断与知识追踪引擎 MVP

这是一个基于 DINA 认知诊断模型与大语言模型 (LLM) 的轻量级教育算法服务 Demo。

## 1. 核心算法逻辑
* **微观诊断 (DINA 模型)**：利用贝叶斯后验概率，结合 Q 矩阵与作答记录，反推学生在 5 个底层知识点上的掌握概率。对不符合知识图谱前置依赖的隐状态进行了先验概率的硬约束惩罚。
* **宏观评估 (IRT 辅助)**：计算综合能力得分，实现学霸与学渣的快速分层。

## 2. 环境依赖
* Python 3.8+
* 推荐使用虚拟环境运行

## 3. 极速运行指南 (3步启动)

**Step 1: 安装依赖**
```bash
pip install -r requirements.txt
```
 --- 
**Step 2: 启动后端服务**
```bash
uvicorn main:app --reload
```
 --- 
**Step 3: 访问前端页面**
打开浏览器，访问：http://127.0.0.1:8000 即可查看可视化认知雷达图与知识图谱。

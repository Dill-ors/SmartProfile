smart_profile_mvp/

├── data_visualization.html   # 初始化数据可视化

├── data_server.py          # 数据后端

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
* **宏观评估 (IRT 辅助)**：在已知学生真实作答矩阵（观测值）的情况下，系统通过极大似然估计近似推导出连续的能力参数 $\theta_i$，并将其线性映射为 $0 \sim 100$ 的综合百分制得分，作为前端界面的“综合能力评分”。

## 2. 环境依赖
* Python 3.9+
* 推荐使用虚拟环境运行
* 需要的包在requirements.txt

## 3. 极速运行指南

**Step 1: 创建环境**
```bash
#利用anaconda创建虚拟环境
#打开Anaconda Prompt
conda create -n myenv python=3.9

# Windows 激活命令
conda activate smartprofile_env

#进入到项目文件夹
cd /d 文件目录

```

**Step 2: 安装依赖**
```bash：
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
```
 --- 
**Step 3: 启动后端服务**
```bash
uvicorn main:app --reload
```
 --- 
**Step 4: 访问前端页面**
打开浏览器，访问：http://127.0.0.1:8000 即可查看可视化认知雷达图与知识图谱。
或者按住ctrl，点击出现的浏览器链接，会自动跳转
```bash
# 如果端口占用，指定端口（比如用 8080 端口）
uvicorn main:app --reload --port 8080
```
 --- 
**Step 5: 核验数据**
```bash
python data_server.py
```
然后打开文件夹中的data_visualization.html文件，就可以看到初始化的数据，可以与算法的生成结果进行对比


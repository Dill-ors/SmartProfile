smart_profile_mvp/
├── main.py                 # FastAPI 后端服务与核心 API
├── dina_model.py           # 核心算法类 (DINA + IRT)
├── init_db.py              # (可选) 数据库初始化与数据生成脚本
├── smart_profile.db        # SQLite 数据库文件 (带上它，评委免去初始化步骤)
├── templates/
│   └── index.html          # 前端可视化大屏
├── requirements.txt        # 核心依赖包声明
├── .gitignore              # Git 忽略文件配置
├── README.md               # 核心说明与运行文档

"""
智能学习档案数据库系统
============================
基于 Python sqlite3 的本地数据库实现
用于模拟学生知识点掌握状态与答题表现的关系

核心概念:
- Knowledge Concepts (知识点): 构成学习路径的基础单元
- Q-Matrix (Q矩阵): 题目与知识点的关联矩阵
- Students (学生): 学习者信息
- X-Matrix (作答矩阵): 学生的答题记录
"""

import sqlite3
import random
from typing import List, Tuple, Dict
from dataclasses import dataclass


# ============ 配置常量 ============

DB_NAME = "smart_profile.db"

# 知识点配置 - 构建一个树状依赖结构
# K1 (基础代数) -> K2 (方程) -> K3 (函数)
# K1 (基础代数) -> K4 (几何基础) -> K5 (立体几何)
CONCEPTS_DATA = [
    ("K1", "基础代数", None),        # 根节点，无前置依赖
    ("K2", "一元方程", "K1"),        # 依赖K1
    ("K3", "函数概念", "K2"),        # 依赖K2
    ("K4", "平面几何", "K1"),        # 依赖K1
    ("K5", "立体几何", "K4"),        # 依赖K4
]

# 学生配置
STUDENT_NAMES = [
    "张伟", "李娜", "王强", "刘洋", "陈静",
    "杨帆", "赵敏", "黄磊", "周杰", "吴倩",
    "徐鹏", "孙丽"
]

# 概率配置
PROB_MASTERY_CORRECT = 0.85    # 掌握知识点后的正确率（Slip = 0.15）
PROB_GUESS_CORRECT = 0.20      # 未掌握知识点的猜测正确率（Guess = 0.2）


@dataclass
class StudentKnowledge:
    """学生知识点掌握状态的数据类"""
    student_id: str
    knowledge_state: Dict[str, int]  # 知识点ID -> 是否掌握(0或1)


# ============ 数据库连接管理 ============

def get_connection() -> sqlite3.Connection:
    """
    获取数据库连接
    返回配置好的sqlite3连接对象
    """
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row  # 使查询结果可以通过列名访问
    return conn


# ============ 表结构创建 ============

def create_tables(conn: sqlite3.Connection) -> None:
    """
    创建所有核心数据表
    
    表结构说明:
    1. knowledge_concepts: 存储知识点及其前置依赖关系
    2. q_matrix: 存储题目与知识点的关联（一道题可关联多个知识点）
    3. students: 存储学生基本信息
    4. x_matrix_responses: 存储学生的答题记录
    """
    cursor = conn.cursor()
    
    # 知识点表 - 包含概念ID、名称和前置知识点ID
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS knowledge_concepts (
            concept_id TEXT PRIMARY KEY,
            concept_name TEXT NOT NULL,
            prerequisite_id TEXT,
            FOREIGN KEY (prerequisite_id) REFERENCES knowledge_concepts(concept_id)
        )
    """)
    
    # Q矩阵表 - 题目与知识点的关联
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS q_matrix (
            question_id TEXT PRIMARY KEY,
            concept_id TEXT NOT NULL,
            FOREIGN KEY (concept_id) REFERENCES knowledge_concepts(concept_id)
        )
    """)
    
    # 学生表 - 存储学习者信息
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS students (
            student_id TEXT PRIMARY KEY,
            student_name TEXT NOT NULL
        )
    """)
    
    # 作答矩阵表 - 存储学生的答题记录
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS x_matrix_responses (
            response_id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            question_id TEXT NOT NULL,
            is_correct INTEGER NOT NULL CHECK (is_correct IN (0, 1)),
            FOREIGN KEY (student_id) REFERENCES students(student_id),
            FOREIGN KEY (question_id) REFERENCES q_matrix(question_id),
            UNIQUE(student_id, question_id)
        )
    """)
    
    conn.commit()
    print("✓ 所有数据表创建成功")


# ============ 知识点数据初始化 ============

def init_knowledge_concepts(conn: sqlite3.Connection) -> None:
    """
    初始化知识点数据
    
    知识点依赖结构:
    K1(基础代数)
    ├── K2(一元方程) → K3(函数概念)
    └── K4(平面几何) → K5(立体几何)
    
    这是一个典型的学习路径分叉结构，体现不同学习分支
    """
    cursor = conn.cursor()
    
    cursor.executemany(
        "INSERT OR REPLACE INTO knowledge_concepts (concept_id, concept_name, prerequisite_id) VALUES (?, ?, ?)",
        CONCEPTS_DATA
    )
    
    conn.commit()
    print(f"✓ 知识点数据初始化完成，共 {len(CONCEPTS_DATA)} 个知识点")
    
    # 打印知识点结构
    print("\n知识点依赖结构:")
    for concept_id, name, prereq in CONCEPTS_DATA:
        prereq_str = f" (前置: {prereq})" if prereq else " (根节点)"
        print(f"  {concept_id}: {name}{prereq_str}")


# ============ Q矩阵数据初始化 ============

def init_q_matrix(conn: sqlite3.Connection) -> None:
    """
    初始化Q矩阵数据 - 20道题目，每道关联1-2个知识点
    
    生成策略:
    - 题目Q1-Q4: 只考K1 (基础代数)
    - 题目Q5-Q8: 只考K2 (一元方程)
    - 题目Q9-Q12: 只考K3 (函数概念)
    - 题目Q13-Q16: 只考K4 (平面几何)
    - 题目Q17-Q20: 只考K5 (立体几何)
    
    每道题只关联一个知识点，简化模型但保留扩展性
    """
    cursor = conn.cursor()
    
    concept_ids = [c[0] for c in CONCEPTS_DATA]
    q_matrix_data = []
    
    # 生成20道题目，每道题关联1-2个知识点
    for i in range(1, 21):
        question_id = f"Q{i:02d}"
        
        # 随机决定关联1个还是2个知识点
        num_concepts = random.randint(1, 2)
        
        if num_concepts == 1:
            # 关联1个知识点
            concept_id = random.choice(concept_ids)
            q_matrix_data.append((question_id, concept_id))
        else:
            # 关联2个有依赖关系的知识点
            # 选择一条路径上的两个知识点
            paths = [
                ["K1", "K2"], ["K2", "K3"],
                ["K1", "K4"], ["K4", "K5"]
            ]
            path = random.choice(paths)
            # 为主知识点创建题目记录
            # 这里简化处理，只记录主要考察的知识点
            q_matrix_data.append((question_id, path[1]))
    
    cursor.executemany(
        "INSERT OR REPLACE INTO q_matrix (question_id, concept_id) VALUES (?, ?)",
        q_matrix_data
    )
    
    conn.commit()
    print(f"✓ Q矩阵初始化完成，共 {len(q_matrix_data)} 道题目")
    
    # 打印Q矩阵分布
    print("\n题目-知识点分布:")
    cursor.execute("""
        SELECT concept_id, COUNT(*) as count 
        FROM q_matrix 
        GROUP BY concept_id
    """)
    for row in cursor.fetchall():
        print(f"  {row['concept_id']}: {row['count']} 道题目")


# ============ 学生数据初始化 ============

def init_students(conn: sqlite3.Connection) -> None:
    """
    初始化学生数据 - 12名学生
    
    学生ID格式: S01, S02, ..., S12
    """
    cursor = conn.cursor()
    
    students_data = [
        (f"S{i+1:02d}", name)
        for i, name in enumerate(STUDENT_NAMES)
    ]
    
    cursor.executemany(
        "INSERT OR REPLACE INTO students (student_id, student_name) VALUES (?, ?)",
        students_data
    )
    
    conn.commit()
    print(f"✓ 学生数据初始化完成，共 {len(students_data)} 名学生")


# ============ 学生知识点掌握状态生成 ============

def generate_student_knowledge_states(conn: sqlite3.Connection) -> List[StudentKnowledge]:
    """
    为每个学生生成隐式的知识点掌握状态
    
    生成逻辑:
    1. 定义合法的知识掌握模式池
    2. 为每个学生随机分配一个固定的真实隐状态
    
    合法模式定义:
    - 学渣型: [0,0,0,0,0]
    - 基础型: [1,0,0,0,0]
    - 偏科型1: [1,1,1,0,0] (代数分支)
    - 偏科型2: [1,0,0,1,1] (几何分支)
    - 学霸型: [1,1,1,1,1]
    
    Returns:
        List[StudentKnowledge]: 每个学生的知识点掌握状态
    """
    cursor = conn.cursor()
    
    # 获取所有学生
    cursor.execute("SELECT student_id FROM students")
    student_ids = [row['student_id'] for row in cursor.fetchall()]
    
    # 定义合法的知识掌握模式池
    # 每个模式是一个字典，键为知识点ID，值为掌握状态(0或1)
    legal_patterns = [
        # 学渣型: 未掌握任何知识点
        {"K1": 0, "K2": 0, "K3": 0, "K4": 0, "K5": 0},
        # 基础型: 只掌握K1
        {"K1": 1, "K2": 0, "K3": 0, "K4": 0, "K5": 0},
        # 偏科型1: 掌握代数分支 (K1->K2->K3)
        {"K1": 1, "K2": 1, "K3": 1, "K4": 0, "K5": 0},
        # 偏科型2: 掌握几何分支 (K1->K4->K5)
        {"K1": 1, "K2": 0, "K3": 0, "K4": 1, "K5": 1},
        # 学霸型: 掌握所有知识点
        {"K1": 1, "K2": 1, "K3": 1, "K4": 1, "K5": 1}
    ]
    
    student_knowledge_states = []
    
    for student_id in student_ids:
        # 随机选择一个合法模式作为学生的真实隐状态
        knowledge_state = random.choice(legal_patterns)
        
        student_knowledge_states.append(
            StudentKnowledge(student_id=student_id, knowledge_state=knowledge_state)
        )
    
    return student_knowledge_states


# ============ 作答数据生成 ============

def init_x_matrix_responses(conn: sqlite3.Connection) -> None:
    """
    初始化学生作答数据
    
    核心算法:
    对于每道题目，检查学生是否掌握了该题目考察的所有知识点:
    - 如果全部掌握: 正确概率 = 85% (PROB_MASTERY_CORRECT, Slip = 0.15)
    - 如果有任何知识点未掌握: 正确概率 = 20% (PROB_GUESS_CORRECT, Guess = 0.2)
    
    这种设计体现了DINA模型的基本思想:
    学生的答题表现取决于其潜在特质(知识点掌握状态)
    """
    cursor = conn.cursor()
    
    # 获取所有题目及其关联的知识点
    # 注意：这里需要按题目分组，因为一道题可能关联多个知识点
    cursor.execute("SELECT question_id, concept_id FROM q_matrix")
    question_concepts = cursor.fetchall()
    
    # 按题目分组知识点
    questions_dict = {}
    for qid, cid in question_concepts:
        if qid not in questions_dict:
            questions_dict[qid] = []
        questions_dict[qid].append(cid)
    
    # 获取所有学生
    cursor.execute("SELECT student_id, student_name FROM students")
    students = cursor.fetchall()
    
    # 生成学生的知识点掌握状态（隐式）
    student_knowledge_states = generate_student_knowledge_states(conn)
    
    # 将掌握状态转换为字典便于查询
    knowledge_dict = {
        sk.student_id: sk.knowledge_state
        for sk in student_knowledge_states
    }
    
    responses_data = []
    
    print("\n生成作答数据...")
    print("-" * 60)
    
    for student_row in students:
        student_id = student_row['student_id']
        student_name = student_row['student_name']
        student_knowledge = knowledge_dict[student_id]
        
        correct_count = 0
        
        # 构建学生的隐状态向量用于打印
        state_vector = [student_knowledge.get(k, 0) for k in ['K1', 'K2', 'K3', 'K4', 'K5']]
        state_str = '[' + ','.join(map(str, state_vector)) + ']'
        
        for question_id, required_concepts in questions_dict.items():
            # 判断学生是否掌握了该题目要求的所有知识点（AND逻辑）
            has_mastery = all(student_knowledge.get(cid, 0) == 1 for cid in required_concepts)
            
            # 根据掌握状态决定正确概率
            if has_mastery:
                prob_correct = PROB_MASTERY_CORRECT
            else:
                prob_correct = PROB_GUESS_CORRECT
            
            # 根据概率生成作答结果
            is_correct = 1 if random.random() < prob_correct else 0
            
            if is_correct:
                correct_count += 1
            
            responses_data.append((student_id, question_id, is_correct))
        
        # 打印该学生的答题统计
        total_questions = len(questions_dict)
        accuracy = correct_count / total_questions * 100
        mastery_str = ", ".join([
            k for k, v in student_knowledge.items() if v == 1
        ])
        print(f"{student_id} {student_name}: 真实隐状态 {state_str}, 正确率 {accuracy:.1f}%, 掌握: {mastery_str}")
    
    # 清空作答表并批量插入新数据
    cursor.execute("DELETE FROM x_matrix_responses")
    cursor.executemany(
        "INSERT INTO x_matrix_responses (student_id, question_id, is_correct) VALUES (?, ?, ?)",
        responses_data
    )
    
    conn.commit()
    print("-" * 60)
    print(f"✓ 作答数据初始化完成，共 {len(responses_data)} 条记录")


# ============ 数据验证与统计 ============

def verify_database(conn: sqlite3.Connection) -> None:
    """
    验证数据库完整性并输出统计信息
    """
    cursor = conn.cursor()
    
    print("\n" + "=" * 60)
    print("数据库验证与统计")
    print("=" * 60)
    
    # 统计各表记录数
    tables = [
        ("knowledge_concepts", "知识点"),
        ("q_matrix", "题目"),
        ("students", "学生"),
        ("x_matrix_responses", "作答记录")
    ]
    
    print("\n数据表统计:")
    for table_name, desc in tables:
        cursor.execute(f"SELECT COUNT(*) as count FROM {table_name}")
        count = cursor.fetchone()['count']
        print(f"  {desc}: {count}")
    
    # 统计各知识点相关的答题情况
    print("\n各知识点答题统计:")
    cursor.execute("""
        SELECT 
            kc.concept_id,
            kc.concept_name,
            COUNT(*) as total_responses,
            SUM(xr.is_correct) as correct_count,
            ROUND(AVG(xr.is_correct) * 100, 2) as accuracy_rate
        FROM knowledge_concepts kc
        JOIN q_matrix qm ON kc.concept_id = qm.concept_id
        JOIN x_matrix_responses xr ON qm.question_id = xr.question_id
        GROUP BY kc.concept_id
    """)
    
    for row in cursor.fetchall():
        print(f"  {row['concept_id']} ({row['concept_name']}):")
        print(f"    答题次数: {row['total_responses']}, 正确: {row['correct_count']}, 正确率: {row['accuracy_rate']}%")
    
    # 统计每个学生的答题正确率
    print("\n学生答题正确率排名:")
    cursor.execute("""
        SELECT 
            s.student_id,
            s.student_name,
            COUNT(*) as total,
            SUM(xr.is_correct) as correct,
            ROUND(AVG(xr.is_correct) * 100, 2) as accuracy
        FROM students s
        JOIN x_matrix_responses xr ON s.student_id = xr.student_id
        GROUP BY s.student_id
        ORDER BY accuracy DESC
    """)
    
    for row in cursor.fetchall():
        print(f"  {row['student_id']} {row['student_name']}: {row['correct']}/{row['total']} = {row['accuracy']}%")
    
    print("=" * 60)


# ============ 主程序入口 ============

def main():
    """
    主程序入口
    按顺序执行：建表 -> 初始化知识点 -> 初始化Q矩阵 -> 初始化学生 -> 生成作答数据 -> 验证
    """
    print("=" * 60)
    print("智能学习档案数据库系统 - 初始化")
    print("=" * 60)
    
    # 设置随机种子以保证可重复性（可选）
    random.seed(42)
    
    # 连接数据库
    conn = get_connection()
    
    try:
        # 创建表结构
        create_tables(conn)
        
        # 初始化知识点
        init_knowledge_concepts(conn)
        
        # 初始化Q矩阵
        init_q_matrix(conn)
        
        # 初始化学生
        init_students(conn)
        
        # 生成作答数据
        init_x_matrix_responses(conn)
        
        # 验证数据库
        verify_database(conn)
        
        print("\n✓ 数据库初始化全部完成！")
        print(f"数据库文件: {DB_NAME}")
        
    except Exception as e:
        print(f"\n✗ 发生错误: {e}")
        raise
    
    finally:
        conn.close()
        print("\n数据库连接已关闭")


if __name__ == "__main__":
    main()

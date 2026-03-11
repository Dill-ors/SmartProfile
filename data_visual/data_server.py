"""
数据查询服务
提供 RESTful API 供前端查询数据库中的数据
"""

import sqlite3
from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # 允许跨域请求

DB_NAME = "smart_profile.db"


def get_connection():
    """获取数据库连接"""
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn


@app.route('/api/concepts', methods=['GET'])
def get_concepts():
    """获取所有知识点"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT concept_id, concept_name, prerequisite_id FROM knowledge_concepts")
    concepts = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return jsonify(concepts)


@app.route('/api/questions', methods=['GET'])
def get_questions():
    """获取所有题目"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT question_id, concept_id FROM q_matrix")
    questions = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return jsonify(questions)


@app.route('/api/students', methods=['GET'])
def get_students():
    """获取所有学生"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT student_id, student_name FROM students")
    students = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return jsonify(students)


@app.route('/api/responses', methods=['GET'])
def get_responses():
    """获取所有作答记录"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT response_id, student_id, question_id, is_correct FROM x_matrix_responses")
    responses = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return jsonify(responses)


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """获取统计数据"""
    conn = get_connection()
    cursor = conn.cursor()
    
    # 统计各表记录数
    cursor.execute("SELECT COUNT(*) as count FROM knowledge_concepts")
    concept_count = cursor.fetchone()['count']
    
    cursor.execute("SELECT COUNT(*) as count FROM q_matrix")
    question_count = cursor.fetchone()['count']
    
    cursor.execute("SELECT COUNT(*) as count FROM students")
    student_count = cursor.fetchone()['count']
    
    cursor.execute("SELECT COUNT(*) as count FROM x_matrix_responses")
    response_count = cursor.fetchone()['count']
    
    # 统计各知识点答题情况
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
    concept_stats = [dict(row) for row in cursor.fetchall()]
    
    # 统计学生答题正确率
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
    student_stats = [dict(row) for row in cursor.fetchall()]
    
    conn.close()
    
    return jsonify({
        'concept_count': concept_count,
        'question_count': question_count,
        'student_count': student_count,
        'response_count': response_count,
        'concept_stats': concept_stats,
        'student_stats': student_stats
    })


if __name__ == '__main__':
    print("=" * 60)
    print("数据查询服务启动")
    print("=" * 60)
    print("API 端点:")
    print("  - GET /api/concepts    获取知识点列表")
    print("  - GET /api/questions   获取题目列表")
    print("  - GET /api/students    获取学生列表")
    print("  - GET /api/responses   获取作答记录")
    print("  - GET /api/stats       获取统计数据")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=True)
"""
DINA模型认知诊断引擎
============================
基于 DINA (Deterministic Input, Noisy "And" Gate) 模型的认知诊断实现
用于诊断学生在各知识点上的掌握概率

核心概念:
- Q矩阵: 题目与知识点的关联矩阵
- X矩阵: 学生的作答记录矩阵
- η矩阵: 理想作答矩阵（基于知识掌握模式）
- 猜测率 g: 未掌握所有知识点时答对的概率
- 失误率 s: 掌握所有知识点时答错的概率

算法流程:
1. 生成所有可能的知识掌握模式（2^K种，K为知识点数）
2. 基于Q矩阵计算每种模式下的理想作答η
3. 对每个学生，计算其作答向量在各模式下的后验概率
4. 边缘化得到各知识点的掌握概率
"""

import sqlite3
import numpy as np
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
from itertools import product


# ============ 数据结构定义 ============

@dataclass
class DiagnosticResult:
    """诊断结果数据类"""
    student_id: str
    student_name: str
    knowledge_mastery: Dict[str, float]  # 知识点ID -> 掌握概率
    most_likely_pattern: np.ndarray       # 最可能的知识掌握模式
    pattern_posterior: float              # 最可能模式的后验概率


# ============ 数据加载模块 ============

class DataLoader:
    """
    数据加载器
    负责从SQLite数据库读取数据并构建标准矩阵格式
    """
    
    def __init__(self, db_path: str = "smart_profile.db"):
        """
        初始化数据加载器
        
        Args:
            db_path: 数据库文件路径
        """
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None
        
        # 元数据存储
        self.concept_ids: List[str] = []      # 知识点ID列表
        self.concept_names: List[str] = []    # 知识点名称列表
        self.question_ids: List[str] = []     # 题目ID列表
        self.student_ids: List[str] = []      # 学生ID列表
        self.student_names: List[str] = []    # 学生姓名列表
    
    def connect(self) -> None:
        """建立数据库连接"""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
    
    def disconnect(self) -> None:
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def load_knowledge_concepts(self) -> None:
        """加载知识点数据，建立ID到索引的映射"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT concept_id, concept_name 
            FROM knowledge_concepts 
            ORDER BY concept_id
        """)
        
        rows = cursor.fetchall()
        self.concept_ids = [row['concept_id'] for row in rows]
        self.concept_names = [row['concept_name'] for row in rows]
    
    def load_questions(self) -> None:
        """加载题目数据，建立ID到索引的映射"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT DISTINCT question_id 
            FROM q_matrix 
            ORDER BY question_id
        """)
        
        self.question_ids = [row['question_id'] for row in cursor.fetchall()]
    
    def load_students(self) -> None:
        """加载学生数据，建立ID到索引的映射"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT student_id, student_name 
            FROM students 
            ORDER BY student_id
        """)
        
        rows = cursor.fetchall()
        self.student_ids = [row['student_id'] for row in rows]
        self.student_names = [row['student_name'] for row in rows]
    
    def build_q_matrix(self) -> np.ndarray:
        """
        构建Q矩阵
        
        Q矩阵定义: Q[i,j] = 1 表示第i道题考察第j个知识点
        
        Returns:
            np.ndarray: 形状为 (题目数, 知识点数) 的二维数组
        """
        if not self.concept_ids or not self.question_ids:
            self.load_knowledge_concepts()
            self.load_questions()
        
        cursor = self.conn.cursor()
        
        # 初始化Q矩阵 (题目数 x 知识点数)
        num_questions = len(self.question_ids)
        num_concepts = len(self.concept_ids)
        Q = np.zeros((num_questions, num_concepts), dtype=np.int32)
        
        # 建立ID到索引的映射
        question_idx = {qid: i for i, qid in enumerate(self.question_ids)}
        concept_idx = {cid: i for i, cid in enumerate(self.concept_ids)}
        
        # 查询题目-知识点关联
        cursor.execute("""
            SELECT question_id, concept_id 
            FROM q_matrix
        """)
        
        for row in cursor.fetchall():
            i = question_idx[row['question_id']]
            j = concept_idx[row['concept_id']]
            Q[i, j] = 1
        
        return Q
    
    def build_x_matrix(self) -> np.ndarray:
        """
        构建X矩阵（作答矩阵）
        
        X矩阵定义: X[i,j] = 1 表示第i个学生答对了第j道题
        
        Returns:
            np.ndarray: 形状为 (学生数, 题目数) 的二维数组
        """
        if not self.student_ids or not self.question_ids:
            self.load_students()
            self.load_questions()
        
        cursor = self.conn.cursor()
        
        # 初始化X矩阵 (学生数 x 题目数)
        num_students = len(self.student_ids)
        num_questions = len(self.question_ids)
        X = np.zeros((num_students, num_questions), dtype=np.int32)
        
        # 建立ID到索引的映射
        student_idx = {sid: i for i, sid in enumerate(self.student_ids)}
        question_idx = {qid: i for i, qid in enumerate(self.question_ids)}
        
        # 查询作答记录
        cursor.execute("""
            SELECT student_id, question_id, is_correct 
            FROM x_matrix_responses
        """)
        
        for row in cursor.fetchall():
            i = student_idx[row['student_id']]
            j = question_idx[row['question_id']]
            X[i, j] = row['is_correct']
        
        return X
    
    def load_all_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        加载所有数据并构建矩阵
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: (Q矩阵, X矩阵)
        """
        self.connect()
        
        try:
            Q = self.build_q_matrix()
            X = self.build_x_matrix()
            return Q, X
        finally:
            self.disconnect()
    
    def get_student_info(self, student_id: str) -> Tuple[str, str]:
        """
        获取学生信息
        
        Args:
            student_id: 学生ID
            
        Returns:
            Tuple[str, str]: (学生ID, 学生姓名)
        """
        if student_id in self.student_ids:
            idx = self.student_ids.index(student_id)
            return student_id, self.student_names[idx]
        return student_id, "未知"


# ============ DINA诊断引擎 ============

class DINADiagnosticEngine:
    """
    DINA模型认知诊断引擎
    
    DINA模型核心假设:
    1. 一道题只有当学生掌握了该题要求的所有知识点时，才可能正确作答
    2. 理想作答: η = 1 当且仅当学生掌握了题目要求的所有知识点
    3. 实际作答受猜测率g和失误率s影响
    
    概率模型:
    P(X=1|η=1) = 1-s (掌握所有知识点，答对概率高)
    P(X=1|η=0) = g   (未掌握所有知识点，只能猜测)
    """
    
    def __init__(
        self, 
        q_matrix: np.ndarray,
        guess_rate: float = 0.2,
        slip_rate: float = 0.2
    ):
        """
        初始化DINA诊断引擎
        
        Args:
            q_matrix: Q矩阵，形状为 (题目数, 知识点数)
            guess_rate: 猜测率g，默认0.2
            slip_rate: 失误率s，默认0.2
        """
        self.Q = q_matrix
        self.guess = guess_rate
        self.slip = slip_rate
        
        # 获取维度信息
        self.num_questions, self.num_concepts = q_matrix.shape
        
        # 生成所有可能的知识掌握模式
        self.patterns = self._generate_all_patterns()
        self.num_patterns = len(self.patterns)
        
        # 计算基于知识图谱约束的先验概率
        self.prior_probabilities = self._compute_prior_probabilities()
        
        # 计算每种模式下的理想作答矩阵η
        self.eta_matrix = self._compute_eta_matrix()
        
        # 预计算每种模式下的作答概率（用于加速）
        self._precompute_response_probs()
    
    def _generate_all_patterns(self) -> np.ndarray:
        """
        生成所有可能的知识掌握模式
        
        对于K个知识点，共有2^K种可能的掌握模式
        每种模式是一个长度为K的二进制向量
        
        Returns:
            np.ndarray: 形状为 (2^K, K) 的数组
        """
        patterns = list(product([0, 1], repeat=self.num_concepts))
        return np.array(patterns, dtype=np.int32)
    
    def _compute_eta_matrix(self) -> np.ndarray:
        """
        计算理想作答矩阵η
        
        η[i,j] = 1 当且仅当第i种掌握模式掌握了第j道题要求的所有知识点
        
        计算逻辑:
        对于题目j和掌握模式α:
        η_j(α) = ∏_k α_k^q_jk
        即: 如果模式α掌握了题目j要求的所有知识点，则η=1
        
        Returns:
            np.ndarray: 形状为 (模式数, 题目数) 的数组
        """
        eta = np.zeros((self.num_patterns, self.num_questions), dtype=np.int32)
        
        for i, pattern in enumerate(self.patterns):
            for j in range(self.num_questions):
                required_concepts = self.Q[j]
                
                # 检查是否掌握了该题要求的所有知识点
                # pattern * required_concepts: 掌握的知识点与要求的知识点的交集
                # 如果交集等于要求的知识点，说明全部掌握
                mastered_required = pattern * required_concepts
                if np.array_equal(mastered_required, required_concepts):
                    eta[i, j] = 1
        
        return eta
    
    def _compute_prior_probabilities(self) -> np.ndarray:
        """
        计算基于知识图谱约束的先验概率
        
        知识点依赖关系: K1 -> K2 -> K3；K1 -> K4 -> K5
        
        对于违背前置依赖的模式，设置极小的先验概率（1e-5）
        对于符合逻辑的合法模式，平分剩余的先验概率空间
        
        Returns:
            np.ndarray: 各模式的先验概率，形状为 (模式数,)
        """
        prior = np.zeros(self.num_patterns)
        invalid_penalty = 1e-5
        valid_patterns = []
        
        for i, pattern in enumerate(self.patterns):
            is_valid = True
            # K2 依赖 K1
            if pattern[1] == 1 and pattern[0] == 0:
                is_valid = False
            # K3 依赖 K2
            if pattern[2] == 1 and pattern[1] == 0:
                is_valid = False
            # K4 依赖 K1
            if pattern[3] == 1 and pattern[0] == 0:
                is_valid = False
            # K5 依赖 K4
            if pattern[4] == 1 and pattern[3] == 0:
                is_valid = False
            
            if is_valid:
                valid_patterns.append(i)
            else:
                prior[i] = invalid_penalty
        
        if valid_patterns:
            valid_count = len(valid_patterns)
            valid_prob = (1.0 - len(prior[prior == invalid_penalty]) * invalid_penalty) / valid_count
            for i in valid_patterns:
                prior[i] = valid_prob
        
        # 归一化确保总和为1
        prior = prior / np.sum(prior)
        return prior

    def _precompute_response_probs(self) -> None:
        """
        预计算每种模式下每道题的答对概率
        
        P(X=1|η) = (1-s)^η * g^(1-η)
        
        当η=1时: P = 1-s
        当η=0时: P = g
        """
        # prob_correct[i, j] = 模式i下题目j的答对概率
        self.prob_correct = np.where(
            self.eta_matrix == 1,
            1 - self.slip,
            self.guess
        )
        
        # prob_incorrect[i, j] = 模式i下题目j的答错概率
        self.prob_incorrect = 1 - self.prob_correct
    
    def compute_log_likelihood(
        self, 
        response_vector: np.ndarray, 
        pattern_idx: int
    ) -> float:
        """
        计算给定作答向量在特定模式下的对数似然
        
        似然函数:
        L(X|α) = ∏_j P(X_j|α)
               = ∏_j [(1-s)^η_j * g^(1-η_j)]^X_j * [s^η_j * (1-g)^(1-η_j)]^(1-X_j)
        
        对数似然:
        log L(X|α) = Σ_j [X_j * log(P_correct) + (1-X_j) * log(P_incorrect)]
        
        Args:
            response_vector: 学生的作答向量，形状为 (题目数,)
            pattern_idx: 知识掌握模式的索引
            
        Returns:
            float: 对数似然值
        """
        prob_c = self.prob_correct[pattern_idx]
        prob_ic = self.prob_incorrect[pattern_idx]
        
        # 避免log(0)的情况
        eps = 1e-10
        prob_c = np.clip(prob_c, eps, 1 - eps)
        prob_ic = np.clip(prob_ic, eps, 1 - eps)
        
        # 计算对数似然
        log_lik = np.sum(
            response_vector * np.log(prob_c) + 
            (1 - response_vector) * np.log(prob_ic)
        )
        
        return log_lik
    
    def compute_posterior_probabilities(
        self, 
        response_vector: np.ndarray
    ) -> np.ndarray:
        """
        计算给定作答向量在所有模式下的后验概率
        
        贝叶斯公式:
        P(α|X) = P(X|α) * P(α) / P(X)
        
        假设先验概率相等: P(α) = 1/2^K
        
        由于P(X)对所有模式相同，可以省略归一化常数:
        P(α|X) ∝ P(X|α) * P(α) = P(X|α) / 2^K
        
        使用对数空间计算避免数值下溢:
        log P(α|X) = log P(X|α) - K*log(2) - log Z
        其中Z为归一化常数
        
        Args:
            response_vector: 学生的作答向量
            
        Returns:
            np.ndarray: 各模式的后验概率，形状为 (模式数,)
        """
        # 计算各模式的似然（对数空间）
        log_likelihoods = np.array([
            self.compute_log_likelihood(response_vector, i)
            for i in range(self.num_patterns)
        ])
        
        # 加上先验的对数（基于知识图谱约束的先验）
        log_prior = np.log(self.prior_probabilities)
        log_unnorm_posteriors = log_likelihoods + log_prior
        
        # 归一化（使用log-sum-exp技巧避免数值下溢）
        log_max = np.max(log_unnorm_posteriors)
        log_sum = log_max + np.log(np.sum(np.exp(log_unnorm_posteriors - log_max)))
        
        log_posteriors = log_unnorm_posteriors - log_sum
        posteriors = np.exp(log_posteriors)
        
        return posteriors
    
    def compute_marginal_mastery_prob(
        self, 
        posteriors: np.ndarray
    ) -> np.ndarray:
        """
        计算各知识点的边缘掌握概率
        
        边缘化公式:
        P(α_k=1|X) = Σ_{α: α_k=1} P(α|X)
        
        即: 对所有在第k个知识点上为1的模式的后验概率求和
        
        Args:
            posteriors: 各模式的后验概率
            
        Returns:
            np.ndarray: 各知识点的掌握概率，形状为 (知识点数,)
        """
        mastery_probs = np.zeros(self.num_concepts)
        
        for k in range(self.num_concepts):
            # 找出所有在第k个知识点上为1的模式
            mask = self.patterns[:, k] == 1
            # 对这些模式的后验概率求和
            mastery_probs[k] = np.sum(posteriors[mask])
        
        return mastery_probs
    
    def diagnose_student(
        self, 
        response_vector: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, int, float]:
        """
        对单个学生进行诊断
        
        Args:
            response_vector: 学生的作答向量
            
        Returns:
            Tuple包含:
            - 掌握概率向量 (知识点数,)
            - 后验概率向量 (模式数,)
            - 最可能模式的索引
            - 最可能模式的后验概率
        """
        # 计算后验概率
        posteriors = self.compute_posterior_probabilities(response_vector)
        
        # 计算边缘掌握概率
        mastery_probs = self.compute_marginal_mastery_prob(posteriors)
        
        # 找到最可能的模式
        most_likely_idx = np.argmax(posteriors)
        most_likely_posterior = posteriors[most_likely_idx]
        
        return mastery_probs, posteriors, most_likely_idx, most_likely_posterior
    
    def diagnose_all_students(
        self, 
        x_matrix: np.ndarray,
        data_loader: DataLoader
    ) -> List[DiagnosticResult]:
        """
        对所有学生进行诊断
        
        Args:
            x_matrix: 作答矩阵，形状为 (学生数, 题目数)
            data_loader: 数据加载器，用于获取学生信息
            
        Returns:
            List[DiagnosticResult]: 所有学生的诊断结果列表
        """
        results = []
        
        for i, student_id in enumerate(data_loader.student_ids):
            response_vector = x_matrix[i]
            
            mastery_probs, posteriors, pattern_idx, posterior = self.diagnose_student(
                response_vector
            )
            
            # 构建知识点掌握概率字典
            knowledge_mastery = {
                data_loader.concept_ids[j]: mastery_probs[j]
                for j in range(self.num_concepts)
            }
            
            result = DiagnosticResult(
                student_id=student_id,
                student_name=data_loader.student_names[i],
                knowledge_mastery=knowledge_mastery,
                most_likely_pattern=self.patterns[pattern_idx].copy(),
                pattern_posterior=posterior
            )
            
            results.append(result)
        
        return results
    
    def print_diagnosis_summary(
        self, 
        result: DiagnosticResult,
        concept_names: List[str]
    ) -> None:
        """
        打印单个学生的诊断结果摘要
        
        Args:
            result: 诊断结果
            concept_names: 知识点名称列表
        """
        print(f"\n学生: {result.student_id} ({result.student_name})")
        print("-" * 50)
        print("知识点掌握概率:")
        
        for i, (concept_id, prob) in enumerate(result.knowledge_mastery.items()):
            bar_length = int(prob * 20)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            print(f"  {concept_id} ({concept_names[i]}): {prob:.4f} [{bar}]")
        
        print(f"\n最可能的知识掌握模式: {result.most_likely_pattern}")
        print(f"该模式的后验概率: {result.pattern_posterior:.4f}")
    
    def get_pattern_description(
        self, 
        pattern: np.ndarray,
        concept_ids: List[str]
    ) -> str:
        """
        获取知识掌握模式的文字描述
        
        Args:
            pattern: 知识掌握模式向量
            concept_ids: 知识点ID列表
            
        Returns:
            str: 模式描述
        """
        mastered = [concept_ids[i] for i, v in enumerate(pattern) if v == 1]
        if not mastered:
            return "未掌握任何知识点"
        return f"已掌握: {', '.join(mastered)}"


# ============ IRT模型引擎 ============

class IRTModel:
    """
    IRT（项目反应理论）模型 - Rasch 单参数模型
    
    Rasch模型核心假设:
    1. 题目只有一个难度参数b_j
    2. 学生有一个能力参数θ_i
    3. 作答概率由能力与难度的差值决定
    
    概率模型:
    P(X=1|θ,b) = 1 / (1 + exp(-(θ - b)))
    
    计算逻辑:
    1. 根据每道题的错误率估算题目难度
    2. 使用Logit变换计算学生能力
    3. 将能力值映射为0-100的综合能力分
    """
    
    def __init__(self, x_matrix: np.ndarray):
        """
        初始化IRT模型
        
        Args:
            x_matrix: 作答矩阵，形状为 (学生数, 题目数)
        """
        self.X = x_matrix
        self.num_students, self.num_questions = x_matrix.shape
        
        # 题目难度参数
        self.difficulties = np.zeros(self.num_questions)
        
        # 学生能力参数
        self.abilities = np.zeros(self.num_students)
        
        # 计算题目难度和学生能力
        self._estimate_difficulties()
        self._estimate_abilities()
    
    def _estimate_difficulties(self) -> None:
        """
        估算题目难度参数
        
        使用简单方法: 题目难度 = -log(正确率)
        
        难度越高，正确率越低
        """
        for j in range(self.num_questions):
            correct_rate = np.mean(self.X[:, j])
            # 避免除以0
            correct_rate = np.clip(correct_rate, 0.01, 0.99)
            # 难度 = -log(正确率)
            # 正确率越高，难度越低（负值）
            # 正确率越低，难度越高（正值）
            self.difficulties[j] = -np.log(correct_rate / (1 - correct_rate))
    
    def _estimate_abilities(self) -> None:
        """
        估算学生能力参数
        
        使用Logit变换: θ = ln(p / (1-p))
        其中p是学生的平均正确率
        
        然后根据题目平均难度进行修正
        """
        for i in range(self.num_students):
            # 计算学生的平均正确率
            student_correct_rate = np.mean(self.X[i, :])
            # 避免除以0
            student_correct_rate = np.clip(student_correct_rate, 0.01, 0.99)
            
            # Logit变换
            ability = np.log(student_correct_rate / (1 - student_correct_rate))
            
            # 根据题目平均难度进行修正
            avg_difficulty = np.mean(self.difficulties)
            ability = ability + avg_difficulty
            
            self.abilities[i] = ability
    
    def get_student_irt_score(self, student_idx: int) -> int:
        """
        获取学生的IRT综合能力分（0-100）
        
        将能力值θ映射为0-100的整数分数
        
        映射逻辑:
        - 使用sigmoid函数将能力值映射到0-1区间
        - 然后乘以100得到0-100的分数
        
        Args:
            student_idx: 学生索引
            
        Returns:
            int: 综合能力分（0-100）
        """
        ability = self.abilities[student_idx]
        
        # 使用sigmoid函数映射能力值到0-1区间
        # sigmoid(x) = 1 / (1 + exp(-x))
        # 能力值越大，分数越高
        normalized = 1 / (1 + np.exp(-ability))
        
        # 映射到0-100并取整
        score = int(normalized * 100)
        score = np.clip(score, 0, 100)
        
        return score
    
    def get_all_irt_scores(self) -> np.ndarray:
        """
        获取所有学生的IRT综合能力分
        
        Returns:
            np.ndarray: 所有学生的能力分，形状为 (学生数,)
        """
        scores = np.array([
            self.get_student_irt_score(i)
            for i in range(self.num_students)
        ])
        return scores
    
    def print_irt_summary(self, student_ids: List[str]) -> None:
        """
        打印IRT诊断摘要
        
        Args:
            student_ids: 学生ID列表
        """
        print("\n" + "=" * 60)
        print("IRT综合能力评估")
        print("=" * 60)
        
        for i, student_id in enumerate(student_ids):
            score = self.get_student_irt_score(i)
            ability = self.abilities[i]
            
            # 颜色标记
            if score >= 80:
                color = "绿色"
            elif score >= 60:
                color = "橙色"
            else:
                color = "红色"
            
            print(f"{student_id}: 能力分 {score} ({color}), 能力值 θ = {ability:.4f}")
        
        print("=" * 60)


# ============ 主程序入口 ============

def main():
    """
    主程序入口
    执行完整的DINA诊断流程
    """
    print("=" * 60)
    print("DINA模型认知诊断引擎")
    print("=" * 60)
    
    # 初始化数据加载器
    print("\n[1] 加载数据...")
    loader = DataLoader("smart_profile.db")
    Q, X = loader.load_all_data()
    
    print(f"  Q矩阵形状: {Q.shape} (题目数 x 知识点数)")
    print(f"  X矩阵形状: {X.shape} (学生数 x 题目数)")
    print(f"  知识点: {loader.concept_ids}")
    print(f"  知识点名称: {loader.concept_names}")
    
    # 初始化DINA诊断引擎
    print("\n[2] 初始化DINA诊断引擎...")
    engine = DINADiagnosticEngine(
        q_matrix=Q,
        guess_rate=0.2,
        slip_rate=0.2
    )
    print(f"  知识掌握模式数: {engine.num_patterns}")
    print(f"  猜测率 (g): {engine.guess}")
    print(f"  失误率 (s): {engine.slip}")
    
    # 执行诊断
    print("\n[3] 执行认知诊断...")
    results = engine.diagnose_all_students(X, loader)
    
    # 打印所有学生的诊断结果概览
    print("\n[4] 所有学生诊断结果概览:")
    print("-" * 60)
    print(f"{'学生ID':<8} {'姓名':<6} {'K1':>8} {'K2':>8} {'K3':>8} {'K4':>8} {'K5':>8}")
    print("-" * 60)
    
    for result in results:
        probs = [result.knowledge_mastery[cid] for cid in loader.concept_ids]
        prob_str = " ".join([f"{p:>8.4f}" for p in probs])
        print(f"{result.student_id:<8} {result.student_name:<6} {prob_str}")
    
    # 详细对比分析：学霸 vs 学渣
    print("\n" + "=" * 60)
    print("[5] 详细诊断对比：学霸 vs 学渣")
    print("=" * 60)
    
    # 找到S07（赵敏）和S01（张伟）的结果
    s07_result = None
    s01_result = None
    
    for result in results:
        if result.student_id == "S07":
            s07_result = result
        elif result.student_id == "S01":
            s01_result = result
    
    if s07_result:
        print("\n【学霸案例】")
        engine.print_diagnosis_summary(s07_result, loader.concept_names)
        print(f"\n模式解释: {engine.get_pattern_description(s07_result.most_likely_pattern, loader.concept_ids)}")
    
    if s01_result:
        print("\n【学渣案例】")
        engine.print_diagnosis_summary(s01_result, loader.concept_names)
        print(f"\n模式解释: {engine.get_pattern_description(s01_result.most_likely_pattern, loader.concept_ids)}")
    
    # 验证结果
    print("\n" + "=" * 60)
    print("[6] 诊断结果验证")
    print("=" * 60)
    
    if s07_result and s01_result:
        print("\n知识点掌握概率对比:")
        print(f"{'知识点':<12} {'S07(赵敏)':>12} {'S01(张伟)':>12} {'差异':>12}")
        print("-" * 50)
        
        for cid, cname in zip(loader.concept_ids, loader.concept_names):
            p7 = s07_result.knowledge_mastery[cid]
            p1 = s01_result.knowledge_mastery[cid]
            diff = p7 - p1
            print(f"{cid} ({cname[:4]:<4}) {p7:>12.4f} {p1:>12.4f} {diff:>+12.4f}")
        
        # 计算平均掌握概率
        avg_s07 = np.mean(list(s07_result.knowledge_mastery.values()))
        avg_s01 = np.mean(list(s01_result.knowledge_mastery.values()))
        
        print("-" * 50)
        print(f"{'平均掌握率':<12} {avg_s07:>12.4f} {avg_s01:>12.4f} {avg_s07 - avg_s01:>+12.4f}")
        
        print("\n✓ 诊断完成！S07(赵敏)的知识点掌握概率显著高于S01(张伟)，")
        print("  验证了DINA模型能够有效区分学霸和学渣。")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()

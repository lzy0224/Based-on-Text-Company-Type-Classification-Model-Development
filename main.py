import pandas as pd
import numpy as np
import re
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# sklearn 核心库
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class CompanyTypeClassifier:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.best_model = None
        self.model_filename = 'best_company_classifier_pipeline.pkl'

    def _clean_text(self, text):
        """
        文本清洗函数
        1. 转字符串
        2. 去除特殊符号，仅保留中文、英文、数字
        3. 字符间加空格（模拟分词，方便TF-IDF处理）
        """
        if pd.isna(text):
            return ""
        text = str(text).strip()
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return ' '.join(list(text))

    def load_and_preprocess(self):
   
        print("\n" + "="*50)
        print(">>> 步骤 1: 数据加载与清洗")
        print("="*50)

        # 1.1 智能加载 CSV 或 Excel
        try:
            if self.file_path.endswith('.csv'):
                self.df = pd.read_csv(self.file_path)
            else:
                self.df = pd.read_excel(self.file_path)
            
            # 1.2 统一列名 (假设第一列是标签，第二列是文本)
            cols = self.df.columns
            if len(cols) >= 2:
                self.df = self.df.rename(columns={cols[0]: 'label', cols[1]: 'text'})
                self.df = self.df[['label', 'text']] # 只保留前两列
            
            print(f"原始数据量: {len(self.df)}")

            # 1.3 去除空值
            self.df.dropna(subset=['text', 'label'], inplace=True)
            
            # 1.4 应用文本清洗
            print("正在清洗文本...")
            self.df['clean_text'] = self.df['text'].apply(self._clean_text)
            
            # 去除清洗后为空的行
            self.df = self.df[self.df['clean_text'].str.len() > 0]
            
            print(f"有效数据量: {len(self.df)}")
            print(f"类别分布:\n{self.df['label'].value_counts().sort_index()}")
            
        except FileNotFoundError:
            print(f"❌ 错误: 文件 {self.file_path} 未找到。")
            raise
        except Exception as e:
            print(f"❌ 数据加载错误: {e}")
            raise

    def split_data(self):
        """步骤2：数据划分 (80% 训练, 20% 验证)"""
        print("\n" + "="*50)
        print(">>> 步骤 2: 数据划分 (防止数据泄露)")
        print("="*50)

        # 使用 stratify 保证训练集和测试集类别分布一致
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.df['clean_text'], 
            self.df['label'], 
            test_size=0.2, 
            random_state=42, 
            stratify=self.df['label']
        )
        
        print(f"训练集大小: {len(self.X_train)}")
        print(f"验证集大小: {len(self.X_test)}")

    def train_and_optimize(self):
        """步骤3：模型训练与超参数优化"""
        print("\n" + "="*50)
        print(">>> 步骤 3: 模型训练与交叉验证")
        print("="*50)

        # 定义 Pipeline
        # 说明：Pipeline 将向量化和分类器打包，fit时只计算训练集数据的TF-IDF，防止泄露
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                token_pattern=r'(?u)\b\w+\b', 
                max_features=5000,
                ngram_range=(1, 2)  # 关键：同时提取单字和双字词组特征
            )),
            ('clf', SVC(class_weight='balanced', probability=True)) # 默认使用 SVM
        ])

        # 定义超参数搜索空间
        param_grid = [
            {
                'clf': [SVC(class_weight='balanced', probability=True, kernel='linear')],
                'clf__C': [1, 10],  # 惩罚系数
                'tfidf__max_features': [3000, 5000]
            },
            {
                'clf': [LogisticRegression(class_weight='balanced', max_iter=1000)],
                'clf__C': [1, 10],
                'tfidf__max_features': [3000, 5000]
            }
        ]

        print("开始网格搜索 (GridSearchCV) 寻找最佳模型...")
        # 5折交叉验证
        grid_search = GridSearchCV(
            pipeline, 
            param_grid, 
            cv=5, 
            scoring='accuracy',
            n_jobs=-1, # 并行计算
            verbose=1
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        self.best_model = grid_search.best_estimator_
        
        print(f"\n✅ 交叉验证最佳准确率: {grid_search.best_score_:.4f}")
        print(f"最佳参数: {grid_search.best_params_}")
        
        # 检查是否达标
        if grid_search.best_score_ < 0.8:
            print("⚠️ 注意: 训练集交叉验证准确率略低于 80%，后续可能需要增加数据或调整特征。")
        else:
            print("✅ 训练阶段指标达标 (>=80%)。")

    def evaluate_final(self):
        """步骤4：最终验证集评估"""
        print("\n" + "="*50)
        print(">>> 步骤 4: 最终验证集 (Hold-out Test) 评估")
        print("="*50)
        
        # 预测
        y_pred = self.best_model.predict(self.X_test)
        
        # 计算指标
        acc = accuracy_score(self.y_test, y_pred)
        
        print(f"测试集准确率 (Accuracy): {acc:.4f}")
        print("\n详细分类报告:")
        print(classification_report(self.y_test, y_pred))
        
        # 绘制混淆矩阵
        plt.figure(figsize=(12, 10))
        cm = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'混淆矩阵 (Accuracy: {acc:.2%})')
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        plt.tight_layout()
        plt.show()
        
        if acc >= 0.8:
            print(f"\n🎉 恭喜！模型最终准确率 {acc:.2%} >= 80%，符合交付标准。")
            return True
        else:
            print(f"\n⚠️ 模型最终准确率 {acc:.2%} 未达到 80%。建议检查数据质量或尝试深度学习模型。")
            return False

    def save_pipeline(self):
        """步骤5：保存模型管道"""
        print("\n" + "="*50)
        print(">>> 步骤 5: 保存模型与交付")
        print("="*50)
        
        joblib.dump(self.best_model, self.model_filename)
        print(f"✅ 模型全流程管道已保存至: {self.model_filename}")
        print("该文件包含：预处理规则 + TF-IDF向量化器 + 训练好的分类器")

    def predict_new_data(self, texts):
        """对外接口：预测新文本"""
        print("\n>>> 模拟新数据预测:")
        
        # 加载模型 (如果是重新运行脚本)
        if self.best_model is None:
            try:
                self.best_model = joblib.load(self.model_filename)
            except:
                print("未找到保存的模型，请先训练。")
                return

        # 清洗输入
        clean_texts = [self._clean_text(t) for t in texts]
        
        # 预测 (Pipeline 会自动处理 TF-IDF)
        preds = self.best_model.predict(clean_texts)
        probs = self.best_model.predict_proba(clean_texts)
        
        for i, (text, pred, prob) in enumerate(zip(texts, preds, probs)):
            print(f"-"*30)
            print(f"文本: {text[:30]}...")
            print(f"预测类别: {pred}")
            print(f"置信度: {np.max(prob):.4f}")

# ==========================================
# 主程序执行入口
# ==========================================
if __name__ == "__main__":
    # 1. 设置文件路径 (请确保 training.xlsx 或 training.csv 在同级目录)
    DATA_FILE = 'training.xlsx' 
    
 
    if not os.path.exists(DATA_FILE) and not os.path.exists('training.csv'):
        print("⚠️ 未检测到数据文件，正在生成模拟数据用于代码测试...")
        dummy_data = {
            'Column1': np.random.randint(1, 12, 200),
            'Column2': ['某某科技公司专注于软件开发 ' + str(i) for i in range(200)]
        }
        pd.DataFrame(dummy_data).to_excel(DATA_FILE, index=False)
    
    # 2. 实例化工作流
    classifier = CompanyTypeClassifier(DATA_FILE)
    
    try:
        # 3. 依次执行任务
        classifier.load_and_preprocess() # 加载清洗
        classifier.split_data()          # 划分数据
        classifier.train_and_optimize()  # 训练优化
        classifier.evaluate_final()      # 最终评估
        classifier.save_pipeline()       # 保存模型
        
        # 4. 演示预测
        test_samples = [
            "本公司专业从事房地产开发与物业管理服务，致力于打造高端住宅。",
            "公司主要业务为软件技术开发、互联网信息服务及大数据分析。",
            "提供专业的金融投资咨询、股权私募及资产管理服务。"
        ]
        classifier.predict_new_data(test_samples)
        
    except Exception as e:
        print(f"\n❌ 程序执行中断: {e}")
        import traceback
        traceback.print_exc()
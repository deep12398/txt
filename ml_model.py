import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import load_iris
import joblib
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

class IrisClassifier:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.target_names = None
        
    def load_data(self):
        # 加载鸢尾花数据集
        print("加载数据...")
        iris = load_iris()
        X = iris.data
        y = iris.target
        self.feature_names = iris.feature_names
        self.target_names = iris.target_names
        
        # 创建DataFrame以便更好地理解数据
        df = pd.DataFrame(X, columns=self.feature_names)
        df['species'] = [self.target_names[i] for i in y]
        
        print(f"数据集形状: {X.shape}")
        print(f"目标分类: {self.target_names}")
        print("\n数据集预览:")
        print(df.head())
        print("\n统计摘要:")
        print(df.describe())
        
        return X, y, df
    
    def explore_data(self, df):
        print("\n开始数据探索...")
        
        # 绘制数据分布
        plt.figure(figsize=(12, 10))
        
        for i, feature in enumerate(self.feature_names):
            plt.subplot(2, 2, i+1)
            for species in self.target_names:
                subset = df[df['species'] == species]
                plt.hist(subset[feature], alpha=0.7, label=species)
            plt.title(f'{feature} 分布')
            plt.xlabel(feature)
            plt.ylabel('频率')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig('iris_distributions.png')
        
        # 绘制特征相关性热图
        plt.figure(figsize=(10, 8))
        correlation = df.drop('species', axis=1).corr()
        sns.heatmap(correlation, annot=True, cmap='coolwarm')
        plt.title('特征相关性矩阵')
        plt.savefig('iris_correlation.png')
        
        # 散点矩阵图
        sns.pairplot(df, hue='species')
        plt.savefig('iris_pairplot.png')
        
        print("探索完成，图表已保存")
    
    def preprocess_data(self, X, y):
        print("\n开始数据预处理...")
        # 分割训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        
        # 标准化特征
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"训练集大小: {X_train.shape}")
        print(f"测试集大小: {X_test.shape}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_model(self, X_train, y_train):
        print("\n开始训练模型...")
        
        # 定义参数网格
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }
        
        # 使用网格搜索找最佳参数
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(
            rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        # 获取最佳模型
        self.model = grid_search.best_estimator_
        
        print(f"最佳参数: {grid_search.best_params_}")
        print(f"交叉验证最佳分数: {grid_search.best_score_:.4f}")
        
        # 特征重要性
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print("\n特征重要性:")
        for i in range(len(self.feature_names)):
            print(f"{self.feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
    
    def evaluate_model(self, X_test, y_test):
        print("\n评估模型...")
        
        # 预测测试集
        y_pred = self.model.predict(X_test)
        
        # 计算准确率
        accuracy = accuracy_score(y_test, y_pred)
        print(f"准确率: {accuracy:.4f}")
        
        # 混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=self.target_names,
            yticklabels=self.target_names
        )
        plt.xlabel('预测')
        plt.ylabel('实际')
        plt.title('混淆矩阵')
        plt.savefig('confusion_matrix.png')
        
        # 分类报告
        print("\n分类报告:")
        print(classification_report(y_test, y_pred, target_names=self.target_names))
    
    def save_model(self, filename='iris_classifier.joblib'):
        print(f"\n保存模型到 {filename}...")
        model_info = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'target_names': self.target_names
        }
        joblib.dump(model_info, filename)
        print("模型已保存")
    
    def predict_sample(self, sample):
        """预测一个样本的类别"""
        if not isinstance(sample, np.ndarray):
            sample = np.array(sample).reshape(1, -1)
        
        scaled_sample = self.scaler.transform(sample)
        prediction = self.model.predict(scaled_sample)
        probabilities = self.model.predict_proba(scaled_sample)
        
        species = self.target_names[prediction[0]]
        probs = {self.target_names[i]: p for i, p in enumerate(probabilities[0])}
        
        return species, probs

def main():
    classifier = IrisClassifier()
    
    # 加载并探索数据
    X, y, df = classifier.load_data()
    classifier.explore_data(df)
    
    # 预处理数据
    X_train, X_test, y_train, y_test = classifier.preprocess_data(X, y)
    
    # 训练模型
    classifier.train_model(X_train, y_train)
    
    # 评估模型
    classifier.evaluate_model(X_test, y_test)
    
    # 保存模型
    classifier.save_model()
    
    # 测试预测
    # 创建一个示例: (sepal length, sepal width, petal length, petal width)
    test_sample = [5.1, 3.5, 1.4, 0.2]  # 这是一个setosa的例子
    species, probabilities = classifier.predict_sample(test_sample)
    
    print(f"\n示例预测:")
    print(f"样本: {test_sample}")
    print(f"预测类别: {species}")
    print("预测概率:")
    for species, prob in probabilities.items():
        print(f"  {species}: {prob:.4f}")

if __name__ == "__main__":
    print("鸢尾花分类器 - 机器学习示例")
    print("=" * 50)
    main()

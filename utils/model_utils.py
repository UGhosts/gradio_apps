import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class FaultClassifierPipeline:
    """内置标准化的分类器Pipeline"""
    
    def __init__(self, model, scaler=None):
        self.model = model
        self.scaler = scaler if scaler else StandardScaler()
        self.is_fitted = False
    
    def fit(self, X, y, **kwargs):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y, **kwargs)
        self.is_fitted = True
        return self
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X_scaled)
        return None

class FaultClassifierInference:
    """故障分类模型推理类"""
    
    def __init__(self, model_path, metadata_path):
        self.model = joblib.load(model_path)
        print(f"✓ 模型加载成功: {model_path}")
        
        metadata = joblib.load(metadata_path)
        self.label_mapping = metadata['label_mapping']
        self.reverse_mapping = metadata['reverse_mapping']
        self.feature_names = metadata['feature_names']
        self.model_name = metadata['model_name']
        self.test_f1 = metadata.get('test_f1', 'N/A')
    
    def prepare_features(self, df):
        """准备特征，确保顺序和模型训练时一致"""
        if isinstance(df, pd.DataFrame):
            # 确保特征顺序与训练时一致
            missing_cols = set(self.feature_names) - set(df.columns)
            if missing_cols:
                raise ValueError(f"缺少特征列: {missing_cols}")
            X = df[self.feature_names]
        else:
            X = df
        return X

    def predict_batch(self, samples, return_proba=False):
        """批量预测"""
        if isinstance(samples, pd.DataFrame):
            X = self.prepare_features(samples)
        else:
            X = np.array(samples)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        y_pred = self.model.predict(X)
        fault_codes = [self.reverse_mapping[pred] for pred in y_pred]
        
        results = pd.DataFrame({
            'fault_code': fault_codes,
            'encoded_label': y_pred
        })
        
        if return_proba:
            try:
                proba = self.model.predict_proba(X)
                results['confidence'] = proba.max(axis=1)
                
                num_model_classes = proba.shape[1]
                for i in range(num_model_classes):
                    label = self.reverse_mapping.get(i, f'未知类别_{i}')
                    results[f'prob_{label}'] = proba[:, i]
            except Exception as e:
                print(f"  警告：无法获取概率。错误: {e}")
                results['confidence'] = None
        
        return results
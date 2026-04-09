# -*- coding: utf-8 -*-
"""
风控模型统一训练脚本
支持：短信NLP模型、APP NLP模型、评分卡模型（LightGBM/XGBoost/Logistic等）
逻辑回归：WOE分箱、IV筛选、VIF多重共线性、相关性筛选
树模型：直接使用原始特征，导出PMML
"""

import pandas as pd
import numpy as np
import os
import sys
import json
import joblib
import warnings
import random
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from .woe import WOEBinning
from .exporters import (
    export_pmml,
    export_scorecard_excel,
    export_training_report,
    plot_feature_importance as export_plot_feature_importance,
    plot_roc_curve as export_plot_roc_curve,
    save_model_artifacts,
)

warnings.filterwarnings('ignore')

try:
    import lightgbm as lgb
    import xgboost as xgb
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError as e:
    print(f"警告: 缺少依赖库 - {e}")

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.isotonic import IsotonicRegression
from scipy.stats import ks_2samp
import statsmodels.api as sm

try:
    from sklearn2pmml import PMMLPipeline, sklearn2pmml
    from sklearn_pandas import DataFrameMapper
    PMML_AVAILABLE = True
except ImportError:
    PMML_AVAILABLE = False
    print("警告: sklearn2pmml未安装，PMML导出功能不可用")


class ModelTrainer:
    """风控模型统一训练器"""
    
    def __init__(self, config=None, field_config=None):
        self.config = config or self._default_config()
        self.field_config = field_config or self._load_field_config()
        self.model = None
        self.var_list = None
        self.woe_binner = None
        self.train_log = {}
        
    def _load_field_config(self, field_config_path='configs/field_config.json'):
        """加载字段配置文件"""
        import json
        from pathlib import Path
        field_config = {
            "fields": {
                "include": [],
                "exclude": [],
                "target": "target"
            },
            "field_groups": {},
            "field_rules": {
                "missing_threshold": 0.95,
                "min_unique_values": 2,
                "max_unique_values": None
            },
            "data_types": {
                "numeric": [],
                "categorical": [],
                "date": []
            }
        }
        config_path = Path(field_config_path)
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    field_config = json.load(f)
                print(f"已加载字段配置文件: {field_config_path}")
            except Exception as e:
                print(f"加载字段配置文件失败: {e}")
        return field_config
    
    def _default_config(self):
        return {
            'seed': 1234,
            'output_dir': './output',
            'model_prefix': 'model',
            
            'scorecard': {
                'y_name': 'target',
                'id_var': 'order_id',
                'date_col': 'endDateDuration',
                'test_size': 0.2,
                
                'oot_size': {'low_num': 1000, 'high_num': 3000, 'threshold': 7000},
                
                'feature_selection': {
                    'missing_threshold': 0.95,
                    'ks_threshold': 0.02,
                    'max_features': 20,
                    'min_features': 10,
                },
                
                'logistic': {
                    'iv_threshold': 0.02,
                    'vif_threshold': 5.0,
                    'corr_threshold': 0.7,
                    'woe_bins': 10,
                },
                
                'tuning': {
                    'enable': True,
                    'n_trials': 30,
                    'target': 'ks',
                },
                
                'scorecard_params': {
                    'pdo': 40,
                    'base_score': 600,
                    'base_odds': 20,
                },
            },
        }
    
    def set_seed(self, seed=None):
        seed = seed or self.config['seed']
        np.random.seed(seed)
        random.seed(seed)
    
    def preprocess_data(self, data):
        """根据字段配置文件进行数据预处理"""
        # 获取字段配置
        include_fields = self.field_config['fields']['include']
        exclude_fields = self.field_config['fields']['exclude']
        target_field = self.field_config['fields']['target']
        field_rules = self.field_config['field_rules']
        
        # 1. 应用包含/排除规则
        if include_fields:
            # 如果指定了包含字段，只保留这些字段
            data = data[[col for col in include_fields if col in data.columns] + [target_field]]
        else:
            # 否则排除指定的字段
            data = data[[col for col in data.columns if col not in exclude_fields]]
        
        # 2. 应用字段规则
        selected_columns = [target_field]
        field_stats = []
        
        for col in data.columns:
            if col == target_field:
                continue
            
            # 检查缺失值
            missing_rate = data[col].isnull().mean()
            if missing_rate > field_rules['missing_threshold']:
                print(f"排除字段 {col}: 缺失率过高 ({missing_rate:.2f})")
                continue
            
            # 检查唯一值数量
            unique_count = data[col].nunique()
            if unique_count < field_rules['min_unique_values']:
                print(f"排除字段 {col}: 唯一值数量不足 ({unique_count})")
                continue
            
            if field_rules['max_unique_values'] and unique_count > field_rules['max_unique_values']:
                print(f"排除字段 {col}: 唯一值数量过多 ({unique_count})")
                continue
            
            selected_columns.append(col)
            field_stats.append({
                'field': col,
                'missing_rate': missing_rate,
                'unique_count': unique_count,
                'dtype': str(data[col].dtype)
            })
        
        data = data[selected_columns]
        self.train_log['field_preprocessing'] = {
            'original_columns': len(data.columns) + (len(field_stats) - len(selected_columns) + 1),
            'selected_columns': len(selected_columns),
            'field_stats': field_stats
        }
        
        print(f"\n数据预处理完成: 从 {len(data.columns) + (len(field_stats) - len(selected_columns) + 1)} 列中选择了 {len(selected_columns)} 列")
        return data
    
    def load_data(self, data_path, **kwargs):
        path = Path(data_path)
        suffix = path.suffix.lower()
        if suffix == '.pkl':
            return pd.read_pickle(data_path, **kwargs)
        elif suffix == '.csv':
            return pd.read_csv(data_path, **kwargs)
        elif suffix in ['.xlsx', '.xls']:
            return pd.read_excel(data_path, **kwargs)
        else:
            raise ValueError(f"不支持的文件格式: {suffix}")
    
    def split_data_by_date(self, data, date_col, y_name, oot_config=None, test_size=0.2):
        oot_config = oot_config or self.config['scorecard'].get('oot_size', {
            'low_num': 1000, 'high_num': 3000, 'threshold': 7000
        })
        
        df_sorted = data.sort_values(date_col, ascending=False).reset_index(drop=True)
        n = oot_config['low_num'] if len(df_sorted) < oot_config['threshold'] else oot_config['high_num']
        
        oot = df_sorted.head(n).reset_index(drop=True)
        remaining = df_sorted.iloc[n:].reset_index(drop=True)
        
        train, test = train_test_split(
            remaining, test_size=test_size, 
            random_state=self.config['seed'],
            stratify=remaining[y_name]
        )
        train = train.reset_index(drop=True)
        test = test.reset_index(drop=True)
        
        print(f"\n数据集划分 (按日期倒序):")
        print(f"  总数据量: {len(data)}, OOT验证集: {len(oot)}, 训练集: {len(train)}, 测试集: {len(test)}")
        
        self.train_log['data_split'] = {
            'total': len(data), 'oot_size': len(oot),
            'train_size': len(train), 'test_size': len(test),
        }
        
        return train, test, oot
    
    def calculate_ks(self, y_true, y_pred):
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        return max(abs(tpr - fpr))
    
    def evaluate_model(self, y_true, y_pred, prefix=''):
        auc = roc_auc_score(y_true, y_pred)
        ks = self.calculate_ks(y_true, y_pred)
        gini = 2 * auc - 1
        return {f'{prefix}auc': auc, f'{prefix}ks': ks, f'{prefix}gini': gini}

    def build_time_validation_split(self, data, date_col, y_name, valid_ratio=0.2):
        if date_col in data.columns and data[date_col].notna().sum() > 0:
            ordered = data.sort_values(date_col).reset_index(drop=True)
        else:
            ordered = data.sample(frac=1.0, random_state=self.config['seed']).reset_index(drop=True)

        split_idx = max(int(len(ordered) * (1 - valid_ratio)), 1)
        split_idx = min(split_idx, len(ordered) - 1)
        subtrain = ordered.iloc[:split_idx].reset_index(drop=True)
        valid = ordered.iloc[split_idx:].reset_index(drop=True)
        if valid[y_name].nunique() < 2 or subtrain[y_name].nunique() < 2:
            subtrain, valid = train_test_split(
                data,
                test_size=valid_ratio,
                random_state=self.config['seed'],
                stratify=data[y_name],
            )
            subtrain = subtrain.reset_index(drop=True)
            valid = valid.reset_index(drop=True)
        return subtrain, valid

    def build_rolling_validation_splits(self, data, date_col, y_name, n_splits=3, valid_ratio=0.2, min_train_ratio=0.4):
        if len(data) < 200:
            return []

        if date_col in data.columns and data[date_col].notna().sum() > 0:
            ordered = data.sort_values(date_col).reset_index(drop=True)
        else:
            ordered = data.sample(frac=1.0, random_state=self.config['seed']).reset_index(drop=True)

        n = len(ordered)
        valid_size = max(int(n * valid_ratio), 1)
        min_train_size = max(int(n * min_train_ratio), 1)
        candidate_starts = []

        latest_valid_start = n - valid_size
        earliest_valid_start = min_train_size
        if latest_valid_start <= earliest_valid_start:
            return []

        raw_points = np.linspace(earliest_valid_start, latest_valid_start, num=max(n_splits, 1), dtype=int)
        for start in raw_points:
            start = int(start)
            end = min(start + valid_size, n)
            if start < min_train_size or end - start <= 0:
                continue
            train_idx = np.arange(0, start)
            valid_idx = np.arange(start, end)
            train_y = ordered.iloc[train_idx][y_name]
            valid_y = ordered.iloc[valid_idx][y_name]
            if train_y.nunique() < 2 or valid_y.nunique() < 2:
                continue
            candidate_starts.append((train_idx, valid_idx))

        unique_splits = []
        seen = set()
        for train_idx, valid_idx in candidate_starts:
            key = (int(train_idx[-1]), int(valid_idx[0]), int(valid_idx[-1]))
            if key in seen:
                continue
            seen.add(key)
            unique_splits.append((train_idx, valid_idx))
        return unique_splits

    def _make_numeric_edges(self, ref_values, bin_count=10):
        clean = pd.Series(ref_values).replace([np.inf, -np.inf], np.nan).dropna()
        if clean.empty or clean.nunique() <= 1:
            return None
        try:
            quantiles = np.linspace(0, 1, bin_count + 1)
            edges = np.unique(np.quantile(clean, quantiles))
        except Exception:
            return None
        if len(edges) < 2:
            return None
        edges = edges.astype(float)
        edges[0] = -np.inf
        edges[-1] = np.inf
        return edges

    def _calculate_feature_psi(self, ref_series, cmp_series, bin_count=10):
        ref = pd.Series(ref_series)
        cmp_ = pd.Series(cmp_series)
        edges = self._make_numeric_edges(ref, bin_count=bin_count)
        if edges is None:
            return 0.0
        ref_bins = pd.cut(ref, bins=edges, include_lowest=True)
        cmp_bins = pd.cut(cmp_, bins=edges, include_lowest=True)
        ref_dist = ref_bins.value_counts(normalize=True, sort=False)
        cmp_dist = cmp_bins.value_counts(normalize=True, sort=False)
        psi_df = pd.DataFrame({'ref_pct': ref_dist, 'cmp_pct': cmp_dist}).fillna(0)
        psi_df['ref_pct'] = psi_df['ref_pct'].clip(lower=1e-6)
        psi_df['cmp_pct'] = psi_df['cmp_pct'].clip(lower=1e-6)
        psi_df['psi'] = (psi_df['cmp_pct'] - psi_df['ref_pct']) * np.log(psi_df['cmp_pct'] / psi_df['ref_pct'])
        return float(psi_df['psi'].sum())

    def evaluate_feature_stability(self, reference_df, validation_df, feature_list):
        stability_config = self.config['scorecard'].get('stability_selection', {})
        psi_threshold = stability_config.get('psi_threshold', 0.25)
        missing_diff_threshold = stability_config.get('missing_rate_diff_threshold', 0.05)
        min_features = stability_config.get('min_features_to_keep', 120)
        bin_count = stability_config.get('psi_bins', 10)

        records = []
        for feature in feature_list:
            ref_series = pd.to_numeric(reference_df[feature], errors='coerce')
            valid_series = pd.to_numeric(validation_df[feature], errors='coerce')
            ref_missing = float(ref_series.isna().mean())
            valid_missing = float(valid_series.isna().mean())
            missing_diff = abs(valid_missing - ref_missing)
            psi_value = self._calculate_feature_psi(ref_series, valid_series, bin_count=bin_count)
            stability_pass = (missing_diff <= missing_diff_threshold) and (psi_value <= psi_threshold)
            records.append({
                'var': feature,
                'dev_missing_rate': ref_missing,
                'validation_missing_rate': valid_missing,
                'missing_rate_diff': missing_diff,
                'feature_psi': psi_value,
                'stability_pass': stability_pass,
            })

        stability_df = pd.DataFrame(records)
        if not stability_df.empty:
            stability_df = stability_df.sort_values(
                ['stability_pass', 'feature_psi', 'missing_rate_diff', 'dev_missing_rate', 'validation_missing_rate'],
                ascending=[False, True, True, True, True],
            ).reset_index(drop=True)
            stability_df['stability_rank'] = np.nan
            passed_mask = stability_df['stability_pass']
            stability_df.loc[passed_mask, 'stability_rank'] = range(1, int(passed_mask.sum()) + 1)
        stable_features = stability_df.loc[stability_df['stability_pass'], 'var'].tolist()
        if len(stable_features) < min_features:
            stability_df['stability_pass'] = True
            stability_df = stability_df.sort_values(
                ['feature_psi', 'missing_rate_diff', 'dev_missing_rate', 'validation_missing_rate'],
                ascending=[True, True, True, True],
            ).reset_index(drop=True)
            stability_df['stability_rank'] = range(1, len(stability_df) + 1)
            stable_features = feature_list.copy()
            stability_status = 'fallback_all_kept'
        else:
            stability_status = 'filtered'

        self.train_log['feature_stability'] = stability_df
        self.train_log['feature_stability_summary'] = {
            'candidate_count': len(feature_list),
            'stable_count': len(stable_features),
            'psi_threshold': psi_threshold,
            'missing_rate_diff_threshold': missing_diff_threshold,
            'status': stability_status,
        }
        return stable_features

    def _compute_tree_stability_score(self, y_train, train_pred, y_valid, valid_pred, target='ks'):
        valid_score = self.calculate_ks(y_valid, valid_pred) if target == 'ks' else roc_auc_score(y_valid, valid_pred)
        valid_auc = roc_auc_score(y_valid, valid_pred)
        valid_ks = self.calculate_ks(y_valid, valid_pred)
        train_auc = roc_auc_score(y_train, train_pred)
        train_ks = self.calculate_ks(y_train, train_pred)
        overfit_penalty = max(train_auc - valid_auc, 0) * 0.20 + max(train_ks - valid_ks, 0) * 0.30
        stability_score = valid_score - overfit_penalty
        return {
            'valid_score': float(valid_score),
            'valid_auc': float(valid_auc),
            'valid_ks': float(valid_ks),
            'train_auc': float(train_auc),
            'train_ks': float(train_ks),
            'stability_score': float(stability_score),
            'overfit_penalty': float(overfit_penalty),
        }

    def _evaluate_tree_candidate(self, model_builder, params, X_train, y_train, cv_splits, target='ks', repeats=1):
        if not cv_splits:
            return None

        stability_scores = []
        valid_scores = []
        valid_auc_list = []
        valid_ks_list = []
        train_auc_list = []
        train_ks_list = []

        for repeat_idx in range(repeats):
            repeat_params = params.copy()
            if 'random_state' in repeat_params:
                repeat_params['random_state'] = int(self.config['seed'] + repeat_idx)

            for train_idx, valid_idx in cv_splits:
                fold_X_train = X_train.iloc[train_idx]
                fold_y_train = y_train.iloc[train_idx]
                fold_X_valid = X_train.iloc[valid_idx]
                fold_y_valid = y_train.iloc[valid_idx]
                model = model_builder(repeat_params)
                model.fit(fold_X_train, fold_y_train)
                train_pred = model.predict_proba(fold_X_train)[:, 1]
                valid_pred = model.predict_proba(fold_X_valid)[:, 1]
                metrics = self._compute_tree_stability_score(
                    fold_y_train, train_pred, fold_y_valid, valid_pred, target=target
                )
                stability_scores.append(metrics['stability_score'])
                valid_scores.append(metrics['valid_score'])
                valid_auc_list.append(metrics['valid_auc'])
                valid_ks_list.append(metrics['valid_ks'])
                train_auc_list.append(metrics['train_auc'])
                train_ks_list.append(metrics['train_ks'])

        if not stability_scores:
            return None

        return {
            'mean_stability_score': float(np.mean(stability_scores)),
            'std_valid_score': float(np.std(valid_scores)),
            'mean_valid_score': float(np.mean(valid_scores)),
            'mean_valid_auc': float(np.mean(valid_auc_list)),
            'mean_valid_ks': float(np.mean(valid_ks_list)),
            'mean_train_auc': float(np.mean(train_auc_list)),
            'mean_train_ks': float(np.mean(train_ks_list)),
            'selection_score': float(np.mean(stability_scores) - np.std(valid_scores) * 0.15),
        }

    def _select_stable_best_params(
        self, study, model_name, model_builder, X_train, y_train, cv_splits, target='ks', top_k=5, repeats=3
    ):
        completed_trials = [trial for trial in study.trials if trial.value is not None and trial.params]
        completed_trials.sort(key=lambda t: t.value, reverse=True)
        candidate_trials = completed_trials[:top_k]
        if not candidate_trials:
            return study.best_params

        candidate_rows = []
        best_params = None
        best_selection_score = -999999.0

        print(f"Evaluating top {len(candidate_trials)} {model_name} candidates with repeated validation...")
        for rank, trial in enumerate(candidate_trials, start=1):
            evaluation = self._evaluate_tree_candidate(
                model_builder,
                trial.params,
                X_train,
                y_train,
                cv_splits,
                target=target,
                repeats=repeats,
            )
            if evaluation is None:
                continue
            row = {
                'candidate_rank': rank,
                'trial_number': int(trial.number),
                'optuna_score': float(trial.value),
                'selection_score': evaluation['selection_score'],
                'mean_valid_score': evaluation['mean_valid_score'],
                'std_valid_score': evaluation['std_valid_score'],
                'mean_valid_auc': evaluation['mean_valid_auc'],
                'mean_valid_ks': evaluation['mean_valid_ks'],
                'mean_train_auc': evaluation['mean_train_auc'],
                'mean_train_ks': evaluation['mean_train_ks'],
                'params': json.dumps(trial.params, ensure_ascii=False),
            }
            candidate_rows.append(row)
            if evaluation['selection_score'] > best_selection_score:
                best_selection_score = evaluation['selection_score']
                best_params = trial.params.copy()

        if candidate_rows:
            self.train_log['tuning_candidates'] = pd.DataFrame(candidate_rows)
        return best_params or study.best_params

    def tune_logistic(self, X_train, y_train, X_test, y_test, n_trials=30, cv_splits=None):
        """Use Optuna to tune LogisticRegression on WOE features with stability-aware selection."""
        try:
            import optuna
        except ImportError:
            print("optuna not installed, fallback to default params")
            return None

        target = self.config['scorecard'].get('tuning', {}).get('target', 'ks')

        def build_model(params):
            fit_params = params.copy()
            fit_params.setdefault('random_state', self.config['seed'])
            fit_params.setdefault('max_iter', 2000)
            fit_params.setdefault('solver', 'liblinear')
            return LogisticRegression(**fit_params)

        def objective(trial):
            params = {
                'C': trial.suggest_float('C', 0.03, 12.0, log=True),
                'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
                'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced']),
                'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
                'solver': 'liblinear',
                'max_iter': 2000,
                'random_state': self.config['seed'],
            }

            model = build_model(params)
            if cv_splits:
                fold_scores = []
                fold_stability = []
                for train_idx, valid_idx in cv_splits:
                    fold_X_train = X_train.iloc[train_idx]
                    fold_y_train = y_train.iloc[train_idx]
                    fold_X_valid = X_train.iloc[valid_idx]
                    fold_y_valid = y_train.iloc[valid_idx]
                    model.fit(fold_X_train, fold_y_train)
                    valid_pred = model.predict_proba(fold_X_valid)[:, 1]
                    train_pred = model.predict_proba(fold_X_train)[:, 1]
                    metrics = self._compute_tree_stability_score(
                        fold_y_train, train_pred, fold_y_valid, valid_pred, target=target
                    )
                    fold_scores.append(metrics['stability_score'])
                    fold_stability.append(metrics['valid_score'])
                if not fold_scores:
                    return -999.0
                return float(np.mean(fold_scores) - np.std(fold_stability) * 0.15)

            model.fit(X_train, y_train)
            y_pred = model.predict_proba(X_test)[:, 1]
            return self.calculate_ks(y_test, y_pred) if target == 'ks' else roc_auc_score(y_test, y_pred)

        sampler = optuna.samplers.TPESampler(seed=self.config['seed'])
        study = optuna.create_study(direction='maximize', sampler=sampler)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best_params = self._select_stable_best_params(
            study,
            'LogisticRegression',
            build_model,
            X_train,
            y_train,
            cv_splits,
            target=target,
            top_k=self.config['scorecard'].get('tuning', {}).get('top_candidate_count', 5),
            repeats=self.config['scorecard'].get('tuning', {}).get('candidate_repeats', 3),
        ) if cv_splits else study.best_params

        print(f"Best Logistic params: {best_params}")
        print(f"Best score: {study.best_value:.4f}")

        self.train_log['tuning'] = {'best_params': best_params, 'best_score': study.best_value}
        return best_params
    
    def tune_lightgbm(self, X_train, y_train, X_test, y_test, n_trials=30, cv_splits=None):
        """Use Optuna to tune LightGBM with stronger regularization and stability penalty."""
        try:
            import optuna
        except ImportError:
            print("optuna not installed, fallback to default params")
            return None

        pos_count = max(int((y_train == 1).sum()), 1)
        neg_count = max(int((y_train == 0).sum()), 1)
        base_spw = neg_count / pos_count
        target = self.config['scorecard'].get('tuning', {}).get('target', 'ks')

        def build_model(params):
            fit_params = params.copy()
            fit_params.setdefault('random_state', self.config['seed'])
            fit_params.setdefault('verbose', -1)
            return lgb.LGBMClassifier(**fit_params)

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 180, 520),
                'learning_rate': trial.suggest_float('learning_rate', 0.015, 0.06, log=True),
                'max_depth': trial.suggest_int('max_depth', 2, 5),
                'num_leaves': trial.suggest_int('num_leaves', 8, 28),
                'min_child_samples': trial.suggest_int('min_child_samples', 180, 420),
                'subsample': trial.suggest_float('subsample', 0.6, 0.88),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.85),
                'reg_alpha': trial.suggest_float('reg_alpha', 3, 120, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 5, 120, log=True),
                'min_split_gain': trial.suggest_float('min_split_gain', 0.3, 2.0),
                'min_child_weight': trial.suggest_float('min_child_weight', 1.0, 30, log=True),
                'path_smooth': trial.suggest_float('path_smooth', 5, 35),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', max(1.0, base_spw * 0.7), base_spw * 1.3),
                'random_state': self.config['seed'],
                'verbose': -1,
            }

            model = build_model(params)
            if cv_splits:
                fold_scores = []
                fold_stability = []
                for train_idx, valid_idx in cv_splits:
                    fold_X_train = X_train.iloc[train_idx]
                    fold_y_train = y_train.iloc[train_idx]
                    fold_X_valid = X_train.iloc[valid_idx]
                    fold_y_valid = y_train.iloc[valid_idx]
                    model.fit(
                        fold_X_train,
                        fold_y_train,
                        eval_set=[(fold_X_valid, fold_y_valid)],
                        eval_metric='auc',
                        callbacks=[lgb.early_stopping(50, verbose=False)],
                    )
                    valid_pred = model.predict_proba(fold_X_valid)[:, 1]
                    train_pred = model.predict_proba(fold_X_train)[:, 1]
                    metrics = self._compute_tree_stability_score(
                        fold_y_train, train_pred, fold_y_valid, valid_pred, target=target
                    )
                    fold_scores.append(metrics['stability_score'])
                    fold_stability.append(metrics['valid_score'])
                if not fold_scores:
                    return -999.0
                return float(np.mean(fold_scores) - np.std(fold_stability) * 0.15)

            model.fit(
                X_train,
                y_train,
                eval_set=[(X_test, y_test)],
                eval_metric='auc',
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
            y_pred = model.predict_proba(X_test)[:, 1]
            return self.calculate_ks(y_test, y_pred) if target == 'ks' else roc_auc_score(y_test, y_pred)

        sampler = optuna.samplers.TPESampler(seed=self.config['seed'])
        study = optuna.create_study(direction='maximize', sampler=sampler)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best_params = self._select_stable_best_params(
            study,
            'LightGBM',
            build_model,
            X_train,
            y_train,
            cv_splits,
            target=target,
            top_k=self.config['scorecard'].get('tuning', {}).get('top_candidate_count', 5),
            repeats=self.config['scorecard'].get('tuning', {}).get('candidate_repeats', 3),
        ) if cv_splits else study.best_params

        print(f"Best LightGBM params: {best_params}")
        print(f"Best score: {study.best_value:.4f}")

        self.train_log['tuning'] = {'best_params': best_params, 'best_score': study.best_value}
        return best_params

    def tune_xgboost(self, X_train, y_train, X_test, y_test, n_trials=30, cv_splits=None):
        """Use Optuna to tune XGBoost with stronger regularization and imbalance handling."""
        try:
            import optuna
        except ImportError:
            print("optuna not installed, fallback to default params")
            return None

        pos_count = max(int((y_train == 1).sum()), 1)
        neg_count = max(int((y_train == 0).sum()), 1)
        base_spw = neg_count / pos_count
        target = self.config['scorecard'].get('tuning', {}).get('target', 'ks')

        def build_model(params):
            fit_params = params.copy()
            fit_params.setdefault('random_state', self.config['seed'])
            fit_params.setdefault('use_label_encoder', False)
            fit_params.setdefault('eval_metric', 'logloss')
            fit_params.setdefault('n_jobs', 1)
            return xgb.XGBClassifier(**fit_params)

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 200, 800),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.08, log=True),
                'max_depth': trial.suggest_int('max_depth', 2, 5),
                'min_child_weight': trial.suggest_int('min_child_weight', 8, 80),
                'subsample': trial.suggest_float('subsample', 0.6, 0.85),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.85),
                'gamma': trial.suggest_float('gamma', 1.0, 10.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 2, 120, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 5, 120, log=True),
                'min_split_gain': trial.suggest_float('min_split_gain', 0.2, 2.0),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', max(1.0, base_spw * 0.7), base_spw * 1.3),
                'max_delta_step': trial.suggest_int('max_delta_step', 1, 10),
                'random_state': self.config['seed'],
                'use_label_encoder': False,
                'eval_metric': 'logloss',
                'n_jobs': 1,
            }

            model = build_model(params)
            if cv_splits:
                fold_scores = []
                fold_stability = []
                for train_idx, valid_idx in cv_splits:
                    fold_X_train = X_train.iloc[train_idx]
                    fold_y_train = y_train.iloc[train_idx]
                    fold_X_valid = X_train.iloc[valid_idx]
                    fold_y_valid = y_train.iloc[valid_idx]
                    model.fit(fold_X_train, fold_y_train, eval_set=[(fold_X_valid, fold_y_valid)], verbose=False)
                    valid_pred = model.predict_proba(fold_X_valid)[:, 1]
                    train_pred = model.predict_proba(fold_X_train)[:, 1]
                    metrics = self._compute_tree_stability_score(
                        fold_y_train, train_pred, fold_y_valid, valid_pred, target=target
                    )
                    fold_scores.append(metrics['stability_score'])
                    fold_stability.append(metrics['valid_score'])
                if not fold_scores:
                    return -999.0
                return float(np.mean(fold_scores) - np.std(fold_stability) * 0.15)

            model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
            y_pred = model.predict_proba(X_test)[:, 1]
            return self.calculate_ks(y_test, y_pred) if target == 'ks' else roc_auc_score(y_test, y_pred)

        sampler = optuna.samplers.TPESampler(seed=self.config['seed'])
        study = optuna.create_study(direction='maximize', sampler=sampler)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best_params = self._select_stable_best_params(
            study,
            'XGBoost',
            build_model,
            X_train,
            y_train,
            cv_splits,
            target=target,
            top_k=self.config['scorecard'].get('tuning', {}).get('top_candidate_count', 5),
            repeats=self.config['scorecard'].get('tuning', {}).get('candidate_repeats', 3),
        ) if cv_splits else study.best_params

        print(f"Best XGBoost params: {best_params}")
        print(f"Best score: {study.best_value:.4f}")

        self.train_log['tuning'] = {'best_params': best_params, 'best_score': study.best_value}
        return best_params

    def plot_roc_curve(self, y_true_dict, y_pred_dict, save_path):
        return export_plot_roc_curve(y_true_dict, y_pred_dict, save_path)

    def plot_feature_importance(self, var_list, importance_values, save_path):
        return export_plot_feature_importance(var_list, importance_values, save_path)

    def calc_vif(self, data, var_list):
        """计算VIF方差膨胀因子"""
        tmp = data[var_list].dropna()
        if len(tmp) < 100:
            return pd.DataFrame({'VAR': var_list, 'VIF': [np.nan] * len(var_list)})
        
        cols_map = {v: f'tmpcol{i}' for i, v in enumerate(var_list)}
        tmp = tmp.rename(columns=cols_map)
        cnames = list(cols_map.values())
        
        vif_list = []
        for i in range(len(cnames)):
            xvars = cnames[:i] + cnames[i+1:]
            yvar = cnames[i]
            try:
                X = sm.add_constant(tmp[xvars])
                y = tmp[yvar]
                model = sm.OLS(y, X).fit()
                vif = 1 / (1 - model.rsquared) if model.rsquared < 1 else 100
            except:
                vif = 100
            vif_list.append({'VAR': var_list[i], 'VIF': vif})
        
        return pd.DataFrame(vif_list)
    
    def remove_high_vif(self, data, var_list, threshold=5.0):
        """逐步移除高VIF变量"""
        current_vars = var_list.copy()
        
        while len(current_vars) > 1:
            vif_df = self.calc_vif(data, current_vars)
            max_vif = vif_df['VIF'].max()
            
            if max_vif <= threshold:
                break
            
            max_var = vif_df.loc[vif_df['VIF'].idxmax(), 'VAR']
            print(f"  移除高VIF变量: {max_var} (VIF={max_vif:.2f})")
            current_vars.remove(max_var)
        
        return current_vars
    
    def remove_high_corr(self, data, var_list, threshold=0.7, iv_rank=None):
        """移除高相关性变量 - 逐个移除，每次重新计算相关性，优先保留IV高的变量"""
        current_vars = var_list.copy()
        removed_vars = []
        
        while True:
            if len(current_vars) <= 1:
                break
            
            corr_matrix = data[current_vars].corr().abs()
            
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            
            max_corr = 0
            max_pair = None
            for col in upper.columns:
                for idx in upper.index:
                    val = upper.loc[idx, col]
                    if pd.notna(val) and val > max_corr:
                        max_corr = val
                        max_pair = (idx, col)
            
            if max_corr <= threshold:
                break
            
            if max_pair:
                var1, var2 = max_pair
                
                if iv_rank is not None:
                    rank1 = iv_rank.get(var1, 999)
                    rank2 = iv_rank.get(var2, 999)
                    var_to_remove = var2 if rank2 >= rank1 else var1
                    var_kept = var1 if var_to_remove == var2 else var2
                else:
                    var_to_remove = var2
                    var_kept = var1
                
                print(f"  移除高相关性变量: {var_to_remove} (与{var_kept}相关性={max_corr:.3f})")
                current_vars.remove(var_to_remove)
                removed_vars.append(var_to_remove)
        
        if removed_vars:
            print(f"  共移除{len(removed_vars)}个高相关性变量")
        
        return current_vars
    
    def feature_selection_logistic(self, data, y_name):
        """逻辑回归特征筛选：IV + VIF + 相关性，支持自动调整"""
        config = self.config['scorecard']
        fs_config = config.get('feature_selection', {})
        lr_config = config.get('logistic', {})
        
        exclude_vars = config.get('exclude_vars', [])
        id_var = config.get('id_var', '')
        date_col = config.get('date_col', '')
        
        feature_cols = [c for c in data.columns if c not in [y_name, id_var, date_col] + exclude_vars]
        
        numeric_features = []
        for col in feature_cols:
            if data[col].dtype == 'object':
                continue
            try:
                data[col].astype(float)
                numeric_features.append(col)
            except:
                continue
        
        print(f"数值型特征: {len(numeric_features)}个")
        
        prefilter_cfg = lr_config.get('tree_prefilter', {})
        if prefilter_cfg.get('enable', True) and len(numeric_features) > prefilter_cfg.get('max_candidates', 200):
            prefilter_model_type = prefilter_cfg.get('model_type', 'xgboost')
            if prefilter_model_type == 'lightgbm':
                prefilter_model = lgb.LGBMClassifier(
                    n_estimators=prefilter_cfg.get('n_estimators', 200),
                    learning_rate=prefilter_cfg.get('learning_rate', 0.05),
                    max_depth=prefilter_cfg.get('max_depth', 4),
                    num_leaves=prefilter_cfg.get('num_leaves', 24),
                    min_child_samples=prefilter_cfg.get('min_child_samples', 40),
                    subsample=prefilter_cfg.get('subsample', 0.8),
                    colsample_bytree=prefilter_cfg.get('colsample_bytree', 0.8),
                    class_weight='balanced',
                    random_state=self.config['seed'],
                    verbose=-1,
                )
            elif prefilter_model_type == 'random_forest':
                prefilter_model = RandomForestClassifier(
                    n_estimators=prefilter_cfg.get('n_estimators', 300),
                    max_depth=prefilter_cfg.get('max_depth', 5),
                    min_samples_leaf=prefilter_cfg.get('min_samples_leaf', 30),
                    class_weight='balanced_subsample',
                    random_state=self.config['seed'],
                    n_jobs=1,
                )
            else:
                prefilter_model = xgb.XGBClassifier(
                    n_estimators=prefilter_cfg.get('n_estimators', 180),
                    learning_rate=prefilter_cfg.get('learning_rate', 0.05),
                    max_depth=prefilter_cfg.get('max_depth', 3),
                    min_child_weight=prefilter_cfg.get('min_child_weight', 20),
                    subsample=prefilter_cfg.get('subsample', 0.8),
                    colsample_bytree=prefilter_cfg.get('colsample_bytree', 0.8),
                    reg_alpha=prefilter_cfg.get('reg_alpha', 2.0),
                    reg_lambda=prefilter_cfg.get('reg_lambda', 10.0),
                    scale_pos_weight=max(float((data[y_name] == 0).sum()) / max(float((data[y_name] == 1).sum()), 1.0), 1.0),
                    random_state=self.config['seed'],
                    use_label_encoder=False,
                    eval_metric='logloss',
                    n_jobs=1,
                )
            prefilter_X = data[numeric_features].apply(pd.to_numeric, errors='coerce').fillna(-9999)
            prefilter_y = data[y_name]
            print(f"\n逻辑回归预筛选: 使用{prefilter_model_type}缩小候选范围...")
            prefilter_model.fit(prefilter_X, prefilter_y)
            prefilter_importance = pd.DataFrame({
                'var': numeric_features,
                'importance': prefilter_model.feature_importances_,
            }).sort_values('importance', ascending=False)
            prefilter_importance = prefilter_importance[prefilter_importance['importance'] > 0].reset_index(drop=True)
            prefilter_importance = self._apply_logistic_group_cap(prefilter_importance)
            if len(prefilter_importance) >= prefilter_cfg.get('min_keep', 120):
                numeric_features = prefilter_importance.head(prefilter_cfg.get('max_candidates', 200))['var'].tolist()
            else:
                numeric_features = prefilter_importance['var'].tolist()
            self.train_log['logistic_tree_prefilter'] = prefilter_importance.copy()
            print(f"  树模型正重要性候选: {len(prefilter_importance)} 个, 进入IV筛选: {len(numeric_features)} 个")

        self.woe_binner = WOEBinning(
            min_bins=lr_config.get('woe_min_bins', 2),
            max_bins=lr_config.get('woe_bins', 7),
            bin_pct=lr_config.get('woe_bin_pct', 0.05),
            raw_bin_multiplier=lr_config.get('woe_raw_bin_multiplier', 3),
            prefer_monotonic=lr_config.get('prefer_monotonic', True),
        )
        
        feature_iv = []
        for col in numeric_features:
            bin_df = self.woe_binner.fit(data, col, y_name)
            if bin_df is not None:
                iv = self.woe_binner.get_iv(col)
                missing_rate = data[col].isnull().mean()
                try:
                    valid_data = data[[col, y_name]].dropna()
                    ks = abs(ks_2samp(
                        valid_data.loc[valid_data[y_name] == 0, col],
                        valid_data.loc[valid_data[y_name] == 1, col]
                    ).statistic)
                except:
                    ks = 0
                feature_iv.append({'var': col, 'iv': iv, 'ks': ks, 'missing_rate': missing_rate})
        
        feature_df = pd.DataFrame(feature_iv)
        feature_df = feature_df[feature_df['missing_rate'] <= fs_config.get('missing_threshold', 0.95)]
        feature_df = feature_df[feature_df['iv'] >= lr_config.get('iv_threshold', 0.02)]
        feature_df = feature_df.sort_values('iv', ascending=False)
        feature_df = self._apply_logistic_group_cap(feature_df)
        
        min_final_features = lr_config.get('min_features', fs_config.get('min_features', 10))
        max_features = lr_config.get('max_features', fs_config.get('max_features', 25))
        max_iterations = 10
        
        selected_features = []
        current_corr_threshold = lr_config.get('corr_threshold', 0.7)
        threshold_adjusted = False
        
        for iteration in range(1, max_iterations + 1):
            current_max_features = max_features + (iteration - 1) * 5
            
            if iteration > 1:
                print(f"\n  特征数量不足{min_final_features}个，重新筛选 (max_features={current_max_features})...")
            
            temp_feature_df = feature_df.head(current_max_features)
            temp_selected = temp_feature_df['var'].tolist()
            print(f"\n特征筛选 (IV方法): IV筛选后 {len(temp_selected)}个")
            
            if len(temp_selected) > 1:
                iv_rank = {row['var']: idx for idx, row in temp_feature_df.iterrows()}
                temp_selected = self.remove_high_corr(
                    data, temp_selected, 
                    threshold=current_corr_threshold,
                    iv_rank=iv_rank
                )
                print(f"  相关性筛选后: {len(temp_selected)}个 (阈值={current_corr_threshold:.1f})")
            
            if len(temp_selected) > 1:
                temp_selected = self.remove_high_vif(
                    data, temp_selected, 
                    threshold=lr_config.get('vif_threshold', 5.0)
                )
                print(f"  VIF筛选后: {len(temp_selected)}个")
            
            selected_features = temp_selected
            
            if len(selected_features) >= min_final_features:
                break
            
            if len(feature_df) <= current_max_features:
                if current_corr_threshold < 0.95:
                    current_corr_threshold = min(current_corr_threshold + 0.05, 0.95)
                    print(f"\n  放宽相关性阈值至 {current_corr_threshold:.2f} 重新筛选...")
                    iteration = 0
                    continue
                break
        
        if len(selected_features) < min_final_features:
            print(f"\n  警告: 最终特征数量{len(selected_features)}个，已达到最大筛选范围")
        
        self.train_log['feature_iv'] = feature_df
        return selected_features
    
    def feature_selection_tree(self, data, y_name, model_type='lightgbm', reference_df=None, validation_df=None):
        """Select tree-model features by feature importance."""
        config = self.config['scorecard']
        fs_config = config.get('feature_selection', {})
        exclude_vars = config.get('exclude_vars', [])
        id_var = config.get('id_var', '')
        date_col = config.get('date_col', '')

        feature_cols = [c for c in data.columns if c not in [y_name, id_var, date_col] + exclude_vars]

        numeric_features = []
        for col in feature_cols:
            if data[col].dtype == 'object':
                continue
            try:
                data[col].astype(float)
                numeric_features.append(col)
            except Exception:
                continue

        print(f"Numeric features: {len(numeric_features)}")
        self.train_log['tree_feature_screening'] = {
            'raw_column_count': len(data.columns),
            'candidate_feature_count': len(numeric_features),
            'model_type': model_type,
            'max_tree_features': fs_config.get('max_tree_features', 50),
        }
        X = data[numeric_features].fillna(-9999)
        y = data[y_name]
        print("\nTraining a preliminary model to get feature importance...")

        if model_type == 'lightgbm':
            try:
                prelim_model = lgb.LGBMClassifier(
                    n_estimators=100, learning_rate=0.1, max_depth=5,
                    random_state=self.config['seed'], verbose=-1
                )
            except NameError:
                print("Error: lightgbm is not installed")
                return numeric_features
        elif model_type == 'xgboost':
            try:
                prelim_model = xgb.XGBClassifier(
                    n_estimators=100, learning_rate=0.1, max_depth=5,
                    random_state=self.config['seed'], use_label_encoder=False, eval_metric='logloss', n_jobs=1
                )
            except NameError:
                print("Error: xgboost is not installed")
                return numeric_features
        else:
            return numeric_features

        prelim_model.fit(X, y)
        importance_values = prelim_model.feature_importances_
        feature_importance = pd.DataFrame({
            'var': numeric_features,
            'importance': importance_values
        }).sort_values('importance', ascending=False)
        positive_feature_importance = feature_importance[feature_importance['importance'] > 0].reset_index(drop=True)
        positive_feature_importance['candidate_rank'] = range(1, len(positive_feature_importance) + 1)
        max_tree_features = fs_config.get('max_tree_features', 50)
        stability_source = positive_feature_importance.copy()

        if (
            self.config['scorecard'].get('stability_selection', {}).get('enable', True)
            and reference_df is not None
            and validation_df is not None
            and not positive_feature_importance.empty
        ):
            stable_features = self.evaluate_feature_stability(
                reference_df,
                validation_df,
                positive_feature_importance['var'].tolist(),
            )
            positive_feature_importance = positive_feature_importance[
                positive_feature_importance['var'].isin(stable_features)
            ].reset_index(drop=True)
            positive_feature_importance['candidate_rank'] = range(1, len(positive_feature_importance) + 1)
            print("\nFeature stability screening:")
            print(f"  Positive-importance features: {len(stability_source)}, stable features kept: {len(positive_feature_importance)}")

        self.train_log['tree_feature_candidates'] = stability_source.copy()
        self.train_log['tree_feature_candidates_after_stability'] = positive_feature_importance.copy()

        # Final tree-model features: importance > 0, pass stability screening,
        # then keep the top-N by feature importance.
        if len(positive_feature_importance) > max_tree_features:
            feature_importance = self._apply_group_feature_cap(positive_feature_importance, max_tree_features).reset_index(drop=True)
            print("\nFeature selection (importance-based):")
            print(f"  Importance>0 features: {len(self.train_log['tree_feature_candidates'])}, selected top {len(feature_importance)} (cap={max_tree_features})")
        else:
            feature_importance = positive_feature_importance.copy()
            print("\nFeature selection (importance-based):")
            print(f"  Importance>0 features: {len(feature_importance)}")

        interaction_plan = self._build_interaction_plan(feature_importance)
        if interaction_plan:
            print(f"  Added interaction features to plan: {len(interaction_plan)}")
        selected_features = feature_importance['var'].tolist()
        self.train_log['feature_importance'] = feature_importance
        return selected_features

    def run_tree_time_window_validation(self, model_type, base_params, X_all, y_all, cv_splits, repeats=3):
        if not cv_splits or not base_params:
            return None, None

        results = []
        for repeat_id in range(1, repeats + 1):
            for window_id, (train_idx, valid_idx) in enumerate(cv_splits, start=1):
                params = base_params.copy()
                params['random_state'] = int(self.config['seed'] + repeat_id - 1)
                if model_type == 'lightgbm':
                    params['verbose'] = -1
                    model = lgb.LGBMClassifier(**params)
                elif model_type == 'logistic':
                    params['solver'] = params.get('solver', 'liblinear')
                    params['max_iter'] = max(int(params.get('max_iter', 2000)), 2000)
                    model = LogisticRegression(**params)
                else:
                    params['use_label_encoder'] = False
                    params['eval_metric'] = 'logloss'
                    params['n_jobs'] = 1
                    model = xgb.XGBClassifier(**params)
                fold_X_train = X_all.iloc[train_idx]
                fold_y_train = y_all.iloc[train_idx]
                fold_X_valid = X_all.iloc[valid_idx]
                fold_y_valid = y_all.iloc[valid_idx]
                model.fit(fold_X_train, fold_y_train)
                train_pred = model.predict_proba(fold_X_train)[:, 1]
                valid_pred = model.predict_proba(fold_X_valid)[:, 1]
                metrics = self._compute_tree_stability_score(
                    fold_y_train, train_pred, fold_y_valid, valid_pred, target='ks'
                )
                results.append({
                    'repeat_id': repeat_id,
                    'window_id': window_id,
                    'train_sample_count': len(train_idx),
                    'validation_sample_count': len(valid_idx),
                    'train_auc': metrics['train_auc'],
                    'train_ks': metrics['train_ks'],
                    'validation_auc': metrics['valid_auc'],
                    'validation_ks': metrics['valid_ks'],
                    'validation_score': metrics['valid_score'],
                    'overfit_penalty': metrics['overfit_penalty'],
                    'stability_score': metrics['stability_score'],
                })

        if not results:
            return None, None

        detail_df = pd.DataFrame(results)
        summary_df = pd.DataFrame({
            'metric_name': ['validation_auc_mean', 'validation_auc_std', 'validation_ks_mean', 'validation_ks_std', 'stability_score_mean', 'stability_score_std'],
            'metric_value': [
                detail_df['validation_auc'].mean(),
                detail_df['validation_auc'].std(ddof=0),
                detail_df['validation_ks'].mean(),
                detail_df['validation_ks'].std(ddof=0),
                detail_df['stability_score'].mean(),
                detail_df['stability_score'].std(ddof=0),
            ],
        })
        self.train_log['time_window_validation_detail'] = detail_df
        self.train_log['time_window_validation_summary'] = summary_df
        return detail_df, summary_df

    def run_ensemble_time_window_validation(
        self, xgb_params, lgb_params, X_all, y_all, cv_splits, repeats=3, weights=None
    ):
        if not cv_splits or not xgb_params or not lgb_params:
            return None, None

        weights = weights or {'xgboost': 0.5, 'lightgbm': 0.5}
        weight_xgb = float(weights.get('xgboost', 0.5))
        weight_lgb = float(weights.get('lightgbm', 0.5))
        total_weight = max(weight_xgb + weight_lgb, 1e-6)
        weight_xgb /= total_weight
        weight_lgb /= total_weight

        results = []
        for repeat_id in range(1, repeats + 1):
            for window_id, (train_idx, valid_idx) in enumerate(cv_splits, start=1):
                xgb_fold_params = xgb_params.copy()
                lgb_fold_params = lgb_params.copy()
                xgb_fold_params['random_state'] = int(self.config['seed'] + repeat_id - 1)
                xgb_fold_params['use_label_encoder'] = False
                xgb_fold_params['eval_metric'] = 'logloss'
                xgb_fold_params['n_jobs'] = 1
                lgb_fold_params['random_state'] = int(self.config['seed'] + repeat_id - 1)
                lgb_fold_params['verbose'] = -1

                xgb_model = xgb.XGBClassifier(**xgb_fold_params)
                lgb_model = lgb.LGBMClassifier(**lgb_fold_params)

                fold_X_train = X_all.iloc[train_idx]
                fold_y_train = y_all.iloc[train_idx]
                fold_X_valid = X_all.iloc[valid_idx]
                fold_y_valid = y_all.iloc[valid_idx]

                xgb_model.fit(fold_X_train, fold_y_train, verbose=False)
                lgb_model.fit(fold_X_train, fold_y_train)

                train_pred = (
                    xgb_model.predict_proba(fold_X_train)[:, 1] * weight_xgb
                    + lgb_model.predict_proba(fold_X_train)[:, 1] * weight_lgb
                )
                valid_pred = (
                    xgb_model.predict_proba(fold_X_valid)[:, 1] * weight_xgb
                    + lgb_model.predict_proba(fold_X_valid)[:, 1] * weight_lgb
                )

                metrics = self._compute_tree_stability_score(
                    fold_y_train, train_pred, fold_y_valid, valid_pred, target='ks'
                )
                results.append({
                    'repeat_id': repeat_id,
                    'window_id': window_id,
                    'train_sample_count': len(train_idx),
                    'validation_sample_count': len(valid_idx),
                    'train_auc': metrics['train_auc'],
                    'train_ks': metrics['train_ks'],
                    'validation_auc': metrics['valid_auc'],
                    'validation_ks': metrics['valid_ks'],
                    'validation_score': metrics['valid_score'],
                    'overfit_penalty': metrics['overfit_penalty'],
                    'stability_score': metrics['stability_score'],
                })

        if not results:
            return None, None

        detail_df = pd.DataFrame(results)
        summary_df = pd.DataFrame({
            'metric_name': [
                'time_window_count',
                'repeat_count',
                'validation_auc_mean',
                'validation_auc_std',
                'validation_ks_mean',
                'validation_ks_std',
                'stability_score_mean',
                'overfit_penalty_mean',
            ],
            'metric_value': [
                detail_df['window_id'].nunique(),
                detail_df['repeat_id'].nunique(),
                detail_df['validation_auc'].mean(),
                detail_df['validation_auc'].std(ddof=0),
                detail_df['validation_ks'].mean(),
                detail_df['validation_ks'].std(ddof=0),
                detail_df['stability_score'].mean(),
                detail_df['overfit_penalty'].mean(),
            ],
        })
        self.train_log['time_window_validation_detail'] = detail_df
        self.train_log['time_window_validation_summary'] = summary_df
        return detail_df, summary_df

    def _feature_group_name(self, feature_name):
        if "_" in feature_name:
            parts = feature_name.split("_")
            return "_".join(parts[:2]) if len(parts) >= 2 else parts[0]
        alpha_prefix = "".join(ch for ch in feature_name if ch.isalpha())
        return alpha_prefix or "OTHER"

    def _apply_group_feature_cap(self, feature_importance_df, max_tree_features):
        group_config = self.config['scorecard'].get('group_feature_limit', {})
        if feature_importance_df.empty:
            return feature_importance_df
        if not group_config.get('enable', False):
            return feature_importance_df.head(max_tree_features).copy()

        max_per_group = int(group_config.get('max_per_group', max_tree_features))
        capped_rows = []
        group_counts = {}
        for _, row in feature_importance_df.iterrows():
            group_name = self._feature_group_name(row['var'])
            current_count = group_counts.get(group_name, 0)
            if current_count >= max_per_group:
                continue
            row_dict = row.to_dict()
            row_dict['feature_group'] = group_name
            capped_rows.append(row_dict)
            group_counts[group_name] = current_count + 1
            if len(capped_rows) >= max_tree_features:
                break

        capped_df = pd.DataFrame(capped_rows)
        if capped_df.empty:
            return feature_importance_df.head(max_tree_features).copy()
        self.train_log['feature_group_distribution'] = pd.DataFrame(
            [{'feature_group': key, 'selected_count': value} for key, value in sorted(group_counts.items())]
        )
        return capped_df

    def _apply_logistic_group_cap(self, feature_df):
        logistic_config = self.config['scorecard'].get('logistic', {})
        group_config = logistic_config.get('group_feature_limit', {})
        if feature_df.empty or not group_config.get('enable', False):
            return feature_df.copy()

        max_per_group = int(group_config.get('max_per_group', len(feature_df)))
        capped_rows = []
        group_counts = {}
        for _, row in feature_df.iterrows():
            group_name = self._feature_group_name(row['var'])
            current_count = group_counts.get(group_name, 0)
            if current_count >= max_per_group:
                continue
            row_dict = row.to_dict()
            row_dict['feature_group'] = group_name
            capped_rows.append(row_dict)
            group_counts[group_name] = current_count + 1

        capped_df = pd.DataFrame(capped_rows)
        if capped_df.empty:
            return feature_df.copy()
        self.train_log['logistic_group_distribution'] = pd.DataFrame(
            [{'feature_group': key, 'selected_count': value} for key, value in sorted(group_counts.items())]
        )
        return capped_df

    def _build_interaction_plan(self, feature_importance_df):
        interaction_config = self.config['scorecard'].get('feature_engineering', {})
        if not interaction_config.get('enable_interactions', False) or feature_importance_df.empty:
            self.train_log['interaction_plan'] = []
            return []

        top_n = int(interaction_config.get('top_base_features', 8))
        max_interactions = int(interaction_config.get('max_interactions', 6))
        base_features = feature_importance_df['var'].head(top_n).tolist()
        plan = []
        seen = set()
        for idx, left in enumerate(base_features):
            for right in base_features[idx + 1:]:
                left_group = self._feature_group_name(left)
                right_group = self._feature_group_name(right)
                if left_group == right_group:
                    continue
                pair_key = (left, right)
                if pair_key in seen:
                    continue
                seen.add(pair_key)
                plan.append({'name': f'{left}__x__{right}', 'left': left, 'right': right, 'type': 'product'})
                if len(plan) >= max_interactions:
                    self.train_log['interaction_plan'] = plan
                    self.train_log['interaction_plan_df'] = pd.DataFrame(plan)
                    return plan
        self.train_log['interaction_plan'] = plan
        self.train_log['interaction_plan_df'] = pd.DataFrame(plan)
        return plan

    def _apply_interaction_plan(self, df, interaction_plan):
        if not interaction_plan:
            return df
        result = df.copy()
        for item in interaction_plan:
            left = pd.to_numeric(result[item['left']], errors='coerce').fillna(0)
            right = pd.to_numeric(result[item['right']], errors='coerce').fillna(0)
            if item['type'] == 'product':
                result[item['name']] = left * right
        return result

    def _fit_probability_calibrator(self, base_model, calib_X, calib_y, method='platt'):
        calib_pred = np.clip(base_model.predict_proba(calib_X)[:, 1], 1e-6, 1 - 1e-6)
        if method == 'isotonic':
            calibrator = IsotonicRegression(out_of_bounds='clip')
            calibrator.fit(calib_pred, calib_y)
        else:
            calibrator = LogisticRegression(random_state=self.config['seed'])
            calibrator.fit(calib_pred.reshape(-1, 1), calib_y)
        return CalibratedModelWrapper(base_model, calibrator, method=method)

    def train_scorecard(self, data=None, data_path=None, model_type='logistic',
                        y_name=None, date_col=None, save_model=True):
        """Train a scorecard model."""
        self.set_seed()
        self.train_log = {'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

        if data is None:
            data = self.load_data(data_path)

        y_name = y_name or self.config['scorecard']['y_name']
        date_col = date_col or self.config['scorecard'].get('date_col', 'endDateDuration')

        print(f"\n{'=' * 60}")
        print(f"Start training scorecard model: {model_type}")
        print(f"Rows: {len(data)}, Columns: {len(data.columns)}")
        print(f"{'=' * 60}\n")

        if date_col and date_col in data.columns and data[date_col].notna().sum() > len(data) * 0.5:
            train_df, test_df, oot_df = self.split_data_by_date(data, date_col, y_name)
        else:
            print("Warning: invalid date column, using random split")
            train_df, temp = train_test_split(
                data, test_size=0.3, random_state=self.config['seed'], stratify=data[y_name]
            )
            test_df, oot_df = train_test_split(
                temp, test_size=0.5, random_state=self.config['seed'], stratify=temp[y_name]
            )
            train_df = train_df.reset_index(drop=True)
            test_df = test_df.reset_index(drop=True)
            oot_df = oot_df.reset_index(drop=True)

        if model_type == 'logistic':
            feature_cols = self.feature_selection_logistic(train_df, y_name)
            lr_config = self.config['scorecard'].get('logistic', {})
            self.woe_binner = WOEBinning(
                min_bins=lr_config.get('woe_min_bins', 2),
                max_bins=lr_config.get('woe_bins', 7),
                bin_pct=lr_config.get('woe_bin_pct', 0.05),
                raw_bin_multiplier=lr_config.get('woe_raw_bin_multiplier', 3),
                prefer_monotonic=lr_config.get('prefer_monotonic', True),
            )
            for col in feature_cols:
                self.woe_binner.fit(train_df, col, y_name)

            train_X = pd.DataFrame({var: self.woe_binner.transform(train_df, var) for var in feature_cols}).fillna(0)
            test_X = pd.DataFrame({var: self.woe_binner.transform(test_df, var) for var in feature_cols}).fillna(0)
            oot_X = pd.DataFrame({var: self.woe_binner.transform(oot_df, var) for var in feature_cols}).fillna(0)
            train_y = train_df[y_name]
            test_y = test_df[y_name]
            tuning_config = self.config['scorecard'].get('tuning', {})
            tuning_valid_ratio = tuning_config.get('validation_size', 0.2)
            rolling_splits = self.build_rolling_validation_splits(
                train_df,
                date_col,
                y_name,
                n_splits=tuning_config.get('rolling_splits', 3),
                valid_ratio=tuning_valid_ratio,
                min_train_ratio=tuning_config.get('min_train_ratio', 0.4),
            )
            tune_train_df, tune_valid_df = self.build_time_validation_split(
                train_df, date_col, y_name, valid_ratio=tuning_valid_ratio
            )
            tune_all_X = train_X
            tune_all_y = train_y
            tune_valid_X = test_X if tune_valid_df.empty else pd.DataFrame(
                {var: self.woe_binner.transform(tune_valid_df, var) for var in feature_cols}
            ).fillna(0)
            tune_valid_y = test_y if tune_valid_df.empty else tune_valid_df[y_name]

            if tuning_config.get('enable', True):
                print(f"\nStart Logistic tuning (rolling time-window validation, n_trials={tuning_config.get('n_trials', 30)})...")
                best_params = self.tune_logistic(
                    tune_all_X, tune_all_y, tune_valid_X, tune_valid_y,
                    n_trials=tuning_config.get('n_trials', 30),
                    cv_splits=rolling_splits,
                )
                if best_params:
                    best_params['random_state'] = self.config['seed']
                    best_params['max_iter'] = max(int(best_params.get('max_iter', 2000)), 2000)
                    best_params['solver'] = best_params.get('solver', 'liblinear')
                    self.train_log['final_model_params'] = best_params.copy()
                    self.model = LogisticRegression(**best_params)
                else:
                    self.model = LogisticRegression(
                        C=1.0, penalty='l2', class_weight='balanced',
                        solver='liblinear', max_iter=2000, random_state=self.config['seed']
                    )
            else:
                self.model = LogisticRegression(
                    C=1.0, penalty='l2', class_weight='balanced',
                    solver='liblinear', max_iter=2000, random_state=self.config['seed']
                )

        elif model_type == 'lightgbm':
            feature_cols = self.feature_selection_tree(
                train_df,
                y_name,
                model_type='lightgbm',
                reference_df=pd.concat([train_df, test_df], ignore_index=True),
                validation_df=oot_df,
            )
            train_X = train_df[feature_cols].fillna(-9999)
            test_X = test_df[feature_cols].fillna(-9999)
            oot_X = oot_df[feature_cols].fillna(-9999)
            interaction_plan = self.train_log.get('interaction_plan', [])
            if isinstance(interaction_plan, pd.DataFrame):
                interaction_plan = interaction_plan.to_dict('records')
            if interaction_plan:
                train_X = self._apply_interaction_plan(train_X, interaction_plan).fillna(-9999)
                test_X = self._apply_interaction_plan(test_X, interaction_plan).fillna(-9999)
                oot_X = self._apply_interaction_plan(oot_X, interaction_plan).fillna(-9999)
                feature_cols = train_X.columns.tolist()
            train_y = train_df[y_name]
            test_y = test_df[y_name]

            tuning_config = self.config['scorecard'].get('tuning', {})
            tuning_valid_ratio = tuning_config.get('validation_size', 0.2)
            rolling_splits = self.build_rolling_validation_splits(
                train_df,
                date_col,
                y_name,
                n_splits=tuning_config.get('rolling_splits', 3),
                valid_ratio=tuning_valid_ratio,
                min_train_ratio=tuning_config.get('min_train_ratio', 0.4),
            )
            tune_train_df, tune_valid_df = self.build_time_validation_split(
                train_df, date_col, y_name, valid_ratio=tuning_valid_ratio
            )
            tune_all_X = train_df[feature_cols].fillna(-9999)
            tune_all_y = train_df[y_name]
            tune_valid_X = tune_valid_df[feature_cols].fillna(-9999)
            tune_valid_y = tune_valid_df[y_name]

            if tuning_config.get('enable', True):
                print(f"\nStart LightGBM tuning (rolling time-window validation, n_trials={tuning_config.get('n_trials', 30)})...")
                best_params = self.tune_lightgbm(
                    tune_all_X, tune_all_y, tune_valid_X, tune_valid_y,
                    n_trials=tuning_config.get('n_trials', 30),
                    cv_splits=rolling_splits,
                )
                if best_params:
                    best_params['random_state'] = self.config['seed']
                    best_params['verbose'] = -1
                    self.train_log['final_model_params'] = best_params.copy()
                    self.model = lgb.LGBMClassifier(**best_params)
                else:
                    self.model = lgb.LGBMClassifier(
                        n_estimators=200, learning_rate=0.1, max_depth=5,
                        random_state=self.config['seed'], verbose=-1
                    )
            else:
                self.model = lgb.LGBMClassifier(
                    n_estimators=200, learning_rate=0.1, max_depth=5,
                    random_state=self.config['seed'], verbose=-1
                )

        elif model_type == 'xgboost':
            feature_cols = self.feature_selection_tree(
                train_df,
                y_name,
                model_type='xgboost',
                reference_df=pd.concat([train_df, test_df], ignore_index=True),
                validation_df=oot_df,
            )
            train_X = train_df[feature_cols].fillna(-9999)
            test_X = test_df[feature_cols].fillna(-9999)
            oot_X = oot_df[feature_cols].fillna(-9999)
            interaction_plan = self.train_log.get('interaction_plan', [])
            if isinstance(interaction_plan, pd.DataFrame):
                interaction_plan = interaction_plan.to_dict('records')
            if interaction_plan:
                train_X = self._apply_interaction_plan(train_X, interaction_plan).fillna(-9999)
                test_X = self._apply_interaction_plan(test_X, interaction_plan).fillna(-9999)
                oot_X = self._apply_interaction_plan(oot_X, interaction_plan).fillna(-9999)
                feature_cols = train_X.columns.tolist()
            train_y = train_df[y_name]
            test_y = test_df[y_name]

            tuning_config = self.config['scorecard'].get('tuning', {})
            tuning_valid_ratio = tuning_config.get('validation_size', 0.2)
            rolling_splits = self.build_rolling_validation_splits(
                train_df,
                date_col,
                y_name,
                n_splits=tuning_config.get('rolling_splits', 3),
                valid_ratio=tuning_valid_ratio,
                min_train_ratio=tuning_config.get('min_train_ratio', 0.4),
            )
            tune_train_df, tune_valid_df = self.build_time_validation_split(
                train_df, date_col, y_name, valid_ratio=tuning_valid_ratio
            )
            tune_all_X = train_df[feature_cols].fillna(-9999)
            tune_all_y = train_df[y_name]
            tune_valid_X = tune_valid_df[feature_cols].fillna(-9999)
            tune_valid_y = tune_valid_df[y_name]

            if tuning_config.get('enable', True):
                print(f"\nStart XGBoost tuning (rolling time-window validation, n_trials={tuning_config.get('n_trials', 30)})...")
                best_params = self.tune_xgboost(
                    tune_all_X, tune_all_y, tune_valid_X, tune_valid_y,
                    n_trials=tuning_config.get('n_trials', 30),
                    cv_splits=rolling_splits,
                )
                if best_params:
                    best_params['random_state'] = self.config['seed']
                    best_params['use_label_encoder'] = False
                    best_params['eval_metric'] = 'logloss'
                    best_params['n_jobs'] = 1
                    self.train_log['final_model_params'] = best_params.copy()
                    self.model = xgb.XGBClassifier(**best_params)
                else:
                    self.model = xgb.XGBClassifier(
                        n_estimators=200, learning_rate=0.1, max_depth=5,
                        random_state=self.config['seed'], use_label_encoder=False, eval_metric='logloss', n_jobs=1
                    )
            else:
                self.model = xgb.XGBClassifier(
                    n_estimators=200, learning_rate=0.1, max_depth=5,
                    random_state=self.config['seed'], use_label_encoder=False, eval_metric='logloss', n_jobs=1
                )
        elif model_type == 'ensemble':
            feature_cols = self.feature_selection_tree(
                train_df,
                y_name,
                model_type='xgboost',
                reference_df=pd.concat([train_df, test_df], ignore_index=True),
                validation_df=oot_df,
            )
            train_X = train_df[feature_cols].fillna(-9999)
            test_X = test_df[feature_cols].fillna(-9999)
            oot_X = oot_df[feature_cols].fillna(-9999)
            interaction_plan = self.train_log.get('interaction_plan', [])
            if isinstance(interaction_plan, pd.DataFrame):
                interaction_plan = interaction_plan.to_dict('records')
            if interaction_plan:
                train_X = self._apply_interaction_plan(train_X, interaction_plan).fillna(-9999)
                test_X = self._apply_interaction_plan(test_X, interaction_plan).fillna(-9999)
                oot_X = self._apply_interaction_plan(oot_X, interaction_plan).fillna(-9999)
                feature_cols = train_X.columns.tolist()
            train_y = train_df[y_name]
            test_y = test_df[y_name]

            tuning_config = self.config['scorecard'].get('tuning', {})
            tuning_valid_ratio = tuning_config.get('validation_size', 0.2)
            rolling_splits = self.build_rolling_validation_splits(
                train_df,
                date_col,
                y_name,
                n_splits=tuning_config.get('rolling_splits', 3),
                valid_ratio=tuning_valid_ratio,
                min_train_ratio=tuning_config.get('min_train_ratio', 0.4),
            )
            tune_train_df, tune_valid_df = self.build_time_validation_split(
                train_df, date_col, y_name, valid_ratio=tuning_valid_ratio
            )
            tune_all_X = train_X
            tune_all_y = train_y
            tune_valid_X = test_X if tune_valid_df.empty else tune_valid_df[feature_cols].fillna(-9999)
            tune_valid_y = test_y if tune_valid_df.empty else tune_valid_df[y_name]

            print(f"\nStart ensemble base-model tuning (rolling time-window validation, n_trials={tuning_config.get('n_trials', 30)})...")
            xgb_params = self.tune_xgboost(
                tune_all_X, tune_all_y, tune_valid_X, tune_valid_y,
                n_trials=tuning_config.get('n_trials', 30),
                cv_splits=rolling_splits,
            ) if tuning_config.get('enable', True) else None
            xgb_candidates = self.train_log.get('tuning_candidates', pd.DataFrame()).copy()

            lgb_params = self.tune_lightgbm(
                tune_all_X, tune_all_y, tune_valid_X, tune_valid_y,
                n_trials=tuning_config.get('n_trials', 30),
                cv_splits=rolling_splits,
            ) if tuning_config.get('enable', True) else None
            lgb_candidates = self.train_log.get('tuning_candidates', pd.DataFrame()).copy()

            if xgb_params is None:
                xgb_params = {
                    'n_estimators': 200, 'learning_rate': 0.1, 'max_depth': 5,
                    'random_state': self.config['seed'], 'use_label_encoder': False,
                    'eval_metric': 'logloss', 'n_jobs': 1,
                }
            else:
                xgb_params['random_state'] = self.config['seed']
                xgb_params['use_label_encoder'] = False
                xgb_params['eval_metric'] = 'logloss'
                xgb_params['n_jobs'] = 1

            if lgb_params is None:
                lgb_params = {
                    'n_estimators': 200, 'learning_rate': 0.1, 'max_depth': 5,
                    'random_state': self.config['seed'], 'verbose': -1,
                }
            else:
                lgb_params['random_state'] = self.config['seed']
                lgb_params['verbose'] = -1

            xgb_model = xgb.XGBClassifier(**xgb_params)
            lgb_model = lgb.LGBMClassifier(**lgb_params)
            xgb_model.fit(train_X, train_y, verbose=False)
            lgb_model.fit(train_X, train_y)

            ensemble_weights = self.config['scorecard'].get('ensemble', {}).get(
                'weights', {'xgboost': 0.5, 'lightgbm': 0.5}
            )
            weight_xgb = float(ensemble_weights.get('xgboost', 0.5))
            weight_lgb = float(ensemble_weights.get('lightgbm', 0.5))
            total_weight = max(weight_xgb + weight_lgb, 1e-6)
            normalized_weights = [weight_xgb / total_weight, weight_lgb / total_weight]

            importance_xgb = getattr(xgb_model, 'feature_importances_', np.zeros(len(feature_cols)))
            importance_lgb = getattr(lgb_model, 'feature_importances_', np.zeros(len(feature_cols)))
            ensemble_importance = (
                np.asarray(importance_xgb, dtype=float) * normalized_weights[0]
                + np.asarray(importance_lgb, dtype=float) * normalized_weights[1]
            )
            self.model = WeightedEnsembleModel(
                [xgb_model, lgb_model],
                normalized_weights,
                feature_importances=ensemble_importance,
            )
            self.train_log['final_model_params'] = {
                'ensemble_weights': {'xgboost': normalized_weights[0], 'lightgbm': normalized_weights[1]},
                'xgboost': xgb_params.copy(),
                'lightgbm': lgb_params.copy(),
            }
            self.train_log['ensemble_components'] = ['xgboost', 'lightgbm']
            if not xgb_candidates.empty:
                xgb_candidates = xgb_candidates.copy()
                xgb_candidates['base_model'] = 'xgboost'
            if not lgb_candidates.empty:
                lgb_candidates = lgb_candidates.copy()
                lgb_candidates['base_model'] = 'lightgbm'
            combined_candidates = pd.concat(
                [df for df in [xgb_candidates, lgb_candidates] if not df.empty],
                ignore_index=True,
            ) if (not xgb_candidates.empty or not lgb_candidates.empty) else pd.DataFrame()
            if not combined_candidates.empty:
                self.train_log['tuning_candidates'] = combined_candidates
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        self.var_list = feature_cols
        oot_y = oot_df[y_name]
        print(f"\nSelected features: {len(self.var_list)}")
        print(f"Bad rate - train: {train_y.mean():.4f}, test: {test_y.mean():.4f}, oot: {oot_y.mean():.4f}")

        print(f"\nTraining {model_type} model...")
        if model_type != 'ensemble':
            self.model.fit(train_X, train_y)

        if model_type == 'logistic':
            repeats = self.config['scorecard'].get('tuning', {}).get('candidate_repeats', 3)
            final_params = self.train_log.get('final_model_params', {})
            if final_params:
                self.run_tree_time_window_validation(
                    'logistic',
                    final_params,
                    train_X,
                    train_y,
                    rolling_splits,
                    repeats=repeats,
                )
        elif model_type in ['lightgbm', 'xgboost']:
            repeats = self.config['scorecard'].get('tuning', {}).get('candidate_repeats', 3)
            final_params = self.train_log.get('final_model_params', {})
            self.run_tree_time_window_validation(
                model_type,
                final_params,
                train_X,
                train_y,
                rolling_splits,
                repeats=repeats,
            )
        elif model_type == 'ensemble':
            repeats = self.config['scorecard'].get('tuning', {}).get('candidate_repeats', 3)
            final_params = self.train_log.get('final_model_params', {})
            self.run_ensemble_time_window_validation(
                final_params.get('xgboost', {}),
                final_params.get('lightgbm', {}),
                train_X,
                train_y,
                rolling_splits,
                repeats=repeats,
                weights=final_params.get('ensemble_weights', {}),
            )

        train_pred = self.model.predict_proba(train_X)[:, 1]
        test_pred = self.model.predict_proba(test_X)[:, 1]
        oot_pred = self.model.predict_proba(oot_X)[:, 1]

        self.train_log['evaluation_data'] = {
            'y_name': y_name,
            'date_col': date_col,
            'train': train_df[[c for c in [y_name, date_col] if c in train_df.columns]].copy(),
            'test': test_df[[c for c in [y_name, date_col] if c in test_df.columns]].copy(),
            'oot': oot_df[[c for c in [y_name, date_col] if c in oot_df.columns]].copy(),
            'train_pred': train_pred,
            'test_pred': test_pred,
            'oot_pred': oot_pred,
        }

        train_metrics = self.evaluate_model(train_y, train_pred, 'train_')
        test_metrics = self.evaluate_model(test_y, test_pred, 'test_')
        oot_metrics = self.evaluate_model(oot_y, oot_pred, 'oot_')
        metrics = {**train_metrics, **test_metrics, **oot_metrics}

        print(f"\n{'='*60}")
        print("Evaluation summary:")
        print(f"  Train - AUC: {train_metrics['train_auc']:.4f}, KS: {train_metrics['train_ks']:.4f}")
        print(f"  Test  - AUC: {test_metrics['test_auc']:.4f}, KS: {test_metrics['test_ks']:.4f}")
        print(f"  OOT   - AUC: {oot_metrics['oot_auc']:.4f}, KS: {oot_metrics['oot_ks']:.4f}")
        print(f"{'='*60}\n")

        self.train_log['metrics'] = metrics
        self.train_log['model_type'] = model_type
        self.train_log['end_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        if save_model:
            self._save_model(
                model_type, metrics,
                {'train': train_y, 'test': test_y, 'oot': oot_y},
                {'train': train_pred, 'test': test_pred, 'oot': oot_pred}
            )

        return self.model, metrics, self.var_list

    def _save_model(self, model_type, metrics, y_true_dict, y_pred_dict):
        save_model_artifacts(self, model_type, metrics, y_true_dict, y_pred_dict)


class CalibratedModelWrapper:
    def __init__(self, base_model, calibrator, method="platt"):
        self.base_model = base_model
        self.calibrator = calibrator
        self.method = method
        if hasattr(base_model, "feature_importances_"):
            self.feature_importances_ = getattr(base_model, "feature_importances_")

    def predict_proba(self, X):
        base_pred = self.base_model.predict_proba(X)[:, 1]
        if self.method == "isotonic":
            calibrated = np.clip(self.calibrator.predict(base_pred), 1e-6, 1 - 1e-6)
        else:
            calibrated = np.clip(self.calibrator.predict_proba(base_pred.reshape(-1, 1))[:, 1], 1e-6, 1 - 1e-6)
        return np.column_stack([1 - calibrated, calibrated])


class WeightedEnsembleModel:
    def __init__(self, models, weights, feature_importances=None):
        self.models = models
        self.weights = weights
        if feature_importances is not None:
            self.feature_importances_ = feature_importances

    def predict_proba(self, X):
        preds = None
        for model, weight in zip(self.models, self.weights):
            current = model.predict_proba(X)[:, 1] * weight
            preds = current if preds is None else preds + current
        preds = np.clip(preds, 1e-6, 1 - 1e-6)
        return np.column_stack([1 - preds, preds])

    def _export_pmml(self, output_dir, prefix, model_type):
        export_pmml(self, output_dir, prefix, model_type)

    def _export_scorecard_excel(self, output_dir, prefix):
        export_scorecard_excel(self, output_dir, prefix)

    def _export_training_report(self, output_dir, prefix, y_true_dict, y_pred_dict):
        export_training_report(self, output_dir, prefix, y_true_dict, y_pred_dict)

    def predict(self, data, model_path=None, woe_path=None, var_list_path=None):
        """使用模型预测"""
        if model_path:
            self.model = joblib.load(model_path)
        if woe_path:
            self.woe_binner = joblib.load(woe_path)
        if var_list_path:
            var_df = pd.read_csv(var_list_path)
            self.var_list = var_df['VAR'].tolist()
        
        if self.model is None:
            raise ValueError("模型未加载")
        
        if self.woe_binner:
            X = pd.DataFrame({var: self.woe_binner.transform(data, var) for var in self.var_list}).fillna(0)
        else:
            X = data[self.var_list].fillna(-9999)
        
        return self.model.predict_proba(X)[:, 1]


def main():
    config = {
        'seed': 1234,
        'output_dir': './output',
        'model_prefix': 'risk_model',
    }
    
    trainer = ModelTrainer(config)
    print("="*60)
    print("风控模型统一训练脚本")
    print("="*60)
    print("\n使用方法:")
    print("  # 逻辑回归（WOE+IV+VIF）")
    print("  trainer.train_scorecard(data_path='data.pkl', model_type='logistic')")
    print("\n  # LightGBM（原始特征+PMML）")
    print("  trainer.train_scorecard(data_path='data.pkl', model_type='lightgbm')")
    print("\n  # XGBoost（原始特征+PMML）")
    print("  trainer.train_scorecard(data_path='data.pkl', model_type='xgboost')")


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""
测试脚本 - 运行评分卡模型训练
测试模型：逻辑回归、XGBoost、LightGBM
"""
import os
import sys
import json
import copy
from datetime import datetime

import numpy as np
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(os.path.dirname(CURRENT_DIR), 'src')
sys.path.insert(0, SRC_DIR)

from risk_model import ModelTrainer

def main():
    print("="*60)
    print("评分卡模型训练测试")
    print("="*60)
    
    data_path = os.path.join(os.path.dirname(CURRENT_DIR), 'data.pkl')
    if not os.path.exists(data_path):
        print(f"错误: 数据文件 {data_path} 不存在!")
        return
    
    data = pd.read_pickle(data_path)
    print(f"\n数据加载完成: {len(data)} 条记录, {len(data.columns)} 列")
    print(f"目标变量分布:\n{data['target'].value_counts()}")
    
    date_col = 'endDateDuration'
    if date_col in data.columns:
        valid_dates = data[date_col].notna().sum()
        print(f"\n日期列 '{date_col}' 有效值: {valid_dates}/{len(data)}")
    
    config_path = os.path.join(os.path.dirname(CURRENT_DIR), 'configs', 'default_config.json')
    with open(config_path, 'r', encoding='utf-8') as f:
        base_config = json.load(f)
    run_suffix = datetime.now().strftime('%H%M')
    base_config['output_dir'] = os.path.join('.', 'output', 'tests')
    
    results = {}
    
    # 测试逻辑回归
    print("\n" + "="*60)
    print("测试1: 训练逻辑回归模型 (WOE+IV+VIF+相关性)")
    print("="*60)
    
    config_lr = copy.deepcopy(base_config)
    config_lr['model_prefix'] = f'lr_{run_suffix}'
    trainer_lr = ModelTrainer(config_lr)
    try:
        model_lr, metrics_lr, var_list_lr = trainer_lr.train_scorecard(
            data=data,
            model_type='logistic'
        )
        results['logistic'] = {
            'status': '成功',
            'auc': metrics_lr.get('test_auc', 0),
            'ks': metrics_lr.get('test_ks', 0),
            'oot_auc': metrics_lr.get('oot_auc', 0),
            'oot_ks': metrics_lr.get('oot_ks', 0),
            'var_count': len(var_list_lr)
        }
    except Exception as e:
        import traceback
        print(f"逻辑回归训练失败: {e}")
        traceback.print_exc()
        results['logistic'] = {'status': f'失败: {str(e)}'}
    
    # 测试LightGBM
    print("\n" + "="*60)
    print("测试2: 训练LightGBM模型 (原始特征+PMML)")
    print("="*60)
    
    config_lgb = copy.deepcopy(base_config)
    config_lgb['model_prefix'] = f'lgb_{run_suffix}'
    trainer_lgb = ModelTrainer(config_lgb)
    try:
        model_lgb, metrics_lgb, var_list_lgb = trainer_lgb.train_scorecard(
            data=data,
            model_type='lightgbm'
        )
        results['lightgbm'] = {
            'status': '成功',
            'auc': metrics_lgb.get('test_auc', 0),
            'ks': metrics_lgb.get('test_ks', 0),
            'oot_auc': metrics_lgb.get('oot_auc', 0),
            'oot_ks': metrics_lgb.get('oot_ks', 0),
            'var_count': len(var_list_lgb)
        }
    except Exception as e:
        import traceback
        print(f"LightGBM训练失败: {e}")
        traceback.print_exc()
        results['lightgbm'] = {'status': f'失败: {str(e)}'}
    
    # 测试XGBoost
    print("\n" + "="*60)
    print("测试3: 训练XGBoost模型 (原始特征+PMML)")
    print("="*60)
    
    config_xgb = copy.deepcopy(base_config)
    config_xgb['model_prefix'] = f'xgb_{run_suffix}'
    trainer_xgb = ModelTrainer(config_xgb)
    try:
        model_xgb, metrics_xgb, var_list_xgb = trainer_xgb.train_scorecard(
            data=data,
            model_type='xgboost'
        )
        results['xgboost'] = {
            'status': '成功',
            'auc': metrics_xgb.get('test_auc', 0),
            'ks': metrics_xgb.get('test_ks', 0),
            'oot_auc': metrics_xgb.get('oot_auc', 0),
            'oot_ks': metrics_xgb.get('oot_ks', 0),
            'var_count': len(var_list_xgb)
        }
    except Exception as e:
        import traceback
        print(f"XGBoost训练失败: {e}")
        traceback.print_exc()
        results['xgboost'] = {'status': f'失败: {str(e)}'}
    
    # 输出结果汇总
    print("\n" + "="*60)
    print("训练结果汇总")
    print("="*60)
    
    print(f"\n{'模型':<15} {'状态':<15} {'测试集AUC':<12} {'测试集KS':<12} {'OOT-AUC':<12} {'OOT-KS':<12} {'变量数':<8}")
    print("-"*86)
    for model_name, result in results.items():
        if result['status'] == '成功':
            print(f"{model_name:<15} {result['status']:<15} {result['auc']:.4f}       {result['ks']:.4f}       {result['oot_auc']:.4f}       {result['oot_ks']:.4f}       {result['var_count']:<8}")
        else:
            print(f"{model_name:<15} {result['status']:<15}")
    
    print("\n输出文件列表:")
    output_dir = base_config['output_dir']
    if os.path.exists(output_dir):
        all_files = sorted([f for f in os.listdir(output_dir) if f.startswith(('lr_', 'lgb_', 'xgb_'))])
        for f in all_files:
            file_path = os.path.join(output_dir, f)
            file_size = os.path.getsize(file_path) / 1024
            print(f"  - {f} ({file_size:.1f} KB)")
    
    print("\n测试完成!")

if __name__ == '__main__':
    main()

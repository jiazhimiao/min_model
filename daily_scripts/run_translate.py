import traceback
import sys

try:
    input_file = r'd:\Trae_pro\min_model\daily_scripts\filtered_sms_all_sheets_v3_1.xlsx'
    output_file = r'd:\Trae_pro\min_model\daily_scripts\filtered_sms_all_sheets_v3_1_translated.xlsx'
    
    sys.argv = [
        'run_translate.py',
        input_file,
        output_file,
        '--batch-size', '12',
        '--max-workers', '3',
        '--request-delay', '1.02',
    ]
    
    exec(open(r'd:\Trae_pro\min_model\daily_scripts\translate_filtered_sms_results_v2.py', encoding='utf-8').read())
    
except Exception as e:
    print(f"错误: {e}")
    traceback.print_exc()

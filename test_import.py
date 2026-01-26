import sys
sys.path.insert(0, '.')
try:
    from utils.visualization import create_full_prediction_chart, get_chinese_config
    print("导入成功")
except Exception as e:
    import traceback
    traceback.print_exc()

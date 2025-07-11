import os
import sys
sys.path.append((os.path.dirname(os.path.abspath(__file__))))

# print(sys.path)

# from convert_torch_to_trt import convert_models
from engine import Engine
from trt_executor import TrtExecutor

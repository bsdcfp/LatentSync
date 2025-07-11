import base64
import gzip
import lz4.frame
import numpy as np
import torch


def tensor_to_json_compressed(tensor, compression='lz4'):
    """支持多种数据类型的压缩函数"""
    if isinstance(tensor, torch.Tensor):
        # 避免多次CPU转换
        tensor = tensor.detach().cpu().numpy()
    
    # 确保我们有正确的数据类型信息
    original_dtype = tensor.dtype
    print(f"Compressing tensor: shape={tensor.shape}, dtype={original_dtype}, size={tensor.nbytes} bytes")
    
    # 转换为bytes
    tensor_bytes = tensor.tobytes()
    
    # 使用更快的压缩
    if compression == 'lz4':
        compressed = lz4.frame.compress(tensor_bytes)
    elif compression == 'gzip' or compression == 'gzip_fast':
        compressed = gzip.compress(tensor_bytes, compresslevel=1)
    else:
        compressed = tensor_bytes
    
    # 使用更快的base64编码
    encoded = base64.b64encode(compressed).decode('ascii')
    
    return {
        'data': encoded,
        'shape': list(tensor.shape),  # 确保shape是serializable的
        'dtype': str(original_dtype),  # 确保dtype字符串格式正确
        'compression': compression,
        'original_bytes': tensor.nbytes  # 添加原始字节数用于验证
    }


def json_to_tensor_decompressed(json_data):
    """支持多种数据类型的解压函数"""
    try:
        # 获取元信息
        compression = json_data.get('compression', 'none')
        shape = json_data['shape']
        dtype_str = json_data['dtype']
        expected_bytes = json_data.get('original_bytes', None)
        
        print(f"Decompressing: compression={compression}, shape={shape}, dtype={dtype_str}")
        
        # 解码base64
        compressed = base64.b64decode(json_data['data'])
        
        # 解压缩
        if compression == 'lz4':
            tensor_bytes = lz4.frame.decompress(compressed)
        elif compression in ['gzip', 'gzip_fast']:
            tensor_bytes = gzip.decompress(compressed)
        else:
            tensor_bytes = compressed
        
        # 验证数据类型
        try:
            np_dtype = np.dtype(dtype_str)
        except TypeError:
            # 处理一些特殊的dtype字符串格式
            if dtype_str == 'torch.float16':
                np_dtype = np.float16
            elif dtype_str == 'torch.float32':
                np_dtype = np.float32
            elif dtype_str == 'torch.float64':
                np_dtype = np.float64
            else:
                # 尝试从字符串解析
                np_dtype = np.dtype(dtype_str.replace('torch.', ''))
        
        print(f"Parsed dtype: {np_dtype}, itemsize: {np_dtype.itemsize}")
        
        # 计算期望的字节数
        expected_elements = np.prod(shape)
        calculated_bytes = expected_elements * np_dtype.itemsize
        
        print(f"Expected elements: {expected_elements}")
        print(f"Calculated bytes: {calculated_bytes}")
        print(f"Actual bytes: {len(tensor_bytes)}")
        if expected_bytes:
            print(f"Original bytes: {expected_bytes}")
        
        # 验证数据长度
        if len(tensor_bytes) != calculated_bytes:
            raise ValueError(
                f"Data length mismatch: "
                f"expected {calculated_bytes} bytes for {expected_elements} elements of {np_dtype}, "
                f"got {len(tensor_bytes)} bytes"
            )
        
        # 检查是否能整除
        if len(tensor_bytes) % np_dtype.itemsize != 0:
            raise ValueError(
                f"Data length {len(tensor_bytes)} is not divisible by dtype size {np_dtype.itemsize}"
            )
        
        # 从buffer创建array
        tensor = np.frombuffer(tensor_bytes, dtype=np_dtype)
        
        # 重塑形状
        reshaped_tensor = tensor.reshape(shape)
        
        print(f"Successfully decompressed tensor: shape={reshaped_tensor.shape}, dtype={reshaped_tensor.dtype}")
        
        return reshaped_tensor
        
    except Exception as e:
        print(f"Error during decompression: {e}")
        print(f"Input data keys: {list(json_data.keys())}")
        print(f"Shape: {json_data.get('shape', 'unknown')}")
        print(f"Dtype string: {json_data.get('dtype', 'unknown')}")
        print(f"Compression: {json_data.get('compression', 'unknown')}")
        print(f"Data length: {len(json_data.get('data', ''))}")
        raise
"""
LMSFC (Learning-based Multi-dimensional Space-Filling Curves) Python包装器
为LMSFC C++实现提供Python接口，集成到sfcBenchmark框架中
"""
import ctypes
import numpy as np
from typing import List, Tuple, Optional
import os
import subprocess
import sys
from src.mvp._paths import lmsfc_lib_candidates as _lmsfc_candidates

class LMSFCProcessor:
    """LMSFC C++实现的Python包装器"""
    
    def __init__(self, order: int, masks: Optional[List[int]] = None):
        """
        初始化LMSFC处理器
        
        参数:
            order: Hilbert曲线的阶数 (网格大小 = 2^order)
            masks: 学习到的位掩码列表 (可选)
        """
        self.order = order
        self.grid_size = 2 ** order
        self.dimensions = 2  # LMSFC主要支持2D
        
        # 设置默认掩码或使用提供的掩码
        if masks is None:
            # 标准Z曲线掩码
            self.masks = [
                0x5555555555555555,  # X维度掩码 (奇数位)
                0xAAAAAAAAAAAAA  # Y维度掩码 (偶数位)
            ]
        else:
            self.masks = masks
            
        # 尝试加载C++共享库
        self._load_library()
    
    def _load_library(self):
        """加载LMSFC C++共享库"""
        try:
            # 查找共享库文件
            lib_path = self._find_library()
            if lib_path is None:
                raise FileNotFoundError("LMSFC共享库未找到")
                
            # 加载库
            self.lib = ctypes.CDLL(lib_path)
            self._setup_function_signatures()
            self.available = True
            print(f"成功加载LMSFC库: {lib_path}")
            
        except Exception as e:
            print(f"警告: 无法加载LMSFC库: {e}")
            self.lib = None
            self.available = False
    
    def _find_library(self) -> Optional[str]:
        """查找LMSFC共享库文件"""
        # 可能的库文件位置
        possible_paths = _lmsfc_candidates()
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        return None
    
    def _setup_function_signatures(self):
        """设置C++函数的签名"""
        if self.lib is None:
            return
            
        # 这里需要根据LMSFC的实际C++接口定义函数签名
        # 由于LMSFC主要是模板类，我们需要创建C接口包装器
        pass
    
    def compile_library(self) -> bool:
        """编译LMSFC C++库"""
        try:
            lmsfc_dir = os.environ.get("LMSFC_ROOT", "")
            
            # 检查是否有CMakeLists.txt
            cmake_file = os.path.join(lmsfc_dir, "CMakeLists.txt")
            if os.path.exists(cmake_file):
                # 使用CMake构建
                build_dir = os.path.join(lmsfc_dir, "build")
                os.makedirs(build_dir, exist_ok=True)
                
                subprocess.run(["cmake", ".."], cwd=build_dir, check=True)
                subprocess.run(["make", "-j4"], cwd=build_dir, check=True)
            else:
                # 使用g++直接编译
                cmd = [
                    "g++", "-std=c++17", "-shared", "-fPIC", "-O3",
                    "-I", "pgm", "-I", "entities", "-I", ".",
                    "test.cpp", "-o", "liblmsfc.so"
                ]
                subprocess.run(cmd, cwd=lmsfc_dir, check=True)
            
            print("LMSFC库编译成功")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"编译LMSFC库失败: {e}")
            return False
        except Exception as e:
            print(f"编译过程中出现错误: {e}")
            return False
    
    def calculate_z_address(self, x: int, y: int) -> int:
        """
        计算点的Z地址 (使用学习到的掩码)
        
        参数:
            x, y: 2D坐标
            
        返回:
            Z地址值
        """
        if not self.available:
            # Fallback到标准Z曲线计算
            return self._standard_z_encode(x, y)
            
        # 使用学习到的掩码进行编码
        z_addr = 0
        
        # 应用X维度掩码
        x_masked = self._apply_mask(x, self.masks[0])
        z_addr |= x_masked
        
        # 应用Y维度掩码  
        y_masked = self._apply_mask(y, self.masks[1])
        z_addr |= y_masked
        
        return z_addr
    
    def _apply_mask(self, value: int, mask: int) -> int:
        """应用位掩码到值"""
        result = 0
        bit_pos = 0
        
        while mask > 0:
            if mask & 1:
                if value & (1 << bit_pos):
                    result |= (1 << bit_pos)
            mask >>= 1
            bit_pos += 1
            
        return result
    
    def _standard_z_encode(self, x: int, y: int) -> int:
        """标准Z曲线编码 (fallback)"""
        z = 0
        for i in range(32):  # 假设32位坐标
            z |= ((x & (1 << i)) << i) | ((y & (1 << i)) << (i + 1))
        return z
    
    def query_segments(self, min_x: int, min_y: int, max_x: int, max_y: int) -> List[Tuple[int, int]]:
        """
        计算查询窗口的Z曲线段
        
        参数:
            min_x, min_y: 查询窗口左下角
            max_x, max_y: 查询窗口右上角
            
        返回:
            Z曲线段列表 [(start, end), ...]
        """
        if not self.available:
            return self._fallback_query_segments(min_x, min_y, max_x, max_y)
            
        # 这里应该调用LMSFC的QueryPartition方法
        # 由于需要C接口包装，暂时使用fallback
        return self._fallback_query_segments(min_x, min_y, max_x, max_y)
    
    def _fallback_query_segments(self, min_x: int, min_y: int, max_x: int, max_y: int) -> List[Tuple[int, int]]:
        """查询段计算的fallback实现"""
        segments = []
        
        # 计算查询窗口的最小和最大Z值
        min_z = self.calculate_z_address(min_x, min_y)
        max_z = self.calculate_z_address(max_x, max_y)
        
        # 简化实现：返回单个连续段
        # 实际LMSFC会进行更精确的分割
        segments.append((min_z, max_z))
        
        return segments

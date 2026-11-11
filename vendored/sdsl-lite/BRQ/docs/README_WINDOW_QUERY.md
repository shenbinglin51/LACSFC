# 窗口查询算法 (Window Query Algorithm)

> Status: Historical/reference note. The current maintained implementation is AC-BRQ v2 under `../code/`.


这是从原始main.cpp中抽取出来的独立窗口查询算法实现，基于Hilbert空间填充曲线进行高效的矩形窗口查询。

## 文件结构

### 完整版本 (需要SDSL库)
- `window_query.hpp` - 完整版头文件，使用SDSL库的压缩数据结构
- `window_query.cpp` - 完整版实现文件
- `window_query_example.cpp` - 完整版示例程序
- `Makefile.window_query` - 完整版编译文件

### 简化版本 (独立实现)
- `window_query_simple.hpp` - 简化版头文件，不依赖SDSL库
- `window_query_simple.cpp` - 简化版实现文件
- `window_query_simple_example.cpp` - 简化版示例程序
- `Makefile.simple` - 简化版编译文件

## 算法核心功能

### 主要组件

1. **HilbertInfo结构体**: 存储Hilbert编码、颜色和入口方向信息
2. **QueryWindow类**: 定义查询窗口的坐标和尺寸
3. **WindowQueryProcessor类**: 核心处理器，包含以下功能：
   - Hilbert空间初始化
   - 窗口查询处理
   - 入口和出口点识别

### 核心算法

1. **Hilbert编码计算**: `computHilbertCode()` - 计算二维坐标的Hilbert编码
2. **空间初始化**: `initialHilbertSpace()` - 构建Hilbert空间和倒排索引
3. **窗口查询**: `getEntrance()` - 获取查询窗口的所有入口和出口点

## 编译和运行

### 简化版本 (推荐)

```bash
# 编译
make -f Makefile.simple

# 运行示例
./window_query_simple_example

# 清理
make -f Makefile.simple clean
```

### 完整版本 (需要SDSL库)

```bash
# 确保SDSL库已编译
make  # 在主项目目录中编译SDSL库

# 编译窗口查询算法
make -f Makefile.window_query

# 运行示例
./window_query_example
```

## 使用示例

```cpp
#include "window_query_simple.hpp"

int main() {
    // 创建处理器 (10阶Hilbert曲线，1024x1024网格)
    WindowQueryProcessor processor(10);
    
    // 初始化
    processor.initialize();
    
    // 创建查询窗口
    QueryWindow window(8, 7, 41, 7);  // x, y, width, height
    
    // 执行查询
    std::vector<int> entrances, exits;
    processor.getEntrance(window, entrances, exits);
    
    // 输出结果
    std::cout << "Found " << entrances.size() << " entrance segments" << std::endl;
    
    return 0;
}
```

## 性能特点

- **初始化时间**: ~275ms (10阶，1024x1024网格)
- **查询时间**: <1ms per query (小型查询窗口)
- **内存使用**: 优化的倒排索引结构
- **支持阶数**: 可配置 (默认10阶)

## 算法原理

1. **Hilbert曲线**: 使用空间填充曲线将二维空间映射到一维
2. **倒排索引**: 为每个方向的入口点构建位向量索引
3. **边界处理**: 分别处理查询窗口的上下左右四个边界
4. **快速查询**: 使用rank/select操作快速定位相关入口点

## 输入格式

### CSV文件格式
```
x,y,width,height
8,7,41,7
33,40,26,6
100,200,50,30
```

### 程序化创建
```cpp
std::vector<QueryWindow> windows = {
    QueryWindow(8, 7, 41, 7),
    QueryWindow(33, 40, 26, 6)
};
```

## 扩展功能

- `computeHilbertRange()`: 计算给定区域的Hilbert编码范围
- `readQueryWindows()`: 从CSV文件读取查询窗口
- 内置排序算法: 基数排序优化

## 注意事项

1. 网格大小必须是2的幂次方 (由Hilbert阶数决定)
2. 查询窗口坐标不能超出网格边界
3. 简化版本使用std::vector<bool>，完整版本使用压缩位向量
4. 初始化过程会预计算所有必要的数据结构

## 测试结果

运行示例程序的输出：
```
Initializing Hilbert space...
Initialization completed in 275 ms
Grid size: 1024x1024

Processing 4 test queries...
Query 1: (8,7,41,7) -> 26 entrance segments, 26 exit segments
Query 2: (33,40,26,6) -> 10 entrance segments, 10 exit segments
Query 3: (100,200,50,30) -> 22 entrance segments, 22 exit segments
Query 4: (0,0,10,10) -> 4 entrance segments, 4 exit segments

Total segments found: 62
Query processing time: 0 ms
```

## 从原始代码的改进

1. **模块化设计**: 将算法封装为独立的类
2. **简化依赖**: 提供不依赖SDSL库的版本
3. **接口友好**: 清晰的API设计
4. **文档完善**: 详细的使用说明和示例
5. **编译简化**: 独立的Makefile配置

# BRQ 窗口查询算法思想说明

> Status: Historical/reference note. The current maintained implementation is AC-BRQ v2 under `../code/`.


## 1. 算法目标

BRQ 的目标是在二维 Hilbert 空间中，对一个矩形查询窗口 `Q = (x, y, width, height)`，快速找出 Hilbert 曲线穿过该窗口时形成的若干连续区间段。

换句话说，二维窗口中的点在 Hilbert 曲线上并不一定对应一个连续区间，而通常会被分成多个连续片段：

```text
S = {(En[0], Ex[0]), (En[1], Ex[1]), ..., (En[k], Ex[k])}
```

其中：

- `En[i]` 表示第 `i` 个连续片段的入口 Hilbert code。
- `Ex[i]` 表示第 `i` 个连续片段的出口 Hilbert code。
- 每个 `(En[i], Ex[i])` 表示 Hilbert 曲线上一段完全位于查询窗口中的连续区间。

对应代码主要位于：

```text
window_query.hpp
window_query.cpp
```

核心函数为：

```cpp
WindowQueryProcessor::getEntrance(...)
WindowQueryProcessor::find_ones_in_range(...)
```

## 2. 核心思想

BRQ 利用一个关键几何事实：

```text
Hilbert 曲线进入或离开一个矩形查询窗口时，一定发生在查询窗口的边界上。
```

因此，不需要枚举窗口内部所有 cell，而只需要检查查询窗口的四条边：

```text
上边界 top
下边界 bottom
左边界 left
右边界 right
```

算法预先为 Hilbert 空间中的每个 cell 计算：

- 该 cell 的 Hilbert code。
- Hilbert 曲线进入该 cell 的方向 `entryDirection`。

然后把不同方向的入口信息组织成行/列倒排位图：

```text
rowInvertedIndices
columnInvertedIndices
```

再用 `sdsl::rrr_vector` 压缩，并建立 `rank` / `select` 支持结构，使得可以快速查询某条边界范围内哪些位置为 1。

## 3. 入口与出口的关系

入口和出口的处理方式不同。

### 3.1 入口 Entrance

入口点位于查询窗口自己的内部边界上。

例如查询窗口的上边界：

```text
y1 = Q.y + Q.height - 1
```

入口查询检查的是：

```text
rowRankSupport[2 * y1 + 1]
rowSelectSupport[2 * y1 + 1]
```

查询范围为：

```text
[Q.x, Q.x + Q.width - 1]
```

如果在该范围内找到 1，就说明 Hilbert 曲线从窗口外进入了窗口内部，对应 cell 的 Hilbert code 就是入口 code。

### 3.2 出口 Exit

出口点的判定需要向查询窗口外侧看一格。

例如查询窗口的上边界，窗口内部最后一行是：

```text
y = Q.y + Q.height - 1
```

窗口外侧上一行是：

```text
yy1 = Q.y + Q.height
```

代码会检查外侧行的入口位图：

```text
rowRankSupport[2 * yy1]
rowSelectSupport[2 * yy1]
```

如果外侧 cell 是从窗口内部进入的，那么窗口内部相邻 cell 就是出口点。

因此代码最后返回的 Hilbert code 仍然来自窗口内部边界：

```cpp
hilbertSpace[position][yy1 - 1].hilbertCode
```

这可以理解为：

```text
出口检测 = 查询框向外扩一圈，只检查外侧相邻 cell 的入口方向，再把窗口内部相邻 cell 作为出口。
```

## 4. 四条边界的入口/出口检查

设查询窗口为：

```text
Q = (x, y, width, height)
```

则内部边界为：

```text
left   = x
right  = x + width  - 1
bottom = y
top    = y + height - 1
```

外侧相邻边界为：

```text
leftOutside   = x - 1
rightOutside  = x + width
bottomOutside = y - 1
topOutside    = y + height
```

对应关系如下：

| 边界 | Entrance 查的位置 | Exit 查的外侧位置 | Exit 返回的内部位置 |
|---|---|---|---|
| 上边界 | `y + height - 1` | `y + height` | `y + height - 1` |
| 下边界 | `y` | `y - 1` | `y` |
| 右边界 | `x + width - 1` | `x + width` | `x + width - 1` |
| 左边界 | `x` | `x - 1` | `x` |

## 5. Grid Space 边界情况

如果查询窗口已经贴到了整个 Hilbert grid space 的边界，则外侧 cell 不存在。

因此代码中有如下判断：

```cpp
if ((queryWindow.y + queryWindow.height) != gridSize) { ... } // 上方存在外侧 cell
if (queryWindow.y != 0) { ... }                               // 下方存在外侧 cell
if ((queryWindow.x + queryWindow.width) != gridSize) { ... }  // 右侧存在外侧 cell
if (queryWindow.x != 0) { ... }                               // 左侧存在外侧 cell
```

这些判断避免访问越界，同时也处理了查询窗口位于 grid space 边界 cell 的情况。

还有一个特殊情况：

```cpp
if(queryWindow.x == 0 && queryWindow.y == 0) {
    getEntranceR.push_back(0);
}
```

如果查询窗口包含 Hilbert 曲线的全局起点 `(0, 0)`，那么 code `0` 没有前驱 cell，但它应该作为窗口内第一个连续片段的入口，因此需要手动加入。

## 6. 示例

假设 Hilbert grid space 大小为 `8 x 8`，查询窗口为：

```text
Q = (2, 2, 3, 3)
```

也就是：

```text
x = 2
y = 2
width = 3
height = 3
```

窗口内部范围为：

```text
x ∈ [2, 4]
y ∈ [2, 4]
```

内部四条边界为：

```text
left   = 2
right  = 4
bottom = 2
top    = 4
```

外侧相邻边界为：

```text
leftOutside   = 1
rightOutside  = 5
bottomOutside = 1
topOutside    = 5
```

### 6.1 网格示意图

下面图中：

- `Q` 表示查询窗口内部 cell。
- `*` 表示查询窗口边界 cell。
- `o` 表示出口检测时需要查看的外侧相邻 cell。
- `.` 表示其他 cell。

```text
y
7 | . . . . . . . .
6 | . . . . . . . .
5 | . o o o o o . .   ← topOutside = 5
4 | . o * * * o . .   ← top = 4
3 | . o * Q * o . .
2 | . o * * * o . .   ← bottom = 2
1 | . o o o o o . .   ← bottomOutside = 1
0 | . . . . . . . .
    ----------------
      0 1 2 3 4 5 6 7  x
        ↑ ↑   ↑ ↑
        | |   | |
        | |   | rightOutside = 5
        | |   right = 4
        | left = 2
        leftOutside = 1
```

### 6.2 入口检查

入口只看窗口自己的四条边界：

```text
上边界：y = 4, x ∈ [2, 4]
下边界：y = 2, x ∈ [2, 4]
右边界：x = 4, y ∈ [2, 4]
左边界：x = 2, y ∈ [2, 4]
```

图示：

```text
7 | . . . . . . . .
6 | . . . . . . . .
5 | . . . . . . . .
4 | . . E E E . . .
3 | . . E Q E . . .
2 | . . E E E . . .
1 | . . . . . . . .
0 | . . . . . . . .
    ----------------
      0 1 2 3 4 5 6 7
```

其中 `E` 表示入口候选位置。真正是否为入口，要看对应的 row/column 位图中该位置是否为 1。

### 6.3 出口检查

出口需要看查询窗口外侧相邻一圈：

```text
上侧外圈：y = 5, x ∈ [2, 4]
下侧外圈：y = 1, x ∈ [2, 4]
右侧外圈：x = 5, y ∈ [2, 4]
左侧外圈：x = 1, y ∈ [2, 4]
```

图示：

```text
7 | . . . . . . . .
6 | . . . . . . . .
5 | . . X X X . . .
4 | . X * * * X . .
3 | . X * Q * X . .
2 | . X * * * X . .
1 | . . X X X . . .
0 | . . . . . . . .
    ----------------
      0 1 2 3 4 5 6 7
```

其中 `X` 表示出口检测时查看的外侧相邻位置。

如果某个外侧位置的入口方向表明 Hilbert 曲线是从查询窗口内部进入外侧 cell 的，那么对应的内部边界 cell 就是出口点。

例如：

```text
外侧位置：    (3, 5)
内部出口 cell：(3, 4)
```

如果 `(3,5)` 在对应位图中为 1，则返回：

```cpp
hilbertSpace[3][4].hilbertCode
```

而不是返回 `hilbertSpace[3][5].hilbertCode`。

### 6.4 入口/出口配对

假设算法得到：

```text
En = [5, 18, 31]
Ex = [9, 24, 39]
```

排序后仍为：

```text
En = [5, 18, 31]
Ex = [9, 24, 39]
```

最终窗口在 Hilbert 曲线上的连续片段为：

```text
S = {(5, 9), (18, 24), (31, 39)}
```

含义是：

```text
Hilbert code ∈ [5, 9]   的点属于查询窗口
Hilbert code ∈ [18, 24] 的点属于查询窗口
Hilbert code ∈ [31, 39] 的点属于查询窗口
```

## 7. 与代码的对应关系

| PDF 伪代码 | 代码实现 |
|---|---|
| `getEntranceHC(...)` | `WindowQueryProcessor::getEntrance(...)` 中入口部分 |
| `getExitHC(...)` / `getExportHC(...)` | `WindowQueryProcessor::getEntrance(...)` 中出口部分 |
| `rang1Query(...)` | `WindowQueryProcessor::find_ones_in_range(...)` |
| `radixSort(En)` | `WindowQueryProcessor::radixSort(getEntranceR)` |
| `radixSort(Ex)` | `WindowQueryProcessor::radixSort(getExitR)` |
| 构造区间集合 `S` | 当前函数返回排序后的 `En` 和 `Ex`，区间配对可由调用方完成 |

## 8. 总结

BRQ 算法的本质是：

```text
预处理阶段：
1. 为每个 cell 计算 Hilbert code。
2. 为每个 cell 计算 Hilbert 曲线进入该 cell 的方向。
3. 按行/列和方向构建倒排位图。
4. 使用 sdsl::rrr_vector 压缩位图并建立 rank/select 支持。

查询阶段：
1. 在查询窗口四条边界上找入口。
2. 在查询窗口外侧相邻一圈上找出口。
3. 对入口和出口 Hilbert code 分别排序。
4. 将排序后的入口和出口按序配对，得到若干 Hilbert 连续区间。
```

该方法避免枚举窗口内部所有点，只检查窗口边界及外侧相邻边界，因此查询代价主要与窗口周长相关。

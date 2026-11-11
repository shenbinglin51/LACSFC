#include <iostream>
#include <vector>
#include <cmath>
#include <tuple>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <chrono>

// 定义象限类型
enum QuadrantType {
    A, B, C, D
};

// 定义段的结构
struct Segment {
    int mins, maxs;
};

// 定义查询框结构
struct QueryWindow {
    int x;
    int y;
    int width;
    int height;
};

class EnhancedHilbertProcessor {
private:
    int grid_size;
    std::vector<Segment> S;  // 存储段序列的数据结构

    // 全局的 determineType 函数
    int determineType(int x, int y, int p, int q, int T) {
        bool cond0 = (x < T / 2 && y < T / 2 && (x + q ) >= T / 2 && (y + p ) >= T / 2);
        bool cond1 = (x < T / 2 && y >= T / 2 && (x + q ) >= T / 2 && (y + p ) >= T / 2);
        bool cond2 = (x < T / 2 && y < T / 2 && (x + q ) >= T / 2 && (y + p ) < T / 2);
        bool cond3 = (x < T / 2 && y < T / 2 && (x + q ) < T / 2 && (y + p ) >= T / 2);
        bool cond4 = (x >= T / 2 && y < T / 2 && (x + q ) >= T / 2 && (y + p ) >= T / 2);
        bool cond5 = (x < T / 2 && y >= T / 2 && (x + q ) < T / 2 && (y + p ) >= T / 2);
        bool cond6 = (x >= T / 2 && y >= T / 2 && (x + q ) >= T / 2 && (y + p ) >= T / 2);
        bool cond7 = (x < T / 2 && y < T / 2 && (x + q ) < T / 2 && (y + p ) < T / 2);
        bool cond8 = (x >= T / 2 && y < T / 2 && (x + q ) >= T / 2 && (y + p ) < T / 2);

        if (cond0) return 0;
        else if (cond1)  return 1;
        else if (cond2)  return 2;
        else if (cond3)  return 3;
        else if (cond4)  return 4;
        else if (cond5)  return 5;
        else if (cond6)  return 6;
        else if (cond7)  return 7;
        else if (cond8)  return 8;
        
        return -1;
    }

    // 全局的 addSequence 函数
    void addSequence(int mins, int maxs) {
        if (S.empty()) {
            S.push_back({mins, maxs});
        } else {
            if (S.back().maxs + 1 == mins) {
                S.back().maxs = std::max(S.back().maxs, maxs);
            } else {
                S.push_back({mins, maxs});
            }
        }
    }

    // 根据类型细分象限的函数
    void QSplit(QuadrantType oABCD, int T, int mino, int x, int y, int p, int q);

    // 细分类型 A 的函数
    void QSplitA(int T, int mino, int x, int y, int p, int q);

    // 细分类型 B 的函数
    void QSplitB(int T, int mino, int x, int y, int p, int q);

    // 细分类型 C 的函数
    void QSplitC(int T, int mino, int x, int y, int p, int q);

    // 细分类型 D 的函数
    void QSplitD(int T, int mino, int x, int y, int p, int q);

public:
    EnhancedHilbertProcessor(int size) : grid_size(size) {}

    // 处理单个查询窗口
    std::vector<Segment> processQuery(int x, int y, int width, int height) {
        S.clear();
        QSplit(A, grid_size, 0, x, y, width, height);
        return S;
    }

    // 获取分段数量
    int getSegmentCount() const {
        return S.size();
    }

    // 清空结果
    void clear() {
        S.clear();
    }
};

// 实现 QSplit 函数，调用各个象限的细分函数
void EnhancedHilbertProcessor::QSplit(QuadrantType oABCD, int T, int mino, int x, int y, int p, int q) {
    if(T >= 1) {
        if (T == p && T == q) {
            addSequence(mino, mino + T * T - 1);
        } else {
            switch (oABCD) {
                case A:
                    QSplitA(T, mino, x, y, p, q);
                    break;
                case B:
                    QSplitB(T, mino, x, y, p, q);
                    break;
                case C:
                    QSplitC(T, mino, x, y, p, q);
                    break;
                case D:
                    QSplitD(T, mino, x, y, p, q);
                    break;
                default:
                    break;
            }
        }
    }
}

// 实现细分类型 A 的函数
void EnhancedHilbertProcessor::QSplitA(int T, int mino, int x, int y, int p, int q) {
    int N2 = T / 2;
    int N4 = N2 * N2;

    int type = determineType(x,y,p,q,T);

    switch (type) {
        case 0:
            QSplit(B, N2, mino, x, y, N2 - y, N2 - x);
            QSplit(A, N2, mino + N4, x, 0, y + p - N2, N2 - x);
            QSplit(A, N2, mino + N4 * 2, 0, 0, y + p - N2, x + q - N2);
            QSplit(D, N2, mino + N4 * 3, 0, y, N2 - y, x + q - N2);
            break;
        case 1:
            QSplit(A, N2, mino + N4, x, y - N2, p, N2 - x);
            QSplit(A, N2, mino + N4 * 2, 0, y - N2, p, x + q - N2);
            break;
        case 2:
            QSplit(B, N2, mino, x, y, p, N2 - x);
            QSplit(D, N2, mino + N4 * 3, 0, y, p, x + q - N2);
            break;
        case 3:
            QSplit(B, N2, mino, x, y, N2 - y, q);
            QSplit(A, N2, mino + N4, x, 0, y + p - N2, q);
            break;
        case 4:
            QSplit(A, N2, mino + N4 * 2, x - N2, 0, y + p - N2, q);
            QSplit(D, N2, mino + N4 * 3, x - N2, y, N2 - y, q);
            break;
        case 5:
            QSplit(A, N2, mino + N4, x, y - N2, p, q);
            break;
        case 6:
            QSplit(A, N2, mino + N4 * 2, x - N2, y - N2, p, q);
            break;
        case 7:
            QSplit(B, N2, mino, x, y, p, q);
            break;
        case 8:
            QSplit(D, N2, mino + N4 * 3, x - N2, y, p, q);
            break;
        default:
            break;
    }
}

// 实现细分类型 B 的函数
void EnhancedHilbertProcessor::QSplitB(int T, int mino, int x, int y, int p, int q) {
    int N2 = T / 2;
    int N4 = N2 * N2;

    int type = determineType(x,y,p,q,T);

    switch (type) {
        case 0:
            QSplit(A, N2, mino, x, y, N2 - y, N2 - x);
            QSplit(B, N2, mino + N4, 0, y, N2 - y, x + q - N2);
            QSplit(B, N2, mino + N4 * 2, 0, 0, y + p - N2, x + q - N2);
            QSplit(C, N2, mino + N4 * 3, x, 0, y + p - N2, N2 - x);
            break;
        case 1:
            QSplit(B, N2, mino + N4 * 2, 0, y - N2, p, x + q - N2);
            QSplit(C, N2, mino + N4 * 3, x, y - N2, p, N2 - x);
            break;
        case 2:
            QSplit(A, N2, mino, x, y, p, N2 - x);
            QSplit(B, N2, mino + N4 * 1, 0, y, p, x + q - N2);
            break;
        case 3:
            QSplit(A, N2, mino, x, y, N2 - y, q);
            QSplit(C, N2, mino + N4 * 3, x, 0, y + p - N2, q);
            break;
        case 4:
            QSplit(B, N2, mino + N4, x - N2, y, N2 - y, q);
            QSplit(B, N2, mino + N4 * 2, x - N2, 0, y + p - N2, q);
            break;
        case 5:
            QSplit(C, N2, mino + N4 * 3, x, y - N2, p, q);
            break;
        case 6:
            QSplit(B, N2, mino + N4 * 2, x - N2, y - N2, p, q);
            break;
        case 7:
            QSplit(A, N2, mino, x, y, p, q);
            break;
        case 8:
            QSplit(B, N2, mino + N4, x - N2, y, p, q);
            break;
        default:
            break;
    }
}

// 实现细分类型 C 的函数
void EnhancedHilbertProcessor::QSplitC(int T, int mino, int x, int y, int p, int q) {
    int N2 = T / 2;
    int N4 = N2 * N2;
  
    int type = determineType(x,y,p,q,T);

    switch (type) {
        case 0:
            QSplit(D, N2, mino, 0, 0, y + p - N2, x + q - N2);
            QSplit(C, N2, mino + N4, 0, y, N2 - y, x + q - N2);
            QSplit(C, N2, mino + N4 * 2, x, y, N2 - y, N2 - x);
            QSplit(B, N2, mino + N4 * 3, x, 0, y + p - N2, N2 - x);
            break;
        case 1:
            QSplit(D, N2, mino, 0, y - N2, p, x + q - N2);
            QSplit(B, N2, mino + N4 * 3, x, y - N2, p, N2 - x);
            break;
        case 2:
            QSplit(C, N2, mino + N4, 0, y, p, x + q - N2);
            QSplit(C, N2, mino + N4 * 2, x, y, p, N2 - x);
            break;
        case 3:
            QSplit(C, N2, mino + N4 * 2, x, y, N2 - y, q);
            QSplit(B, N2, mino + N4 * 3, x, 0, y + p - N2, q);
            break;
        case 4:
            QSplit(D, N2, mino, x - N2, 0, y + p - N2, q);
            QSplit(C, N2, mino + N4, x - N2, y, N2 - y, q);
            break;
        case 5:
            QSplit(B, N2, mino + N4 * 3, x, y - N2, p, q);
            break;
        case 6:
            QSplit(D, N2, mino, x - N2, y - N2, p, q);
            break;
        case 7:
            QSplit(C, N2, mino + N4 * 2, x, y, p, q);
            break;
        case 8:
            QSplit(C, N2, mino + N4, x - N2, y, p, q);
            break;
        default:
            break;
    }   
}

// 实现细分类型 D 的函数
void EnhancedHilbertProcessor::QSplitD(int T, int mino, int x, int y, int p, int q) {
    int N2 = T / 2;
    int N4 = N2 * N2;

    int type = determineType(x,y,p,q,T);

    switch (type) {
        case 0:
            QSplit(C, N2, mino, 0, 0, y + p - N2, x + q - N2);
            QSplit(D, N2, mino + N4, x, 0, y + p - N2, N2 - x);
            QSplit(D, N2, mino + N4 * 2, x, y, N2 - y, N2 - x);
            QSplit(A, N2, mino + N4 * 3, 0, y, N2 - y, x + q - N2);
            break;
        case 1:
            QSplit(C, N2, mino, 0, y - N2, p, x + q - N2);
            QSplit(D, N2, mino + N4, x, y - N2, p, N2 - x);
            break;
        case 2:
            QSplit(D, N2, mino + N4 * 2, x, y, p, N2 - x);
            QSplit(A, N2, mino + N4 * 3, 0, y, p, x + q - N2);
            break;
        case 3:
            QSplit(D, N2, mino + N4, x, 0, y + p - N2, q);
            QSplit(D, N2, mino + N4 * 2, x, y, N2 - y, q);
            break;
        case 4:
            QSplit(C, N2, mino, x - N2, 0, y + p - N2, q);
            QSplit(A, N2, mino + N4 * 3, x - N2, y, N2 - y, q);
            break;
        case 5:
            QSplit(D, N2, mino + N4, x, y - N2, p, q);
            break;
        case 6:
            QSplit(C, N2, mino, x - N2, y - N2, p, q);
            break;
        case 7:
            QSplit(D, N2, mino + N4 * 2, x, y, p, q);
            break;
        case 8:
            QSplit(A, N2, mino + N4 * 3, x - N2, y, p, q);
            break;
        default:
            break;
    }   
}

// C接口函数，供Python调用
extern "C" {
    EnhancedHilbertProcessor* create_processor(int grid_size) {
        return new EnhancedHilbertProcessor(grid_size);
    }

    void delete_processor(EnhancedHilbertProcessor* processor) {
        delete processor;
    }

    int process_query(EnhancedHilbertProcessor* processor, int x, int y, int width, int height) {
        processor->processQuery(x, y, width, height);
        return processor->getSegmentCount();
    }

    void clear_processor(EnhancedHilbertProcessor* processor) {
        processor->clear();
    }
}

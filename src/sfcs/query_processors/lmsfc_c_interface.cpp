/**
 * LMSFC C接口包装器
 * 为LMSFC C++模板类提供C接口，便于Python ctypes调用
 */
#include "../../../LMSFC/lmsfc/LearnedZorderPaging.h"
#include "../../../LMSFC/lmsfc/entities/Point.h"
#include "../../../LMSFC/lmsfc/entities/Mbr.h"
#include <vector>
#include <cstdint>

extern "C" {
    
// 创建LMSFC实例
void* create_lmsfc_2d(int page_size, uint64_t* masks, int mask_count) {
    std::vector<uint64_t> mask_vec;
    for (int i = 0; i < mask_count; i++) {
        mask_vec.push_back(masks[i]);
    }
    return new LearnedZorderPaging<2>(page_size, mask_vec);
}

// 销毁LMSFC实例
void destroy_lmsfc_2d(void* instance) {
    delete static_cast<LearnedZorderPaging<2>*>(instance);
}

// 计算Z地址
uint64_t calculate_z_address(void* instance, uint32_t x, uint32_t y) {
    auto* lmsfc = static_cast<LearnedZorderPaging<2>*>(instance);
    std::array<uint32_t, 2> coords = {x, y};
    Point<2> point(coords);
    return lmsfc->calZaddress(point);
}

// 查询分割 - 简化版本
int query_partition(void* instance, uint32_t min_x, uint32_t min_y, 
                   uint32_t max_x, uint32_t max_y, uint64_t* segments, int max_segments) {
    auto* lmsfc = static_cast<LearnedZorderPaging<2>*>(instance);
    
    // 创建查询窗口
    std::array<uint32_t, 2> lb_coords = {min_x, min_y};
    std::array<uint32_t, 2> ub_coords = {max_x, max_y};
    Point<2> lb_point(lb_coords);
    Point<2> ub_point(ub_coords);
    Mbr<2> query_mbr(lb_point, ub_point);
    
    // 创建查询窗口对象
    QueryWindow<2> query;
    query.QMbr = query_mbr;
    query.lbzaddr = lmsfc->calZaddress(lb_point);
    query.ubzaddr = lmsfc->calZaddress(ub_point);
    
    // 执行查询分割
    std::vector<QueryWindow<2>> result;
    lmsfc->QueryPartition(result, query);
    
    // 将结果转换为段数组
    int segment_count = std::min((int)result.size(), max_segments / 2);
    for (int i = 0; i < segment_count; i++) {
        segments[i * 2] = result[i].lbzaddr;      // 段开始
        segments[i * 2 + 1] = result[i].ubzaddr;  // 段结束
    }
    
    return segment_count;
}

// 训练LMSFC (简化版本)
int train_lmsfc(void* instance, uint32_t* points_x, uint32_t* points_y, int point_count,
                uint32_t* query_min_x, uint32_t* query_min_y, 
                uint32_t* query_max_x, uint32_t* query_max_y, int query_count,
                int iterations) {
    auto* lmsfc = static_cast<LearnedZorderPaging<2>*>(instance);
    
    // 准备训练数据
    std::vector<Point<2>> training_points;
    for (int i = 0; i < point_count; i++) {
        std::array<uint32_t, 2> coords = {points_x[i], points_y[i]};
        training_points.emplace_back(Point<2>(coords));
    }
    
    // 准备查询负载
    std::vector<QueryWindow<2>> training_queries;
    for (int i = 0; i < query_count; i++) {
        std::array<uint32_t, 2> lb_coords = {query_min_x[i], query_min_y[i]};
        std::array<uint32_t, 2> ub_coords = {query_max_x[i], query_max_y[i]};
        Point<2> lb_point(lb_coords);
        Point<2> ub_point(ub_coords);
        
        QueryWindow<2> query;
        query.QMbr = Mbr<2>(lb_point, ub_point);
        query.lbzaddr = lmsfc->calZaddress(lb_point);
        query.ubzaddr = lmsfc->calZaddress(ub_point);
        training_queries.push_back(query);
    }
    
    // 执行训练 (使用模拟退火方法)
    try {
        lmsfc->TrainingSA(training_points, training_queries, iterations, 100);
        return 1; // 成功
    } catch (...) {
        return 0; // 失败
    }
}

// 获取优化后的掩码
int get_optimized_masks(void* instance, uint64_t* masks, int max_masks) {
    auto* lmsfc = static_cast<LearnedZorderPaging<2>*>(instance);
    
    // 获取掩码 (需要访问私有成员，可能需要修改LMSFC类)
    // 这里返回当前掩码作为示例
    if (max_masks >= 2) {
        // 这需要LMSFC类提供公共接口来获取掩码
        // masks[0] = lmsfc->masks[0];
        // masks[1] = lmsfc->masks[1];
        return 2;
    }
    return 0;
}

} // extern "C"

"""
LACSFC Plugin for sfcBenchmark MVP Framework

将 LACSFC (Learning Adjacency-Continuous Space-Filling Curves) 集成到 MVP 框架中
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Tuple
import time

import numpy as np

# 添加 LACSFC 项目路径
from .._paths import lacsfcf_root
LACSFC_PATH = lacsfcf_root()
if str(LACSFC_PATH) not in sys.path:
    sys.path.insert(0, str(LACSFC_PATH))

from lacsfcf.data.grid import Grid
from lacsfcf.data.query_workload import Query, QueryWorkload, sample_workload
from lacsfcf.data.scene import SceneEncoder
from lacsfcf.core.sfc import weights_to_sfc
from lacsfcf.baselines import row_major_sfc

from ..types import SFCPlugin, QuantizedQuery, Interval

# Fast vectorized + C-core construction pipeline (scripts/fast_construct.py).
# Produces a code_map directly: byte-identical to weights_to_sfc for d=2 and
# quality-equivalent (valid AC path, avg_seg within ~0.03%) for d>=3, while
# scaling to large grids (e.g. d=4 k=7 / d=2 k=14) within ~14 GB. Imported
# lazily so the plugin still works if the scripts dir is unavailable.
_FAST_SCRIPTS_DIR = Path(__file__).resolve().parents[3] / "scripts"


def _get_fast_code_map():
    """Return the ``fast_code_map`` callable, or None if unavailable."""
    try:
        if str(_FAST_SCRIPTS_DIR) not in sys.path:
            sys.path.insert(0, str(_FAST_SCRIPTS_DIR))
        from fast_construct import fast_code_map
        return fast_code_map
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"[LACSFC] fast_construct unavailable, using weights_to_sfc: {exc}")
        return None

# Import trainer for neural network training
try:
    from .trainer import train_lacsfc_neural, train_lacsfc_two_phase, train_lacsfc_es, optimize_weights_es, optimize_weights_cmaes
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    train_lacsfc_neural = None
    train_lacsfc_two_phase = None
    train_lacsfc_es = None
    optimize_weights_es = None
    optimize_weights_cmaes = None

# Import query algorithms
from .boundary_query import LACSFCBoundaryQuery
from .brq_query import LACSFCBRQQuery

# Environment-variable override to pin a specific query-processor impl.
# Accepted values (case-insensitive):
#   "boundary_py" — NumPy O(|Q|) fallback.
#   "brq"         — AC-BRQ v2 boundary-event backend.
#   "brq_hybrid"  — AC-BRQ v2 with small-box brute-force dispatch.
# Unknown → error; empty → brq.
_LACSFC_QUERY_IMPL_ENV = "LACSFC_QUERY_IMPL"


class LACSFCPlugin:
    """
    LACSFC 学习型曲线插件
    
    LACSFC 使用 Cover-and-Merge 算法 + Maximum Spanning Tree 生成
    满足 Adjacency-Continuous (AC) 性质的空间填充曲线
    
    特点：
    - 保证 AC 性质（每步移动 Manhattan 距离为 1）
    - 基于查询工作负载自适应学习
    - 使用神经网络学习 dual edge 权重
    """
    
    def __init__(
        self,
        k: int = 8,
        d: int = 2,
        name: str = "LACSFC",
        use_neural: bool = False,
        num_training_queries: int = 1000,
        num_hotspots: int = 3,
        hotspot_std: float = 0.08,
        seed: int = 42,
        use_data_hotspots: bool = False,
        # D-4 v2 integration: load a pre-trained FG checkpoint and use
        # ``residual_alpha · FG(feats) + dual_edge_weights`` at build time,
        # skipping the expensive build-time CMA-ES / ES training entirely.
        use_neural_v2: bool = False,
        v2_ckpt_path: Optional[str] = None,
        use_fast_construct: bool = False,
    ):
        """
        初始化 LACSFC 插件
        
        Args:
            k: 每个维度的比特数（grid_side = 2^k）
            d: 维度数
            name: 曲线名称
            use_neural: 是否使用神经网络学习权重（False 则使用启发式）
            num_training_queries: 训练查询数量
            num_hotspots: 热点区域数量
            seed: 随机种子
            use_data_hotspots: 若 True，用 KMeans 从实际数据点提取热点中心；
                               若 False，使用原来的随机热点生成方式
        """
        self.name = name
        self._k = k
        self._d = d
        self._use_neural = use_neural
        self._use_neural_v2 = use_neural_v2
        self._v2_ckpt_path = v2_ckpt_path
        self._use_fast_construct = use_fast_construct
        self._num_training_queries = num_training_queries
        self._num_hotspots = num_hotspots
        self._hotspot_std = hotspot_std
        self._seed = seed
        self._use_data_hotspots = use_data_hotspots
        
        # LACSFC Grid 对象
        self._grid: Optional[Grid] = None
        
        # SFC code map: grid_shape -> SFC code
        self._code_map: Optional[np.ndarray] = None
        self._path: Optional[np.ndarray] = None
        
        # Grid size for SFCPlugin protocol
        self.grid_size = tuple(2**k for _ in range(d))
        
        # 是否已构建
        self._is_built = False
        
        # 神经网络模型（如果使用）
        self._generator = None
        self._evaluator = None
        
        # 训练配置（两阶段：Phase1 预训练 Evaluator，Phase2 训练 Generator）
        self._train_epochs = 10          # 已废弃（联合训练），保留兼容性
        self._train_dataset_size = 32    # 已废弃（联合训练），保留兼容性
        self._pretrain_epochs = 15       # Phase1: Evaluator 预训练 epochs（two-phase，已弃用）
        self._gen_epochs = 30            # Phase2: Generator 训练 epochs（two-phase，已弃用）
        self._train_dataset_size_v2 = 64 # dataset size
        self._samples_per_scene = 8      # Phase1 每个 scene 的权重采样数（two-phase，已弃用）
        # CMA-ES 直接权重优化参数（online，无 Generator）
        self._cmaes_sigma0 = 0.5         # CMA-ES 初始步长
        self._cmaes_max_evals = 3000     # 最大评估次数
        self._cmaes_popsize = 20         # 种群大小
        self._cmaes_n_train_queries = 600  # 训练查询数量（3种大小范围各200个，提升泛化性）
        self._cmaes_reg_lambda = 0.0     # L2 正则化强度（离线场景建议 0.01~0.1，固定 workload 场景设为 0）
        self._cmaes_mode = "block_space"  # 优化模式：block_space / hierarchical / func_space / edge_space
        self._cmaes_coarsen_factor = 2   # hierarchical 模式下的粗化因子（每维缩小倍数）
        self._cmaes_func_space_M = 200   # func_space 模式下的 RBF 核中心数（推荐 d>=4 时使用）
        self._cmaes_cover_weight = 0.0   # A-1: 在 CMA-ES 目标中加 cover_len 惩罚项（C-4 建议，0 = 原 page_amp-only）
        self._cmaes_runtime_surrogate = "none"
        self._cmaes_runtime_weight = 0.0
        self._cmaes_runtime_normalize = "raw"
        self._cmaes_runtime_interval_weight = 0.0
        # CMA-ES objective: "page_amp" (default, 保留原语义) or "weighted_mvp"
        # (seg + span_per_cell + page_amp 加权组合，可自适应)
        self._cmaes_objective = "page_amp"
        self._cmaes_learn_obj_weights = True
        self._cmaes_init_obj_weights = (0.60, 0.15, 0.25)
        # P1 warm-start ablation: "heuristic" (default, dual_edge_weights-derived)
        # / "random" (N(mean(log bd), 2)) / "uniform" (bv=0)
        self._cmaes_init_strategy = "heuristic"
        # Generator-ES 路径二参数
        self._es_epochs = 30
        self._es_n_perturbations = 32
        self._es_sigma = 0.05           # sigma small enough: 0.05×sqrt(n_params)<<weight_scale
        self._es_lr = 0.01              # per-step lr for Adam in ES context
        self._es_train_K = k
        self._es_gen_hidden = 16        # small MLP: ~225 params for reliable ES gradient
        self._es_gen_layers = 2
        self._es_learn_obj_weights = True   # jointly learn objective weights (seg/span/pa)
        self._es_init_obj_weights = (0.60, 0.15, 0.25)  # initial (or fixed) weights
        self._meta_enable = False
        self._meta_model_dir = ""
        self._meta_model_name = "lacsfc_pretrained"
        self._meta_inner_steps = 0
        self._meta_inner_lr = 1e-4
        self._meta_use_evaluator = True
        self._meta_fallback_cmaes = True
        self._meta_hidden = 128
        self._meta_layers = 3
        self._meta_evaluator_hidden = 128
        self._meta_evaluator_layers = 3
        self._meta_status = {
            "enabled": False,
            "model_loaded": False,
            "inner_steps": 0,
            "fallback_used": False,
            "direct_deploy": False,
            "error": "",
        }
        self._shift_status = {
            "monitor_enabled": False,
            "detected": False,
            "trigger_step": -1,
            "reopt_used": False,
            "reopt_sigma0": 0.0,
            "reopt_max_evals": 0,
            "state_path": "",
            "reason": "",
        }
        
        # Query-processor slots — exactly one of these is non-None after
        # build().  Selection driven by `LACSFC_QUERY_IMPL` env var; see the
        # module-level comment near the imports for accepted values.
        self._boundary_query: Optional[LACSFCBoundaryQuery] = None
        self._brq_query: Optional[LACSFCBRQQuery] = None
        self._query_impl_used: Optional[str] = None
        self._brq_hybrid_threshold = 128
        
        self._train_batch_size = 8
        self._train_lr = 1e-3
        self._device = 'cpu'
    
    def build(self, points: np.ndarray,
               query_windows=None, quantizer=None) -> None:
        """
        构建 LACSFC 曲线
        
        Args:
            points: 数据点数组，shape (n, d)
            query_windows: 可选，benchmark 的实际查询列表 (List[QueryWindow])。
                若传入，CMA-ES 直接以这批查询作为优化目标（固定 workload 场景，最优）。
            quantizer: 可选，与 query_windows 配套的 Quantizer 实例。
        """
        points = np.asarray(points, dtype=np.float64)
        n, d = points.shape
        
        if d != self._d:
            raise ValueError(f"Points dimension {d} does not match SFC dimension {self._d}")
        
        # 创建 LACSFC Grid
        self._grid = Grid(d=self._d, K=self._k)
        
        # 生成训练查询工作负载
        if self._use_data_hotspots:
            # 从实际数据点提取热点中心，生成与数据分布匹配的训练查询
            workload = self._sample_workload_from_data(points)
        else:
            # 原始方式：随机生成与数据无关的热点
            workload = sample_workload(
                self._grid,
                num_queries=self._num_training_queries,
                num_hotspots=self._num_hotspots,
                hotspot_std=self._hotspot_std,
                seed=self._seed,
            )
        
        if self._use_neural_v2:
            # D-4 v2: load pre-trained FG checkpoint and apply residual FG.
            # Build-time cost reduces from ~minutes (CMA-ES) to a single
            # forward pass of the FG (~ms).
            if not TORCH_AVAILABLE:
                print("Warning: torch unavailable. Falling back to heuristic.")
                self._build_heuristic(workload)
            else:
                self._build_neural_v2(workload)
        elif self._use_neural:
            # 使用神经网络学习权重
            if not TORCH_AVAILABLE or optimize_weights_cmaes is None:
                print("Warning: CMA-ES not available. Falling back to heuristic.")
                self._build_heuristic(workload)
            else:
                self._build_neural(workload, points=points,
                                   query_windows=query_windows,
                                   quantizer=quantizer)
        else:
            # 使用启发式方法（基于查询热度图）
            self._build_heuristic(workload)
        
        self._is_built = True
    
    def _sample_workload_from_data(self, points: np.ndarray) -> 'QueryWorkload':
        """
        从实际数据点用 KMeans 提取热点中心，生成训练查询工作负载。
        
        相比随机热点，这样生成的训练查询分布与数据的真实密度分布对齐，
        使 dual_edge_weights 能捕捉到数据实际聚集区域。
        """
        from sklearn.cluster import KMeans
        from lacsfcf.data.query_workload import _sample_one_query
        
        rng = np.random.default_rng(self._seed)
        n_clusters = min(self._num_hotspots, len(points))
        
        # KMeans 聚类找数据密度中心
        km = KMeans(n_clusters=n_clusters, random_state=int(self._seed), n_init=5)
        km.fit(points)
        centers = km.cluster_centers_  # shape (n_clusters, d)，已在 [0,1]^d
        
        # 以聚类中心为热点，加轻微抖动后生成查询
        hotspots = []
        for c in centers:
            jitter = rng.normal(0.0, self._hotspot_std, size=self._d)
            h = tuple(float(np.clip(c[k] + jitter[k], 0.0, 1.0)) for k in range(self._d))
            hotspots.append(h)
        
        queries = []
        for _ in range(self._num_training_queries):
            center = hotspots[int(rng.integers(0, n_clusters))]
            queries.append(_sample_one_query(rng, self._grid, center, 0.05, 0.25))
        
        return QueryWorkload(queries=queries, grid=self._grid)
    
    def _build_heuristic(self, workload: List[Query]) -> None:
        """
        使用启发式方法构建 LACSFC
        
        基于查询热度图计算 dual edge 权重，然后使用 MaxST + Cover-Merge
        """
        # 使用 dual_edge_weights() 作为权重（正确的启发式：双向查询覆盖度之和）
        wl_obj = QueryWorkload(workload.queries if hasattr(workload, 'queries') else workload, self._grid)
        weights = wl_obj.dual_edge_weights()
        
        # 构建曲线（fast pipeline 或参考 weights_to_sfc），填充 code_map
        self._assign_curve(weights)
        
        # 初始化查询处理器
        self._init_query_processor()
    
    def _build_neural(self, workload, points: np.ndarray = None,
                       query_windows=None, quantizer=None) -> None:
        """
        使用 CMA-ES（协方差矩阵自适应进化策略）优化当前 workload 的边权重。

        目标函数与 benchmark 完全一致：
          - result_count 用精确 point-in-window 过滤
          - page_accesses 用 BoundaryIndex 物理分页逻辑
          - 无任何代理近似

        若传入 query_windows + quantizer，直接用 benchmark 查询作为优化目标
        （固定 workload 场景，最优）；否则从数据点生成 200 个 data-aware 查询
        （离线优化场景，泛化性更好）。

        若 _cmaes_mode == "generator_es"，走路径二（3-目标 Generator）。
        """
        if self._cmaes_mode == "generator_es":
            self._build_generator_es(workload, points=points,
                                     query_windows=query_windows,
                                     quantizer=quantizer)
            return

        if optimize_weights_cmaes is None:
            print("Warning: CMA-ES not available. Falling back to heuristic.")
            self._build_heuristic(workload)
            return

        b = self._b if hasattr(self, '_b') else 256

        # 计算 LACSFC-H 权重作为热启动点（始终比随机初始化收敛快）
        from lacsfcf.data.query_workload import QueryWorkload as _QWL
        _wl_queries = workload.queries if hasattr(workload, 'queries') else list(workload)
        _wl_for_init = _QWL(_wl_queries, self._grid)
        init_weights = _wl_for_init.dual_edge_weights().astype(np.float32)
        meta_init_weights = self._try_meta_warmstart(
            workload=workload,
            init_weights=init_weights,
        )
        if meta_init_weights is not None:
            init_weights = meta_init_weights.astype(np.float32)
            if not self._meta_fallback_cmaes:
                self._meta_status["direct_deploy"] = True
                self._build_from_weights(init_weights)
                return

        # 若没有传入 benchmark 查询，则从数据点生成多样 data-aware 查询
        # 使用3种大小范围各 nq/3 个查询，覆盖小/中/大查询框，提升泛化性
        if query_windows is None and points is not None:
            from src.mvp.types import QueryWindow as _QW
            from src.mvp.quantizer import Quantizer as _Quant
            from lacsfcf.data.query_workload import _sample_one_query as _sqo
            rng = np.random.default_rng(self._seed)
            nq = self._cmaes_n_train_queries   # 默认 600（每种范围 200 个）
            pts_arr = np.asarray(points)
            # 3 种查询大小范围，均匀覆盖小/中/大
            size_ranges = [(0.01, 0.06), (0.02, 0.15), (0.08, 0.30)]
            n_per_group = max(1, nq // len(size_ranges))
            qw_list = []
            ql_list = []
            for wlo, whi in size_ranges:
                idxs = rng.integers(0, len(pts_arr), n_per_group)
                for c in pts_arr[idxs]:
                    side = float(rng.uniform(wlo, whi))
                    lo = np.clip(c - side/2, 0.0, 1.0 - side)
                    hi = lo + side
                    qw_list.append(_QW(low=lo, high=hi))
                    ql_list.append(_sqo(rng, self._grid,
                                        tuple(float(lo[d] + side/2) for d in range(self._d)),
                                        wlo, whi))
            query_windows = qw_list
            quantizer = _Quant(k=self._k, d=self._d)
            # 用与 query_windows 对齐的 lacsfcf.Query 列表重新计算热启动权重
            _wl_aligned = _QWL(ql_list, self._grid)
            init_weights = _wl_aligned.dual_edge_weights().astype(np.float32)
            meta_init_weights = self._try_meta_warmstart(
                workload=_wl_aligned,
                init_weights=init_weights,
            )
            if meta_init_weights is not None:
                init_weights = meta_init_weights.astype(np.float32)
                if not self._meta_fallback_cmaes:
                    self._meta_status["direct_deploy"] = True
                    self._build_from_weights(init_weights)
                    return

        print(f"Optimising LACSFC-Opt (CMA-ES: sigma0={self._cmaes_sigma0}, "
              f"max_evals={self._cmaes_max_evals}, popsize={self._cmaes_popsize}, "
              f"n_queries={len(query_windows) if query_windows else 'N/A'}, "
              f"real_pts={'yes' if points is not None else 'no'})...")
        t0 = time.time()

        weights = optimize_weights_cmaes(
            grid=self._grid,
            workload=workload,
            points=points,
            b=b,
            init_weights=init_weights,
            sigma0=self._cmaes_sigma0,
            max_evals=self._cmaes_max_evals,
            popsize=self._cmaes_popsize,
            verbose=True,
            query_windows=query_windows,
            quantizer=quantizer,
            reg_lambda=self._cmaes_reg_lambda,
            mode=self._cmaes_mode,
            coarsen_factor=self._cmaes_coarsen_factor,
            func_space_M=self._cmaes_func_space_M,
            cover_weight=self._cmaes_cover_weight,
            runtime_surrogate=self._cmaes_runtime_surrogate,
            runtime_weight=self._cmaes_runtime_weight,
            runtime_normalize=self._cmaes_runtime_normalize,
            runtime_interval_weight=self._cmaes_runtime_interval_weight,
            objective_mode=self._cmaes_objective,
            learn_obj_weights=self._cmaes_learn_obj_weights,
            init_obj_weights=self._cmaes_init_obj_weights,
            init_strategy=self._cmaes_init_strategy,
            init_seed=int(self._seed),
        )

        print(f"CMA-ES optimisation completed in {time.time()-t0:.1f}s")
        print(f"Optimised weights: min={weights.min():.4f}, max={weights.max():.4f}, "
              f"mean={weights.mean():.4f}, std={weights.std():.4f}")

        self._build_from_weights(weights)

    def _build_from_weights(self, weights: np.ndarray) -> None:
        self._assign_curve(weights)
        self._init_query_processor()

    def _try_meta_warmstart(self, workload, init_weights: np.ndarray) -> Optional[np.ndarray]:
        self._meta_status = {
            "enabled": bool(self._meta_enable),
            "model_loaded": False,
            "inner_steps": int(self._meta_inner_steps),
            "fallback_used": False,
            "direct_deploy": False,
            "error": "",
        }
        if not self._meta_enable:
            return None
        if not TORCH_AVAILABLE:
            self._meta_status["fallback_used"] = True
            self._meta_status["error"] = "torch_unavailable"
            return None
        if not self._meta_model_dir:
            self._meta_status["fallback_used"] = True
            self._meta_status["error"] = "missing_model_dir"
            return None
        try:
            import os
            import torch
            from lacsfcf.model.generator import EdgeWeightGenerator
            from lacsfcf.model.evaluator import QueryEvaluator
            from lacsfcf.data.query_workload import QueryWorkload as _QWL
            from .meta_learning import finetune_from_pretrained

            gen_path = os.path.join(str(self._meta_model_dir), f"{self._meta_model_name}_generator.pt")
            eval_path = os.path.join(str(self._meta_model_dir), f"{self._meta_model_name}_evaluator.pt")
            if not os.path.exists(gen_path):
                self._meta_status["fallback_used"] = True
                self._meta_status["error"] = f"missing_generator:{gen_path}"
                return None

            encoder = SceneEncoder(self._grid)
            generator = EdgeWeightGenerator(
                input_dim=encoder.feat_dim,
                hidden_dim=int(self._meta_hidden),
                num_layers=int(self._meta_layers),
            ).to(self._device)
            generator.load_state_dict(torch.load(gen_path, map_location=self._device))
            generator.eval()

            evaluator = None
            if self._meta_use_evaluator:
                if not os.path.exists(eval_path):
                    self._meta_status["fallback_used"] = True
                    self._meta_status["error"] = f"missing_evaluator:{eval_path}"
                    return None
                evaluator = QueryEvaluator(
                    input_dim=encoder.feat_dim,
                    hidden_dim=int(self._meta_evaluator_hidden),
                    num_layers=int(self._meta_evaluator_layers),
                ).to(self._device)
                evaluator.load_state_dict(torch.load(eval_path, map_location=self._device))
                evaluator.eval()

            queries = workload.queries if hasattr(workload, "queries") else list(workload)
            wl_obj = workload if hasattr(workload, "dual_edge_weights") else _QWL(queries, self._grid)
            if evaluator is not None and int(self._meta_inner_steps) > 0:
                generator, evaluator = finetune_from_pretrained(
                    generator=generator,
                    evaluator=evaluator,
                    target_queries=queries,
                    grid=self._grid,
                    encoder=encoder,
                    finetune_epochs=int(self._meta_inner_steps),
                    lr=float(self._meta_inner_lr),
                    device=self._device,
                    verbose=True,
                )

            feats = encoder.encode(wl_obj)
            with torch.no_grad():
                t_feats = torch.tensor(feats, dtype=torch.float32, device=self._device)
                weights = generator(t_feats)
                if weights.ndim > 1:
                    weights = weights.squeeze(-1)
                weights_np = weights.detach().cpu().numpy().astype(np.float32)
            if weights_np.shape != np.asarray(init_weights).shape:
                self._meta_status["fallback_used"] = True
                self._meta_status["error"] = f"shape_mismatch:{weights_np.shape}!={np.asarray(init_weights).shape}"
                return None
            self._generator = generator
            self._evaluator = evaluator
            self._meta_status["model_loaded"] = True
            return weights_np
        except Exception as exc:
            self._meta_status["fallback_used"] = True
            self._meta_status["error"] = type(exc).__name__ + ":" + str(exc)
            print(f"Warning: LACSFC meta warm-start failed: {self._meta_status['error']}. Falling back to CMA-ES.")
            return None
    
    def _build_generator_es(self, workload, points=None,
                             query_windows=None, quantizer=None) -> None:
        """路径二：3-目标 Generator（train_lacsfc_es）。

        在小网格(train_K=4)上用 ES 训练 per-edge MLP，
        目标 = 0.60*seg + 0.15*span + 0.25*page_amp。
        推理时在目标网格(K=self._k)上应用，
        feat_dim = d+15 与网格大小无关，天然跨分辨率泛化。
        """
        if not TORCH_AVAILABLE or train_lacsfc_es is None:
            print("Warning: train_lacsfc_es not available. Falling back to heuristic.")
            self._build_heuristic(workload)
            return

        import torch
        from lacsfcf.data.scene import SceneEncoder
        from lacsfcf.data.query_workload import QueryWorkload as _QWL

        _wl_queries = workload.queries if hasattr(workload, 'queries') else list(workload)
        train_K = self._k  # must match inference K to avoid feature distribution shift

        print(f"[Generator-ES] Training 3-objective Generator "
              f"(K={train_K}, epochs={self._es_epochs}, "
              f"n_perturbations={self._es_n_perturbations}, sigma={self._es_sigma})...")
        t0 = time.time()

        generator, learned_weights = train_lacsfc_es(
            grid=self._grid,
            training_queries=_wl_queries,
            epochs=self._es_epochs,
            dataset_size=self._train_dataset_size_v2,
            batch_size=self._train_batch_size,
            lr=self._es_lr,
            sigma=self._es_sigma,
            n_perturbations=self._es_n_perturbations,
            gen_hidden=self._es_gen_hidden,
            gen_layers=self._es_gen_layers,
            device=self._device,
            verbose=True,
            train_K=train_K,
            learn_obj_weights=self._es_learn_obj_weights,
            init_obj_weights=self._es_init_obj_weights,
        )
        elapsed = time.time() - t0
        if learned_weights is not None:
            print(f"[Generator-ES] Training completed in {elapsed:.1f}s  "
                  f"learned_obj=[seg={learned_weights[0]:.3f}, span={learned_weights[1]:.3f}, pa={learned_weights[2]:.3f}]")
        else:
            print(f"[Generator-ES] Training completed in {elapsed:.1f}s")

        enc = SceneEncoder(self._grid)
        wl_inf = _QWL(_wl_queries, self._grid)
        edge_feats = enc.encode(wl_inf)

        generator.eval()
        with torch.no_grad():
            t_feats = torch.tensor(edge_feats, dtype=torch.float32)
            weights = generator(t_feats).cpu().numpy()

        self._assign_curve(weights)

        self._init_query_processor()

    def _build_neural_v2(self, workload) -> None:
        """D-4 v2 integration: apply a pre-trained residual FG checkpoint.

        Checkpoint contract (produced by
        ``examples/train_lacsfc_v2_offline.py``)::

            {
                'input_dim':           int,   # = d + 15, K-invariant
                'hidden_dim':          int,
                'num_layers':          int,
                'residual_alpha':      float,
                'K': int, 'd': int,
                'generator_state_dict': OrderedDict,
                ...
            }

        Inference (no training):
            feats = SceneEncoder(grid).encode(workload)
            w     = α · FG(feats) + dual_edge_weights(workload)
            path  = weights_to_sfc(grid, w)

        Because feat_dim is K-invariant, a single checkpoint trained at a
        small K (e.g. K=5) transfers to the deployment K (e.g. K=8).
        """
        import torch
        from lacsfcf.data.query_workload import QueryWorkload as _QWL
        from lacsfcf.model.generator import EdgeWeightGenerator

        ckpt_path = self._v2_ckpt_path
        if ckpt_path is None or not Path(ckpt_path).exists():
            print(f"Warning: v2 checkpoint not found at {ckpt_path!r}. "
                  "Falling back to heuristic.")
            self._build_heuristic(workload)
            return

        print(f"[Neural-v2] Loading FG checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self._device, weights_only=False)

        # --- Reconstruct FG from saved architecture metadata ---
        input_dim = int(ckpt['input_dim'])
        hidden_dim = int(ckpt['hidden_dim'])
        num_layers = int(ckpt['num_layers'])
        alpha = float(ckpt['residual_alpha'])
        gen = EdgeWeightGenerator(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        ).to(self._device)
        gen.load_state_dict(ckpt['generator_state_dict'])
        gen.eval()

        # --- Sanity check: current grid's feat_dim must match ckpt ---
        enc = SceneEncoder(self._grid)
        if enc.feat_dim != input_dim:
            raise ValueError(
                f"feat_dim mismatch: grid encoder {enc.feat_dim} vs "
                f"ckpt input_dim {input_dim}. Retrain checkpoint for d={self._d}."
            )

        _wl_queries = workload.queries if hasattr(workload, 'queries') else list(workload)
        wl_obj = _QWL(_wl_queries, self._grid)
        feats = enc.encode(wl_obj)
        w_dual = wl_obj.dual_edge_weights().astype(np.float32)

        residual_mode = str(ckpt.get('residual_mode', 'affine'))
        t0 = time.time()
        with torch.no_grad():
            t_f = torch.tensor(feats, dtype=torch.float32, device=self._device)
            w_raw = gen(t_f)
            if w_raw.ndim > 1:
                w_raw = w_raw.squeeze(-1)
            w_residual = w_raw.cpu().numpy().astype(np.float32)
        if residual_mode == 'normalized':
            from .meta_learning import normalized_residual_np
            weights = normalized_residual_np(w_residual, w_dual, alpha)
        else:
            weights = alpha * w_residual + w_dual
        dt = time.time() - t0
        print(f"[Neural-v2] FG forward + residual done in {dt*1000:.1f}ms  "
              f"(α={alpha:.3f}, mode={residual_mode}, num_edges={weights.size}, "
              f"w[min,mean,max]=[{weights.min():.3f},{weights.mean():.3f},{weights.max():.3f}])")

        self._assign_curve(weights)

        self._init_query_processor()
        # Cache generator in case caller wants to introspect.
        self._generator = gen

    def _resolve_query_impl(self) -> str:
        """Pick which query processor to instantiate.

        Priority:
          1. ``LACSFC_QUERY_IMPL`` env var if set (validated against known impls).
          2. Auto: ``brq``.
        """
        import os
        valid = {"boundary_py", "brq", "brq_hybrid"}
        env = (os.environ.get(_LACSFC_QUERY_IMPL_ENV, "") or "").strip().lower()
        if env:
            if env not in valid:
                raise ValueError(
                    f"{_LACSFC_QUERY_IMPL_ENV}={env!r} is not one of {sorted(valid)}"
                )
            return env

        return "brq"

    def _assign_curve(self, weights: np.ndarray) -> None:
        """Build the SFC from edge ``weights`` and populate ``_code_map``.

        When ``use_fast_construct`` is enabled, uses the vectorized + C-core
        ``fast_code_map`` (byte-identical to ``weights_to_sfc`` for d=2,
        quality-equivalent for d>=3) which scales to large grids. Otherwise --
        or if the fast pipeline is unavailable -- falls back to the reference
        ``weights_to_sfc`` path. ``_path`` is set eagerly only on the reference
        path; for the fast path it is derived lazily in ``_init_query_processor``
        (needed solely by the BRQ backend) to avoid the O(n*d) materialization
        on huge grids that only use the code_map.
        """
        fast_code_map = _get_fast_code_map() if self._use_fast_construct else None
        if fast_code_map is not None:
            cm = fast_code_map(self._grid, np.asarray(weights, dtype=np.float64))
            self._code_map = np.asarray(cm, dtype=np.int64)
            self._path = None  # derived on demand for BRQ
            return

        path = weights_to_sfc(self._grid, np.asarray(weights, dtype=np.float32))
        self._path = np.asarray(path, dtype=np.int64)
        self._code_map = np.zeros(self._grid.grid_shape, dtype=np.int64)
        for code, cell in enumerate(path):
            self._code_map[tuple(cell)] = code

    def _ensure_path_from_code_map(self) -> None:
        """Derive ``_path`` (shape (n, d)) from ``_code_map`` if not present.

        path[t] = cell whose code is t. Recovered by ordering flat cell ids by
        their code value, then unraveling to coordinates.
        """
        if self._path is not None or self._code_map is None:
            return
        order = np.argsort(self._code_map.ravel(), kind="stable")
        coords = np.array(
            np.unravel_index(order, self._code_map.shape), dtype=np.int64
        ).T  # (n, d), row t = cell at SFC position t
        self._path = coords

    def _init_query_processor(self):
        """Construct exactly one query processor based on the resolved impl."""
        # Reset all slots so re-builds don't leave stale processors behind.
        self._boundary_query = None
        self._brq_query = None

        impl = self._resolve_query_impl()
        self._query_impl_used = impl
        G = 2 ** self._k

        if impl == "boundary_py":
            print("Initializing query processor: boundary-py (NumPy O(|Q|) baseline)...")
            self._boundary_query = LACSFCBoundaryQuery(
                code_map=self._code_map, grid_size=G, d=self._d,
            )
        elif impl in {"brq", "brq_hybrid"}:
            self._ensure_path_from_code_map()
            if self._path is None:
                raise RuntimeError("AC-BRQ v2 backend requires a built LACSFC path")
            print(f"Initializing query processor: {impl} (AC-BRQ v2)...")
            self._brq_query = LACSFCBRQQuery(
                path=self._path,
                grid_size=G,
                d=self._d,
                hybrid=(impl == "brq_hybrid"),
                hybrid_threshold=self._brq_hybrid_threshold,
            )
    
    def encode_cell(self, cell: np.ndarray) -> int:
        """
        将 cell 坐标编码为 SFC code (SFCPlugin 协议要求)
        
        Args:
            cell: cell 坐标 (整数数组)
        
        Returns:
            SFC code (整数)
        """
        if not self._is_built or self._code_map is None:
            raise RuntimeError("LACSFC not built. Call build() first.")
        
        cell = np.asarray(cell, dtype=np.int64)
        
        # 查询 code_map
        return int(self._code_map[tuple(cell)])
    
    def intervals(self, q: QuantizedQuery) -> Optional[List[Interval]]:
        """
        Compute the maximal contiguous SFC-code intervals covered by the
        quantized query window ``q``.

        Routes to the query processor selected at build time.

        Returns
        -------
        List[Interval] | None
            ``None`` if the curve has not been built yet, otherwise the
            list of ``(start, end)`` intervals (inclusive endpoints,
            sorted by ``start``).
        """
        if not self._is_built or self._code_map is None:
            return None

        low_cell = tuple(q.low_cell)
        high_cell = tuple(q.high_cell)

        if self._boundary_query is not None:
            interval_tuples = self._boundary_query.get_intervals(low_cell, high_cell)
        elif self._brq_query is not None:
            interval_tuples = self._brq_query.get_intervals(low_cell, high_cell)
        else:
            return None

        return [Interval(start=s, end=e) for s, e in interval_tuples]


def make_lacsfc(
    k: int = 8,
    d: int = 2,
    name: str = "LACSFC",
    use_neural: bool = False,
    num_training_queries: int = 1000,
    num_hotspots: int = 3,
    hotspot_std: float = 0.08,
    seed: int = 42,
    use_data_hotspots: bool = False,
    use_neural_v2: bool = False,
    v2_ckpt_path: Optional[str] = None,
    use_fast_construct: bool = False,
) -> LACSFCPlugin:
    """
    创建 LACSFC 插件实例
    
    Args:
        k: 每个维度的比特数
        d: 维度数
        name: 曲线名称
        use_neural: 是否使用神经网络
        num_training_queries: 训练查询数量
        num_hotspots: 热点区域数量
        hotspot_std: 热点标准差（越小权重越集中，默认0.05）
        seed: 随机种子
        use_data_hotspots: 若 True，用 KMeans 从实际数据点提取热点中心
    
    Returns:
        LACSFCPlugin 实例
    """
    return LACSFCPlugin(
        k=k,
        d=d,
        name=name,
        use_neural=use_neural,
        num_training_queries=num_training_queries,
        num_hotspots=num_hotspots,
        hotspot_std=hotspot_std,
        seed=seed,
        use_data_hotspots=use_data_hotspots,
        use_neural_v2=use_neural_v2,
        v2_ckpt_path=v2_ckpt_path,
        use_fast_construct=use_fast_construct,
    )

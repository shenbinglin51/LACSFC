from .paper_repro import AdaCurvePaperReproPlugin, make_adacurve_paper_repro

# The alternative engines below wrap the original AdaCurve project (ADACURVE_ROOT);
# they are not used by the benchmark, which goes through paper_repro only.
try:
    from .native_engine import AdaCurveNativeEngine, make_adacurve_native
    from .disk_simulation_engine import AdaCurveDiskSimulationEngine, make_adacurve_disk_sim
    from .adapter import convert_to_adacurve_dataset, convert_to_adacurve_queries
except ImportError:  # external AdaCurve project not installed
    AdaCurveNativeEngine = None
    AdaCurveDiskSimulationEngine = None
    make_adacurve_native = None
    make_adacurve_disk_sim = None
    convert_to_adacurve_dataset = None
    convert_to_adacurve_queries = None

AdaCurvePlugin = AdaCurvePaperReproPlugin
make_adacurve = make_adacurve_paper_repro

__all__ = [
    "AdaCurvePlugin",
    "AdaCurvePaperReproPlugin",
    "AdaCurveNativeEngine",
    "AdaCurveDiskSimulationEngine",
    "make_adacurve",
    "make_adacurve_paper_repro",
    "make_adacurve_native",
    "make_adacurve_disk_sim",
    "convert_to_adacurve_dataset",
    "convert_to_adacurve_queries",
]

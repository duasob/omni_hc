from . import ch5, darcy_report, ns_report, pipe_report, plasticity_report

ARTIFACTS = [
    *ch5.ARTIFACTS,
    *darcy_report.ARTIFACTS,
    *ns_report.ARTIFACTS,
    *pipe_report.ARTIFACTS,
    *plasticity_report.ARTIFACTS,
]

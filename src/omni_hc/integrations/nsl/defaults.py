from typing import Any

# ── NSL CLI plumbing ─────────────────────────────────────────────────────────
# Static values that satisfy NSL's argument namespace but are never configured
# through omni_hc. These are overridden by nothing and should never need to
# change.
_NSL_PLUMBING: dict[str, Any] = {
    "gpu": "0",
    "eval": 0,
    "save_name": "omni_hc_checkpoint",
    "vis_num": 0,
    "vis_bound": None,
    "data_path": "/data/fno/",
    "train_ratio": 0.8,
    "ntrain": 1000,
    "ntest": 200,
    "downsamplex": 1,
    "downsampley": 1,
    "downsamplez": 1,
    "radius": 0.2,
    # NSL declares these training hyperparams on its arg namespace; our system
    # reads training config from the training: YAML section instead.
    "lr": 1e-3,
    "epochs": 500,
    "weight_decay": 1e-5,
    "pct_start": 0.3,
    "batch_size": 8,
    "optimizer": "AdamW",
    "scheduler": "OneCycleLR",
    "step_size": 100,
    "gamma": 0.5,
    "max_grad_norm": None,
}

# ── Cross-backbone optional model defaults ───────────────────────────────────
# These are meaningful model params, but backbone-specific: a backbone that
# doesn't use a param simply ignores it. Backbone YAMLs should override these
# for any param they actually care about.
#
# If a param here affects a backbone you're using and is NOT in that backbone's
# YAML, add it to the YAML — don't silently rely on this default.
_MODEL_OPTIONAL_DEFAULTS: dict[str, Any] = {
    "dropout": 0.0,
    "act": "gelu",
    "mlp_ratio": 1,
    "slice_num": 32,
    "T_in": 10,         # time steps in; add explicitly to backbone YAMLs for time-series models
    "T_out": 10,
    "teacher_forcing": 1,
    "time_input": False,
    "psi_dim": 8,
    "attn_type": "nystrom",
    "mwt_k": 3,
    "shapelist": None,
}


def get_nsl_default_args() -> dict[str, Any]:
    return {**_NSL_PLUMBING, **_MODEL_OPTIONAL_DEFAULTS}

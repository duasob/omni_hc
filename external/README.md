# External Backends

Put third-party backend checkouts here.

The preferred location for Neural-Solver-Library is:

```text
external/Neural-Solver-Library
```

`omni_hc` will look for NSL in this folder by default. It also supports:

- `backend.nsl_root` in the experiment config
- `OMNI_HC_NSL_ROOT` environment variable
- `--nsl-root` on CLI scripts

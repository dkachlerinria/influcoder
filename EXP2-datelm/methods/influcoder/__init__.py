"""DATE-LM <-> InfluCoder glue.

Everything in this subpackage is new code local to this benchmark
integration (EXP2-datelm). It does not modify the parent `influcoder` repo's
own package -- it only imports its (model-agnostic) library code:
`influcoder.gradients.CountSketch`/`hardware_profile` and
`influcoder.encoder.load_encoder`/`distill`/`embed`. See
`teacher_grads.py`'s docstring for how that package is located on sys.path.
"""

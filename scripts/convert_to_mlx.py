"""P4: flownet.pkl -> dist/<hf_name>/{model.safetensors, config.json} (fp16/fp32).

Strip `module.` prefix, transpose conv / conv-transpose weights, mx.eval all,
save. config.json carries the pinned arch params. See utils/convert.py.
"""

raise NotImplementedError("P4")

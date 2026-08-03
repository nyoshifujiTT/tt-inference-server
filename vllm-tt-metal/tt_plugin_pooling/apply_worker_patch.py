import sys
p = sys.argv[1]  # path to worker.py in the installed plugin
s = open(p).read()
if "TTPoolingModelRunner" in s:
    print("worker.py already patched; skipping"); sys.exit(0)
imp = "from vllm_tt_plugin.model_runner import TTModelRunner"
assert imp in s, "TTModelRunner import not found"
s = s.replace(imp, imp + "\nfrom vllm_tt_plugin.pooling_runner import TTPoolingModelRunner", 1)
old_ctor = (
'        self.model_runner: TTModelRunner = TTModelRunner(\n'
'            vllm_config=self.vllm_config,\n'
'            mesh_device=self.mesh_device,\n'
'            trace_mode=self.trace_mode,\n'
'            enable_model_warmup=self.enable_model_warmup,\n'
'            num_devices=self.num_devices,\n'
'        )'
)
assert old_ctor in s, "runner ctor not found"
new_ctor = (
'        runner_type = getattr(self.model_config, "runner_type", "generate")\n'
'        runner_cls = (\n'
'            TTPoolingModelRunner if runner_type == "pooling" else TTModelRunner\n'
'        )\n'
'        self.model_runner = runner_cls(\n'
'            vllm_config=self.vllm_config,\n'
'            mesh_device=self.mesh_device,\n'
'            trace_mode=self.trace_mode,\n'
'            enable_model_warmup=self.enable_model_warmup,\n'
'            num_devices=self.num_devices,\n'
'        )'
)
s = s.replace(old_ctor, new_ctor, 1)
old_kv = (
'        spec_from_hook = self._try_get_spec_from_model_hook()\n'
'        if spec_from_hook is not None:\n'
'            return spec_from_hook\n\n'
'        return self._build_default_kv_cache_spec()'
)
assert old_kv in s, "get_kv_cache_spec body not found"
new_kv = (
'        if getattr(self.model_config, "runner_type", "generate") == "pooling":\n'
'            return {}\n\n'
'        spec_from_hook = self._try_get_spec_from_model_hook()\n'
'        if spec_from_hook is not None:\n'
'            return spec_from_hook\n\n'
'        return self._build_default_kv_cache_spec()'
)
s = s.replace(old_kv, new_kv, 1)
old_init = (
'        _validate_tt_kv_cache_capacity(self.vllm_config, kv_cache_config)\n'
'        self.model_runner.initialize_kv_cache(kv_cache_config)'
)
assert old_init in s, "initialize_from_config body not found"
new_init = (
'        if getattr(self.model_config, "runner_type", "generate") == "pooling":\n'
'            return\n'
'        _validate_tt_kv_cache_capacity(self.vllm_config, kv_cache_config)\n'
'        self.model_runner.initialize_kv_cache(kv_cache_config)'
)
s = s.replace(old_init, new_init, 1)
open(p, "w").write(s)
print("worker.py patched for pooling")

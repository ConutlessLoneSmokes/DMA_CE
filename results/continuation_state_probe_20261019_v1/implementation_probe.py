"""One optimizer update from one archived server model; implementation verification only."""
import json
import os
from pathlib import Path
import shutil
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path("/home/lsy/DMA_CE")
sys.path.insert(0, str(ROOT / "src"))
import torch
from dma_ce.continuation import verify_lineage
from dma_ce.experiment import file_hash, write_json
from dma_ce.research import recipe
from dma_ce.suite import Study

torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
torch.use_deterministic_algorithms(True)
started = time.time()
parent = ROOT / "runs/research_001"
parent_export = ROOT / "results/research_001"
before = {p: file_hash(p) for folder in (parent, parent_export) for p in folder.rglob("*") if p.is_file()}
study = Study(ROOT, None, "continuation_state_probe_20261019_v1", continue_from="research_001",
    diffusion_checkpoints=[20001])
lineage = verify_lineage(study.path, study.export)
assert len(lineage["origin_states"]) == 6
chosen = study.result("budget")
cache = study.get_cache(chosen["cache_id"])
spec = recipe(study.config, chosen["architecture"], chosen["loss"], "none", "diffusion")
seed = study.config["research"]["training_seeds"][0]
origin = next(item for item in lineage["origin_states"] if item["recipe"] == spec and item["seed"] == seed)
old = torch.load(study.path / origin["snapshot_path"], map_location="cpu", weights_only=True)
assert old["step"] == 20000
assert all(float(value["step"]) == 20000 for value in old["optimizer"]["state"].values())
checkpoint = study.train(cache, spec, seed, 20001)
new = torch.load(checkpoint.parent / "last.pt", map_location="cpu", weights_only=True)
assert new["step"] == 20001
assert new["optimizer"]["param_groups"][0]["lr"] == .0001
assert not torch.equal(old["batch_rng"], new["batch_rng"])
assert not torch.equal(old["noise_rng"], new["noise_rng"])
maximum = max(float((new["state_dict"][k] - value).abs().max()) for k, value in old["state_dict"].items())
assert maximum > 0 and all(torch.isfinite(value).all() for value in new["state_dict"].values())
assert all(float(value["step"]) == 20001 for value in new["optimizer"]["state"].values())
verify_lineage(study.path, study.export)
assert all(file_hash(path) == expected for path, expected in before.items())
Study(ROOT, None, study.run_id, continue_from="research_001", diffusion_checkpoints=[20001])
report = {"status": "passed", "implementation_probe_only": True, "parent_run_id": "research_001",
    "inherited_diffusion_states": len(lineage["origin_states"]), "continued_jobs": 1, "optimizer_updates": 1,
    "original_step": old["step"], "continued_step": new["step"], "extension_learning_rate": .0001,
    "maximum_parameter_change": maximum, "optimizer_and_rng_restored_and_advanced": True,
    "parent_files_verified_unchanged": len(before), "inherited_files_and_log_prefixes_verified": True,
    "source_zip_sha256": file_hash(study.path / "source.zip"), "checkpoint_sha256": file_hash(checkpoint),
    "seconds": time.time() - started, "screening_used": False}
write_json(study.export / "implementation_probe.json", report)
write_json(study.export / "parent_integrity.json", {str(p.relative_to(ROOT)): value for p, value in before.items()})
shutil.copy2(__file__, study.export / "implementation_probe.py")
(study.export / "report.md").write_text(
    "# 真实服务器断点的单步续训检查\n\n"
    "这是实现验证：从 research_001 复制全部六个扩散断点，仅选择一个模型恢复优化器和随机状态，"
    "从20,000步执行到20,001步。末端学习率0.0001。\n\n"
    f"检查通过；原运行的{len(before)}个文件哈希保持不变，继承模型、缓存、前五阶段与日志前缀核对通过。"
    "本记录未运行完整扩散选择阶段和最终筛选，不能作为方法性能结果。\n", encoding="utf-8")
print(json.dumps(report, indent=2))

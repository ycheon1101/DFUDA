# DFUDA Ablation Configs (Paper Table 3)

Standalone configs under this folder. For batch runs use `experiments.py`:

| exp id | UDA method | Networks | Components |
|--------|------------|----------|------------|
| **201** | DACS | DeepLabV2 + DAFormer | baseline / sam / dino / full |
| **202** | DAFormer | DeepLabV2 + DAFormer | baseline / sam / dino / full |
| **203** | HRDA | DeepLabV2 + DAFormer | baseline / sam / dino / full |
| **204** | MIC | DeepLabV2 + DAFormer | baseline / sam / dino / full |

**204 MIC** additionally runs `daformer_mic` + `full` on GTA+SYNTHIA with seeds 0/1/2 (paper main result).

```bash
python run_experiments.py --exp 201   # DACS (8 runs)
python run_experiments.py --exp 204   # MIC (10 runs: 8 + 2 extra seeds x2 datasets for full)
```

Regenerate standalone configs:

```bash
python tools/generate_dfuda_ablation_configs.py
```

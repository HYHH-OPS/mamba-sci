param(
  [switch]$Execute,
  [string]$Python = "/root/miniconda3/envs/mamba_a800/bin/python",
  [string]$ProjectRoot = "/autodl-tmp/mamba-sci",
  [string]$Csv = "/autodl-tmp/caption_train_private_v5_ord_all.csv",
  [string]$MambaModel = "/autodl-tmp/models/mamba-2.8b-hf"
)

$common = @(
  "--csv", $Csv,
  "--epochs", "30",
  "--batch_size", "8",
  "--num_workers", "0",
  "--lr", "1e-5",
  "--max_visual_tokens", "164",
  "--save_every_steps", "100",
  "--log_every_steps", "1",
  "--mamba_model", $MambaModel,
  "--lambda_cls", "1.0",
  "--use_lora",
  "--cls_focal_gamma", "2.0",
  "--bf16", "True"
)

$cmds = @(
  @("--ablation_mode", "full",        "--output_dir", "/autodl-tmp/mamba-sci/outputs/ablation_full") + $common,
  @("--ablation_mode", "global_only", "--output_dir", "/autodl-tmp/mamba-sci/outputs/ablation_global_only") + $common,
  @("--ablation_mode", "local_only",  "--output_dir", "/autodl-tmp/mamba-sci/outputs/ablation_local_only") + $common
)

Set-Location $ProjectRoot
foreach ($args in $cmds) {
  $line = "$Python train_vlm.py " + ($args -join " ")
  Write-Host $line
  if ($Execute) { & $Python train_vlm.py @args }
}

if (-not $Execute) {
  Write-Host "Dry-run only. Add -Execute to actually run ablations."
}

# 将服务器代码推送到 GitHub 的说明

## 当前状态

- **本地 main 分支**：已将所有服务器上的修改与新增代码**提交**到本仓库的 main 分支（commit: Sync server: ablation/Transformer bridge, LIDC/ReX scripts, eval bundle, paper docs, tests）。
- **远程**：`origin` 已指向 `https://github.com/HYHH-OPS/mamba-sci.git`。
- **未完成**：`git push origin main` 因未配置 GitHub 认证而失败，需要你在本机或服务器上配置认证后再执行推送。

## 本次已提交内容概览

- **.gitignore**：增加对 checkpoints、wandb、ablation 产物、备份文件的忽略。
- **bridge**：`transformer_bridge.py` 新增；`__init__.py`、`vim_bridge.py` 修改。
- **config**：`paths.yaml` 更新。
- **data**：`medical_vlm_dataset.py` 修改。
- **docs**：新增 SCI_VAL_KITS19.md、论文 LaTeX 片段与图表插入说明、figures 下示例图。
- **inference / train / train_vlm**：WandB、ablation、参数与流程更新。
- **llm / model / vision**：对应修改。
- **scripts**：新增/更新：align_private_ct_mask_pairs、build_private_caption_from_reports、clean_private_caption_csv、lidc_idri_preprocess、pack_ablation_results.sh、plot_stage2_train_loss、prepare_public_ablation_data、run_neurips_*、run_private_eval_bundle、run_public_ablation.sh、run_stage2_background、watch_neurips。
- **tests**：`verify_3d_pipeline.py` 新增。
- **根目录**：RUN_COMMANDS.md、run_ablations.ps1、plot_fig2.py 等。

## 你需要在有 Git 与网络的环境下执行

在**本仓库目录**下执行（任选一种认证方式）：

### 方式一：HTTPS + Personal Access Token（推荐）

1. 在 GitHub：Settings → Developer settings → Personal access tokens 中生成一个有 `repo` 权限的 token。
2. 在服务器或本机本仓库目录执行：
   ```bash
   cd /root/autodl-tmp/mamba-sci   # 或你的 mamba-sci 路径
   git push origin main
   ```
3. 若提示输入密码，**用户名**填你的 GitHub 用户名，**密码**处粘贴上述 token。

### 方式二：改用 SSH 推送

1. 若已配置 SSH 公钥并添加到 GitHub，可将远程改为 SSH 再推送：
   ```bash
   git remote set-url origin git@github.com:HYHH-OPS/mamba-sci.git
   git push origin main
   ```

### 方式三：在本地 Windows 上拉取服务器代码后再推送

1. 将服务器上本仓库打包（含 `.git`）拷到本机，或在本机 `git clone` 后把服务器上改动的文件覆盖过去并 `git add` / `git commit`。
2. 在本机（已登录 GitHub 或已配置 credential）执行：
   ```bash
   git push origin main
   ```

推送成功后，GitHub 上的 [HYHH-OPS/mamba-sci](https://github.com/HYHH-OPS/mamba-sci) 的 main 分支即与服务器代码一致；未创建其他分支，仅更新 main。

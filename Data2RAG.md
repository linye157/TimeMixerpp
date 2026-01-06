不要新起一个独立工程，不要重写整个项目。目标是在当前代码基础上“增量实现”：
1) 训练序列级 embedding（SupCon，可选 BCE 联合）
2) 生成三尺度 embedding 写入 Qdrant 三个 collection
3) 提供查询脚本：三库检索 topK → 尺度内概率 → 尺度间融合 → 输出解释

【已知现状】
- 我已有 NPZ 特征文件：features/alldata_features_no_tid.npz
- NPZ keys: scale_0 (N,48,64), scale_1 (N,24,64), scale_2 (N,12,64), labels (N,)
- 我已经有 scripts/extract_features.py 可读取 npz 并打印形状（说明 numpy/日志环境已OK）
- 目标向量库：Qdrant（本地 http://localhost:6333）
- 我希望做三个知识库（scale0/1/2 各一个），并可解释融合

【你需要做什么（按顺序执行，给出最终代码 diff/文件内容，并确保能运行，最后将改动写入README.md）】

(0) 先扫描仓库结构：
- 找到已有的模型/训练脚本组织方式（例如已有的 train.py、models/、utils/ 等）。
- 复用现有日志、argparse、torch 训练循环风格。
- 不要引入过多新依赖；最多新增 qdrant-client 和 tqdm（如果仓库已用则复用）。

(1) 新增“序列级 embedding encoder”
- 新增文件（优先放在现有的模型目录，例如 models/ 或 src/ 下，与现有风格一致）：
  - models/metric_encoder.py（或你在仓库里找到的合适位置）
- 实现类 TemporalConvEmbedder：
  - forward(x): x shape (B,L,64) -> embedding (B,emb_dim)
  - 结构要求：
    - x 转为 (B,64,L) 做 Conv1d
    - 3 层 Conv1d(kernel=3,pad=1), hidden=128, GELU, Dropout=0.1
    - attention pooling（必须）：输出 alpha (B,L) 与 pooled (B,hidden)
    - projection MLP：hidden->emb_dim（默认 128）
    - L2 normalize 输出 embedding
  - 可选：分类头 Linear(emb_dim,1)（用 --use_bce 开关控制）
- 注意：不能用简单 mean pooling 作为唯一 embedding（会丢时序）。

(2) 新增 SupCon loss
- 新增文件：losses/supcon.py（或符合你仓库结构的位置）
- 实现 SupConLoss(temperature=0.07)：
  - 输入 embeddings (B,d) 已归一化，labels (B,)
  - 正对：同 label
  - 边界：某样本在 batch 内没有正样本时跳过或安全处理（避免 NaN）
- 若 --use_bce，联合 BCEWithLogitsLoss，loss = supcon + lambda_bce * bce

(3) 新增 Dataset 与 split（尽量复用现有 Dataset/loader 结构）
- 新增/复用 data 模块，实现 NPZMultiScaleDataset：
  - __getitem__ 返回 (x0,x1,x2,label,index)
  - x0 float32 (48,64), x1 (24,64), x2 (12,64)
- 支持生成 splits.json（train_ids/val_ids/test_ids），并支持读取复用
- 支持 balanced sampling（可选）：按 labels 用 WeightedRandomSampler 保证 batch 内正样本比例

(4) 新增训练入口脚本（只增量）
- 新增脚本：scripts/train_embedding.py
- 功能：
  - args: --npz_path, --out_dir, --emb_dim, --epochs, --batch_size, --lr, --tau, --use_bce, --lambda_bce, --balanced_sampling, --seed, --split_ratio, --splits_path(可选)
  - 训练时对每个 batch：
    - 同一 encoder 分别跑三尺度：e0=enc(x0), e1=enc(x1), e2=enc(x2)
    - loss 采用三尺度加权和（默认 w0=0.5,w1=0.3,w2=0.2，可配）：
      loss = w0*SupCon(e0)+w1*SupCon(e1)+w2*SupCon(e2) + (BCE 若启用：对每尺度或对融合 embedding 计算都可以；请选择实现最简单且稳定的方式，并写清楚)
  - 保存 checkpoint（包含 encoder state_dict、融合权重/配置、训练配置）
  - 输出 metrics.json（至少包含训练 loss、val AUROC/F1/Acc；阈值默认0.5即可，若仓库已有阈值选择逻辑则复用）

(5) 新增 Qdrant 三库入库脚本（增量）
- 新增工具模块：utils/qdrant_utils.py
  - create_or_validate_collection(client, name, vector_size, distance="Cosine")
  - upsert_points(client, name, ids, vectors, payloads, batch_size)
- 新增脚本：scripts/ingest_to_qdrant_3scales.py
  - args: --npz_path, --ckpt_path, --splits_path, --split(train/val/test/all), --qdrant_url, --collection_prefix, --batch_size, --id_offset
  - 只对指定 split 的 ids：
    - 计算 e0/e1/e2（torch.no_grad, 批量）
    - 分别 upsert 到：
      {prefix}_scale0, {prefix}_scale1, {prefix}_scale2
  - payload 至少包含：label、sample_id、scale、embed_dim、ckpt_path basename、可选：attn_top_timesteps（每尺度取 alpha 最大的 top3 idx）

(6) 新增查询与融合脚本（增量）
- 新增脚本：scripts/query_rag_3scales.py
  - args: --npz_path, --ckpt_path, --qdrant_url, --collection_prefix, --query_index, --top_k, --gamma, --fusion_mode(fixed/learned), --w0,--w1,--w2, --json_output
  - 过程：
    - 从 npz 取 query 的 x0/x1/x2
    - 计算 e0/e1/e2
    - 分别对三库 search topK（cosine，按 score 降序）
    - 计算每尺度概率 p_m（相似度加权投票）：
      w_i = exp(gamma * score_i)
      p_m = Σ w_i*label_i / Σ w_i
    - 融合：
      - fixed：p = w0*p0+w1*p1+w2*p2（w 归一化）
      - learned：从 ckpt 读取 3 个融合权重 logits，经 softmax 得到 w（训练脚本若未实现 learned，则 learned 模式报错并提示）
    - 输出解释：
      - 每尺度 topK 列表（rank,id,label,score）
      - p0/p1/p2、融合权重、最终 p
    - 支持 --json_output 输出 JSON（便于后续接 LLM 做解释）

(7) 最后给出三条可运行命令（必须保证命令与文件路径对得上）
- 训练：
  python scripts/train_embedding.py --npz_path features/alldata_features_no_tid.npz --out_dir runs/emb_exp1 --epochs 20 --batch_size 256 --lr 1e-3 --use_bce true --lambda_bce 0.5 --balanced_sampling true
- 入库：
  python scripts/ingest_to_qdrant_3scales.py --npz_path features/alldata_features_no_tid.npz --ckpt_path runs/emb_exp1/checkpoint.pt --splits_path runs/emb_exp1/splits.json --split train --qdrant_url http://localhost:6333 --collection_prefix accident_kb_no_tid --batch_size 256
- 查询：
  python scripts/query_rag_3scales.py --npz_path features/alldata_features_no_tid.npz --ckpt_path runs/emb_exp1/checkpoint.pt --qdrant_url http://localhost:6333 --collection_prefix accident_kb_no_tid --query_index 123 --top_k 10 --gamma 10 --fusion_mode fixed --w0 0.5 --w1 0.3 --w2 0.2 --json_output true

【输出要求】
- 以“增量修改”为目标：只新增必要文件、少量改动现有代码（如需要注册模块/路径）
- 给出每个新增文件的完整内容
- 若你需要修改现有文件（比如 __init__.py、requirements、pyproject），请以 diff 或明确指出修改点
- 代码必须能在 Windows 下运行，路径处理用 pathlib
- 所有数组 shape、dtype 在关键处 assert 校验并日志打印
- 默认不把原始序列 (L,64) 存入 Qdrant payload（太大），只存轻量 summary 与 top timesteps

现在开始按上述要求实现。


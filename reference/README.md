# Reference: prior work (MovieLens-1M, HyRec-LBERT-LongTail)

These files are READ-ONLY context. Do not import from them or copy code verbatim.
Port ideas into ml/ following CLAUDE.md, especially section 2 (lessons and bugs).

| File | What it contains | Use |
| --- | --- | --- |
| LightGcn_implementation.ipynb | Per-user 80/20 chronological split, LightGCN (3 layers, BPR, multi-negative, ego-embedding reg, early stopping) | Port the model and loss into ml/models/lightgcn.py. Do NOT port the split, the item-aware split fix, or the Python-loop negative sampler |
| MLM_implementation.ipynb | Frozen BERT [CLS] chunk embeddings, random Gaussian projection 768->64, alignment to LightGCN item ids | Context only. Replaced by MiniLM. Cells ran out of order; top-to-bottom execution saves misaligned embeddings |
| MLM_Final.ipynb | BERT mean pooling + PCA(64) + MinMax scaling | Context only. Never MinMax-scale embeddings before dot products |
| Evaluation.ipynb | Metrics, static fusion alpha sweep, ProjectionHybrid (W: 64x64, BPR), long-tail and cold-item evaluation | Read the outputs: raw vs L2-normalized LightGCN explains most of the reported fusion effect (CLAUDE.md section 2.2 #1). The item_final_emb.npy evaluated in later cells is the static fusion, not the projection model; projection training was interrupted |
| paper_projection_fusion.pdf | The submitted paper | Background for the projection-hybrid idea and the long-tail protocol |

Known bugs to never repeat: CLAUDE.md section 2.2 (rows 1-13).
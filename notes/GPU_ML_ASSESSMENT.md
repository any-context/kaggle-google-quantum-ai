# GPU / 機械学習的手法でスコアは動くか — 評価と実測（2026-11-10）

対象: Kaggle「Google Quantum AI — Verifiable Quantum Circuit Compilation」。
現行提出 P3（invsqrt `13f90cb4` 95,035.50 / dqi `337cb049` / chem `d994cd72`）、public 実測 **25.17707**（提出 58153543）。
問い: RTX 4080（16 GB）で勾配法・集団探索などの ML 的手法を回して、ここから上げられるか。
結論を先に書くと **現行部品に対しては動かない**。理由と実測を残す。

---

## 1. 三課題のどこに連続自由度があるか

| 課題 | コスト | 自由度 | ML/GPU の余地 |
|---|---|---|---|
| invsqrt | `Q × (Toffoli + T/4)` | 回路そのもの（離散） | 無い。前回の SAT シフタ探索（CPU）が正しい道具 |
| dqi | `R(eps+) × Q² × T` | 回路そのもの（離散）。`R` は eps+ の床（失敗 0 でも 1.19）で動かない | 無い |
| dfthcsos | `λ_eff × T × Q` | **重み w (R,B,C) と W (R,C) は連続**。角度語（1,220 行 × 75 語、8 bit）は 15 個のローダ回路が焼き込んでいるので固定。T, Q は (R,B,C,brot) とローダで決まる | ここだけ。固定 words での重みリフィット、または C の削減 |

角度語を動かす「再学習」は、Syed v9 ローダ（1 パス 4,152 Toffoli、ANF の積ランク下界に張り付き）を作り直せる
コンパイラが手元に無いので不可能（参照 QROM に戻すと ANGLES だけで 10 倍近く悪化する）。
したがって試せるのは **words 固定** の 2 本だけ: (a) 重みリフィット、(b) コピー数 C の削減。

## 2. 道具: 閉形式コストモデル（`tools/chem_refit_gpu.py`）

公式検証器の指標を N⁴ テンソルを経由せずに書き直し、words 固定で一度だけ核を前計算する。

```
<X_rc, X_r'c'> = w_rcᵀ P_rr' w_r'c'        P[(r,b),(r',b')] = (u_rb · u_r'b')²            (1144×1144)
<G, Tt>        = Σ_rc w_rcᵀ K_r w_rc        K_r[b,b'] = (u_rb u_rb | u_rb' u_rb')_Tt         (8×143×143)
error_2²       = Σ (wPw)² − 2 Σ wKw + ‖Tt‖²                 Tt = tei + (S⊗I + I⊗S)/2
error_1        = ‖Σ_j λ_match(j) q_j q_jᵀ − h1'‖_F          q_j = 固定の one-body 行、h1' は w,W にアフィン
```

- 公式 numpy 計算（`evaluate_dfthc_angle_words`）との一致: error_1/2, Λ, E_gap, λ_eff すべて相対 1e-12。
- 1 評価（f + 勾配、float64）: CPU 3.8 ms、GPU 4.6 ms（k=1）、GPU 0.9 ms/個体（k=32 バッチ）。
  旧 `tools/chem_refit.py`（N⁴ 直接）の約 2 s/step から 500 倍速い。核の前計算（Tt は 5776² の float64 = 267 MB）を GPU で行う。
- 変数は w = a − b（a, b ≥ 0）に分割し、|w| の折れ点を境界制約として扱う（Adam のように符号で振動しない）。
- 最適化: 拡張ラグランジュ（3 ゲート: error_1, error_2 ≤ 0.02 − 2e-9、E_gap ≥ 3.914898 + 2e-9）を外側で更新し、内側を scipy L-BFGS-B（有界）で解く。
  最後に 3 制約への Gauss–Newton 復元 → 公式 numpy で再評価 → npz 書き出し。集団版（Adam、個体ごとの lr・摂動）も同梱。

## 3. (a) 固定 words の重みリフィット — 結果: 既に最適

出発点 `d994cd72`（λ_eff 46.890790935、error_1 = error_2 = 0.02 − 2.3e-10、E_gap = 床 + 2.5e-10。三制約すべて有効）。

| 実験 | 結果 |
|---|---|
| ALM + L-BFGS-B、無摂動、6 外側 × 300 内側 | 違反 1e-4 の途中では 46.884 まで下がるが、違反 → 0 で 46.8907 に戻る。制約乗数 μ(error_2) ≈ 340: ゲートを 1e-9 緩めても得は 3e-7 |
| 摂動再出発 8 本（相対 1e-3 … 3e-2、各 12 外側 × 400 内側、GPU 並列） | 全て出発点より悪い: 46.8910 / 46.8910 / 46.8930 / 46.8917 / 46.8949 / 46.8951 / 46.9545 / 46.9246。別の盆地は見つからない |
| Adam 集団（lr 1e-6 … 1e-5） | 境界追従ができず発散気味（λ 51 → 46.89 付近へ戻るが不可行） |

前セッションの射影勾配（46.89129、`build/chem/runs`）と maoxin の公開 refit（46.89079）は同じ盆地の収束精度の差で、
公開ファイルの方が既に到達点。discussion の「固定 factorization では refit は ≤ 0.3 %」「任意 factorization でも λ_eff ≥ 46.8」とも整合。

## 4. (b) コピー数 C の削減 — 結果: 容量不足で不可

C を 1 減らすと heuristic Toffoli が 78 減る（T 14,966 → 14,888、−0.52 %、Q は 427 のまま）。words は不変なのでローダはそのまま使える。
問題は error_2 ゲート。`tools/chem_drop_screen.py` で、各 c を落として error_2 だけを L-BFGS（3,000 反復）で最小化した床:

| 構成 | error_2 の床 | 判定 |
|---|---:|---|
| C = 44（現行） | **0.019744** | ゲート 0.020 まで 1.3 % しか余裕が無い |
| C = 43（最良の c = 5 または 33 を削除） | **0.021285** | ゲート超過（+6.4 %）。44 通りすべて 0.0213 以上 |

words を固定したまま C を減らす経路は閉じている。alexchilton らの「pruning the weakest copy +0.29 %」「Capacity, not the optimiser」と同じ結論。

## 5. 結論と次の一手

- GPU / ML 的手法で現行部品から上げられる連続自由度は、化学の重み 50,688 個だけで、それは既に KKT 点にあり、容量（error_2 床 0.01974）も使い切っている。
  **本ループでは提出を行わない**（改善 0）。P3 25.17707 が現行最良。
- 動くとすれば離散側: invsqrt 逆正規化ブロック（CSWAP 48 個、`notes/CV_WORST_CASE.md` §8.3 の続き）の SAT 探索。
  正規化側で 47 → 42 が出ているので、同程度なら −5 × 62 = −310 コスト（総合 +0.11 %、約 +0.027）。
- GPU 環境は残してある: `build/tvenv`（uv、torch 2.15 + CUDA 13、Python 3.13）。検証器のデータは main checkout の `data/` を自動検出。

```bash
# 一致確認だけ
build/tvenv/bin/python tools/chem_refit_gpu.py --init <dfthcsos>/submission.npz --check-only
# ALM + L-BFGS-B（既定）
build/tvenv/bin/python tools/chem_refit_gpu.py --init <dfthcsos>/submission.npz --outer 12 --inner 400 --out build/chem_gpu/refit.npz
# コピー削減の床
build/tvenv/bin/python tools/chem_drop_screen.py --init <dfthcsos>/submission.npz --iters 3000
```

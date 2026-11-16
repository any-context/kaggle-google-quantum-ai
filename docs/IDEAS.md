# アイデア台帳（parked / rejected / adopted）

ルール: 数値のない棄却はしない。駄目なら **parked + 復活条件**、捨てるなら **rejected + 証拠**（memory: no-premature-rejection）。
`build/cv/ledger.jsonl` が一次記録。ここは判断の索引。

| id | 日付 | 案 | 状態 | 証拠 / 復活条件 |
|---|---|---|---|---|
| I-01 | 2026-11-08 | **DQI: 測定先 Toffoli 恒等式**（CCX a b t; HMR t m → HMR t m; CZ a b if m） | adopted（再現） | `tools/dqi_measure_first.py scan` が 77bace9d に 14 サイト検出、apply 出力は Kirill の 337cb049 と sha256 一致、公式検証 PASS 3.74738e11（既定 draw） |
| I-02 | 2026-11-08 | I-01 を他の公開 DQI に適用（2c94a1d5 / 4432ec5f / f277b054 / 3ec03d5d） | rejected（測定済み null） | 4 本とも 0 サイト。既に取り込まれている |
| I-03 | 2026-11-08 | I-01 の一般化: HMR 前の窓に CX/CCX/X の**書き込みのみ**を許し、各書き込みを m 条件付き位相（CZ / Z / NEG）に変換 | rejected（測定済み null） | `--generalized` で 77bace9d 14（同じ）、2c94a1d5 0。追加サイトなし。**復活条件**: 窓内の「読み」を許す拡張（例: `CZ t x` は位相なので移動可能か要証明）、または invsqrt / ANGLE_k 回路への適用（未測定） |
| I-04 | 2026-11-08 | **化学: 固定 words での W/H2_I リフィット**を torch で自作 | parked | 閉形式 λ_eff は公式と 1e-11 相対で一致（`tools/chem_refit.py`）。Adam+ペナルティは制約内側へ逸れて悪化。射影勾配+Gauss-Newton 復元（alpha 1e-3/1e-2, 300 step）は余白（err −5e-10, gap +5e-9）の分だけ悪化し λ_eff −1.2e-5〜−1.5e-5（相対 −3e-7）。maoxin の点は 3 制約が 2.4e-10 の余白で活性。**復活条件**: 制約多様体上の 2 次法（活性集合 SQP、Hessian-vector 積）か、maoxin が示した凸緩和下界 46.8497 との差 0.09 % を狙う専用ソルバ。期待利得は総合 +0.0075 程度 |
| I-05 | 2026-11-08 | DQI ファイル選択を **8 salt の期待コスト**で行う（public の 1 draw ではなく） | adopted | `tools/dqi_salt_summary.py`（9 draw = default+g1..g8）: 2c94a1d5 0/9 fail, mean 3.74713e11（最良）；337cb049 0/9, 3.74738e11；77bace9d 2/9 draw で fail, 3.75134e11；f277b054 3/9, 3.75471e11；4432ec5f 3/9, 3.75472e11。public では S5(f277b054)=25.13680, S3(4432ec5f)=25.13678 が上位だが期待値では下位 2 本。**最終 2 提出は 2c94a1d5 / 337cb049 を推奨** |
| I-06 | 2026-11-08 | 前身リポの自作回路（invsqrt flags Q=231、dqi E8 freeze）を再利用 | rejected | 公開フロンティア（invsqrt 62q / 95,493、DQI 892q）に対し桁違いに劣る（ratio 1.26 vs 21.7） |
| I-07 | 2026-11-08 | invsqrt への I-01 / 測定先行（hosen42 の measure-first）再走査 | rejected（測定済み null） | 13b5bf7a (53,711 行) に基本・一般化とも 0 サイト。maoxin の「write-only run は残っていない」を自前で確認 |
| I-08 | 2026-11-08 | 化学: 新しい table（Lera C=45 / Syed C=44 以外）や 7-bit words | parked | maoxin: (7,143,54) は 18.7 mHa で不適合。**復活条件**: 自前の factorization 最適化器（U も動かす）を持てたとき |

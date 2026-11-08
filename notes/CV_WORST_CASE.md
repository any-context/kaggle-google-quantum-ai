# CV–LB 相関と worst-case CV、公開フロンティアの取り込み（2026-11-08）

対象: Kaggle「Google Quantum AI — Verifiable Quantum Circuit Compilation」。
この文書は (1) ディスカッションの CLI 取得、(2) 複数 salt の worst-case を CV にする手順、
(3) 公開手法のうち現行提出より良い部品の取り込み結果、の 3 点を記録する。
数値はすべて未改変の公式検証器（`data/verifiers/*`）をローカルで回した実測。

---

## 1. CV と LB の関係（結論）

- 隠しテストデータは無い。採点は提出ファイルそのものを検証器で走らせたコストの比の幾何平均。
  乱数は `sha256(ファイル) + 行ごとの seed_salt` で種付けされ、public 行と private 行は salt が違うだけ。
- invsqrt と dfthcsos は salt に対して実質不変（invsqrt は 2 salt で相対差 1e-8、dfthcsos はビット一致）。
- DQI だけ `R(eps+)` が 102,400 shot 中の失敗数 k で跳ねる。k=1 で DQI コスト +0.4593 %、総合 −0.153 %。
  `eps+ >= 1/16` だと metric が例外を投げ、提出全体が無効。
- したがって CV–LB 相関はほぼ 1。ずれる経路は DQI の離散ノイズ（1 draw あたり 0.15 % 刻み）と、
  サンプル依存の構成が private salt で崖を踏むテールだけ。
- 実例: `2c94a1d5` はローカル 9 draw で k=0 だったが public draw では k=2（25.06646、S4 提出）。
  public LB の値も「1 draw の観測」として worst-case の集合に入れるべき。

## 2. 道具

| 道具 | 役割 | 出力 |
|---|---|---|
| `tools/fetch_discussions.py` | Kaggle CLI（`competitions topics list` / `topic-messages`）で全トピックと返信を取得。増分更新、`--grep`、`--show` | `build/discussions/topics.json`, `topics/<id>.json`, `digest.md` |
| `tools/salt_cv.py` | 公式検証器を複数 salt で回し、課題ごとに default / mean / **worst** コストと失敗数、総合スコアを default / mean / WORST で出す。`--salt-set gate`（g1..g8）で候補間の CRN。worktree からでも main checkout の `data/` と `.venv` を自動検出 | `build/cv/salt_ledger.jsonl`, `build/cv/salt_raw/<tag>/` |

```bash
python3 tools/fetch_discussions.py                 # 増分取得 + digest 再生成
python3 tools/fetch_discussions.py --no-fetch --grep "RezOps" --grep "invsqrt"
python3 tools/salt_cv.py build/sub/<tag>/submission.zip --salt-set gate --tag <tag>
python3 tools/salt_cv.py some_dqi.kmb --tasks dqi --salt-set gate --tag dqi_<hash>
python3 tools/salt_cv.py --summary                 # 台帳の worst-case 表
```

採用ゲート: **WORST（全 draw の最悪コスト、失敗があれば 0）で比較する。** mean は期待値の参考値。

## 3. 公開フロンティアの価格付け（公式検証器、2026-11-08）

出典は RezOps Briefing（manwithacat、discussion 756824 ほか）と heliosli の `qcc-circuits-live` データセット、
isabe11am00re v88、lucifer19 black-cat v3、prvsiyan `vqcc-verified-best-of-circuits-1106a`、ozertuu / alexahern のノートブック出力。

### invsqrt（salt 不変）

| ファイル | 出典 | コスト | 比 | 判定 |
|---|---|---:|---:|---|
| `13b5bf7a` | 現行（S1–S5） | 95,492.959 | 21.74793 | |
| **`1c5b1ee2`** | heliosli live v95 = `13b5bf7a` + hosen42 の 11 書き込み→1 CCX 統合 + maoxin123 の Block-4 条件分割（各 −1 Toffoli） | **95,368.949** | **21.77621** | **採用** |
| `bbfa9675` | Atakan Aldemir、prvsiyan alt_p3 | 95,957.894 | 21.64256 | 劣る |
| `95621d7e` | maoxin123 LB 25.09（RezOps 11-07 時点の最良） | 96,020（RezOps 再検証値） | 21.63 | 劣る |

### dfthcsos / chemistry（salt 不変、15 loader + npz は 1 単位）

| npz | 出典 | コスト | 比 | 判定 |
|---|---|---:|---:|---|
| `c1321df2` | 現行（maoxin123 refit） | 299,657,851.81 | 11.19810 | |
| **`fd4d4d00`** | heliosli live v95（maoxin v3 重みの再精錬） | **299,656,901.22** | **11.19814** | **採用**（+0.0003 %） |
| `29418b6c` | isabe11am00re v88 | 299,662,412.46 | 11.19793 | 劣る |
| `3be62649` | alexahern v45 / maoxin v3 | 299,662,440.61 | 11.19793 | 劣る |
| `a71c1664` | prvsiyan 既定 | 299,671,051.22 | 11.19761 | 劣る |

### DQI（salt 依存。default + g1..g8 の 9 draw）

RezOps の 48 draw 平均では `2c94a1d5`（3.7504e11）が期待値最小、`337cb049` は 14 サイトの測定恒等式のみ（3.7508e11 級）。
公開ファイルで新しいものは無かった（heliosli live の `f277b054`、isabe の `4432ec5f`、prvsiyan の `7b6cbb92` / `b4bbf86d` / `df65bd5b` は既出で、
`f277b054` と `4432ec5f` は main checkout の 9 draw で各 3 draw 失敗）。

| ファイル | draws | 失敗 | k（default, g1..g8） | default | mean | worst | 比(worst) | public draw |
|---|---:|---:|---|---:|---:|---:|---:|---|
| **`337cb049`**（Kirill、現行 S2） | 9 | 0 | 0 0 0 0 0 0 0 0 0 | 374,738,368,864 | 374,738,407,186 | 374,738,760,562 | 65.21734 | k=0（25.13672） |
| `2c94a1d5`（Syed、期待値最良） | 9 | 0 | 0 0 0 0 0 0 0 0 0 | 374,712,921,286 | 374,712,914,575 | 374,713,075,061 | 65.22181 | **k=2**（25.06646） |

ローカル 9 draw だけなら `2c94a1d5` が worst でも 0.007 % 上回る。しかし public draw を 10 本目の観測に入れると
`2c94a1d5` の worst は k=2（DQI コスト +0.92 %）になり、`337cb049` が worst-case で勝つ。
**採用: `337cb049`。** `2c94a1d5` を private 用の 2 本目に残す余地はある（期待値では最良、
RezOps 48 draw でも最小）が、同一ファイルで k=2 を一度引いた事実は消えない。

## 4. 組み立てた候補と期待スコア

`build/sub/P1_1c5b1ee2_337cb049/submission.zip`（28,075,639 B、sha256 `1ef6c75d…`、metric 形式チェック ACCEPT）

| 部品 | ファイル | 比 |
|---|---|---:|
| invsqrt | `1c5b1ee2` | 21.77621 |
| dqi | `337cb049` | 65.21740（k=0） |
| dfthcsos | Syed v9 loaders + `fd4d4d00` | 11.19814 |

| 条件 | 総合 |
|---|---:|
| 現行 S2（13b5bf7a / 337cb049 / c1321df2）public 実測 | 25.13672 |
| 候補 P1、DQI k=0 | **25.14764** |
| 候補 P1、DQI k=1 | 25.10926 |
| 参考: P1 の DQI を 2c94a1d5 にした場合、k=0 | 25.14821 |

改善は +0.043 %（≈ +0.011）。全て salt 不変の部品差し替えなので public と private の両方にそのまま乗る。
public 25.09052（33 チーム同点、silver 線）の上、25.1368 群の上に出る。

## 5. 次の一手（ディスカッションから）

- discussion 757030「Open problems」: 正規化バレルシフタの CSWAP 下限（w≈21 で現状 40 台後半、下限 30 台後半）、
  invsqrt 62 qubit の床、GF(2^12) 積 42 Toffoli の非双線形トリック、loader-aware rounding。
  公開フロンティアはこれらで止まっており、上位 10（27–29.3）との差はここにある。
- RezOps 11-07: 化学は固定 words で残り ≈0.09 %（λ_eff ≥ 46.8497 の凸緩和下界）、総合 +0.008 相当。
- 1 日 5 提出のうち、アンカー 1 本は「既知 draw の構成」にして採点パリティを確認し続ける。

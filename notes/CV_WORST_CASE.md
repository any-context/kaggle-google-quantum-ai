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
| **P1 実測 public（提出 58144077、2026-11-09 00:01 UTC）** | **25.14764**（予測と 5 桁一致、DQI public draw k=0） |
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

## 6. 技術ガイド PDF（2026-11-02 改訂版）の手法による追加探索 — 2026-11-09

`Google_Verifiable_QCC_Technical_Guide.pdf` 第 3 章の局所手法を、現行最良部品に全入力で適用した結果。
道具: `tools/inv_scan.py`（全入力ビット並列シミュレータ + アフィン Toffoli / 冗長 CSWAP / 死 CCZ 走査）、
`tools/inv_scan2.py`（ダーティ 5→4 CCX、履歴キャッシュ + 占有幅、回転統合）、`tools/angle_scan.py`（ANGLE_k 族を到達可能領域で走査）。

| 手法（PDF 節） | 対象 | 結果 |
|---|---|---|
| 3.3.1 AND の測定消去（measure-first） | invsqrt `1c5b1ee2` | 0 サイト（基本・一般化とも） |
| 3.3.2 ダーティ借用線 5→4 CCX | invsqrt | パターン 0 件 |
| 3.3.3 履歴キャッシュ | invsqrt | 同一積の再計算 241 対、全区間で占有が 62 に達し保持不可（PDF の 10/28 解析と同じ構造） |
| 3.3.4 アフィンで置換できる Toffoli、冗長 CSWAP、死 CCZ | invsqrt、全 3,133,452 入力 × 3 乱数系列 | **無条件ゲートでは 0 件** |
| 3.5.2 / 3.5.5 Z_POW 統合（同一線、値が等しい線、定数線） | invsqrt 48 回転 | 0 件 |
| 同上の走査 | dfthcsos ANGLE_0..14（4,152 Toffoli 様、到達可能 2,440 状態） | 0 件 |

教訓: 条件ブロック内のゲートを「発火したショットだけで」判定してはいけない。初回の走査は 13 段ネストの
条件ブロック内の CCZ 15 個を「積が常に 0」と誤判定し、公式検証器で即 FAIL した（x=68 の 1 ショットで発火 ∧ 積=1、
発生率 ≈ 2^-10 × 稀な積）。修正後の基準は「条件マスクに関係なく全ショットで性質が成り立つ」こと。

結論: PDF の局所手法は公開フロンティア回路で既に使い尽くされている（PDF 自身の結論とも一致）。
これ以上は構造的な再設計（区間分割・Horner 精度配分・lookup 合成、discussion 757030 の未解決問題）が必要。
本ループの提出は P1（§4）のまま。

## 7. 提出前の公開フロンティア再確認 — 2026-11-09 06:30 JST

- RezOps Briefing 757068（11-08 夜）: 公開フロンティアは invsqrt `1c5b1ee2`（95,368.95）、dqi `2c94a1d5`（48 draw 平均 3.7504e11、新規 3 本は期待値で 0.01 % 以内）、
  chemistry `cfe4e821`（λ_eff 46.89113、2.99657e8）。最良部品の幾何平均 25.1409（RezOps 静的モデル）。P1 の invsqrt / chem はこれと同等。
- isabe11am00re v96（mxfu1/3/6）: invsqrt は `1c5b1ee2`（同一）、chem npz `9cd87121` = 299,656,901.27（P1 の `fd4d4d00` 299,656,901.22 と実質同値、僅かに劣る）。DQI は新規 3 本（`b9af43b6` / `6658b1c5` / `b71b21ca`）。
- prvsiyan `vqcc-measured-cswap-rewrite`: `4432ec5f` の CSWAP 48 サイトを測定先行に書き換えた DQI `4e9b50b1`、無塩コスト 3.74690e11（`337cb049` 比 −0.013 %、総合 +0.004 %）。9 draw 採点中。
- 判断: P1 の DQI は `337cb049` を維持。理由は (1) 同一バイト列の public draw が k=0 と既知で、新ファイルは public の draw を引き直す宝くじになる（k=1 で −0.15 %、期待利得 +0.004 % を桁で上回る）、(2) worst-case CV は既知 draw を含めて評価する方針（§3）。
- 追記 06:35 JST: `4e9b50b1` の 9 draw 結果は default 3.74690e11 だが g1・g8 で k=1（2/9 失敗 draw）、worst 3.76411e11（比 64.928 < `337cb049` の 65.217）。worst-case CV でも `337cb049` 維持が正しい。

## 8. P2 提出と invsqrt シフタ探索（discussion 757030 問題 1） — 2026-11-09

### 8.1 P2（提出 58153288、2026-11-09 05:48 UTC）

公開フロンティアが動いた（maoxin123「LB 25.1498」、kirilltym「Conditional Uncompute Audit」、heliosli live v99 が同じ部品を出荷）。

| 部品 | P1 | P2 | 公式検証器 |
|---|---|---|---:|
| invsqrt | `1c5b1ee2` 95,368.95 | **`034c7c2a`** | 95,345.54（比 21.78156） |
| dqi | `337cb049` | `337cb049`（据え置き、public draw k=0 既知） | 65.21740 |
| dfthcsos | `fd4d4d00` 299,656,901 | **`d994cd72`**（Syed v9 loaders 同一、npz のみ更新） | 299,654,755.44（比 11.19822） |

期待値 **25.14976**（k=0）/ 25.11137（k=1）。P1 実測 25.14764 → +0.0021。
yoyobar v? の `46743f7f` / `3f72cbe5` は 95,345.51 / 299,654,755.41 で 1e-7 相対しか変わらないため、3 者が同一バイト列を出荷している P2 構成を採用。
kirilltym の audit に DQI `e1354c89`（3.74688e11、比 65.2262、k=0 単 draw）が記録されているが、ファイルは配布されておらず、また新ファイルは public draw を引き直す（期待 +0.013 % 対 k=1 で −0.15 %）ので見送り。

### 8.2 invsqrt 正規化シフタの SAT 探索（道具: `tools/shifter_sat.py`, `tools/shifter_splice.py`, `tools/shifter_layers.py`）

- 現行回路の正規化ブロック（行 130–293）は 22 線・22 シフト量のバレルシフタで CSWAP 47 個。制御は z の閾値温度計（q22, q23..q41, q47）の XOR で無料に作れる。逆正規化（行 53588–53696）は 48 個。
- `shifter_splice.py` で任意の CSWAP 網を差し替え、全入力シミュレータで一致確認 → 公式検証器。79 ゲートの素朴バレルを差し込むと厳密に +32×62 のコスト増で PASS（配管の確認）。1 CSWAP = 62 コスト = 総合 +0.022 %。
- SAT モデル（pysat / CaDiCaL）: discussion の小規模最適値（n=4..7: 4, 5, 7, 9）を再現。先頭の 1 を定数として扱う緩和（X は無料）で n=4..9 の最適は 2, 4, 5, 7, 9, 11（= 2n−7 → n=22 なら 37 の可能性）。
- n=22 直接は CaDiCaL で未決（discussion と同じ）。公開設計のペア集合をプールにした「部分集合選択」モードは K=47/46 とも UNSAT（公開設計は CX 補助で成立しており純 CSWAP 網ではない）。ランダム反転層プール（約 220 試行）も UNSAT。
- 進行中: rich / barrel / barrelpub / public2x プール（K=46）、n=10..12 の 2n−7 確認、n=22 N=44 直接。結果は `build/shifter/{pool,cad}/*.log`。

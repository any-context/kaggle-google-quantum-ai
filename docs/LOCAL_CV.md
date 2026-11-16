# ローカル CV（採点）ベストプラクティス — Google Quantum AI 回路コンパイル

対象: Kaggle「Google Quantum AI — Verifiable Quantum Circuit Compilation」(締切 2027-01-20, 1日5提出)。
any-context の過去コンペ（rogii / biohub / gemma-agent / kaggriculture / 本リポの前身）で確立した
実践を、このコンペの採点構造に合わせて書き直したもの。作成 2026-11-08。

---

## 0. このコンペの「CV」は何か

通常の ML コンペと違い、ホールドアウトも fold もない。**公式検証器がそのままローカルで動き、
ホスト側と同じコストを返す**。したがって CV = 「公式検証器を、採点時と同じ条件で、複数 draw について回すこと」。

| 課題 | コスト | draw 依存 | ローカル所要（本機 24C/29GB） |
|---|---|---|---|
| invsqrt | Q × C（C = Toffoli + T/4） | なし（全 3,133,452 入力を全数検証） | 約 60–90 s |
| dqi | R(ε+) × Q² × T | **あり**: 102,400 shot の失敗数で R が跳ねる | 約 25–30 s（`--max-workers 12`, RSS 4.7 GB） |
| dfthcsos | λ_eff × T × Q | なし（λ_eff は words と重みから閉形式） | 約 6 s |

総合スコア = Π(基準コスト/コスト)^(1/3)。基準コストは公式メトリクスの定数
（invsqrt 2,076,774.23 / dqi 2.4439e13 / dfthcsos 3.3556e9）。

**draw の正体**: 採点側は「salt + ファイルの SHA-256」で乱数を種付けする（Fiat–Shamir）。
同じバイト列を再提出しても public の draw は変わらない（RezOps 2026-11-07 の実測: ホスト採点と
ローカル同 seed は失敗 shot まで一致）。private は別 salt なので **全ファイルが新しい draw** を引く。

---

## 1. 原則（any-context 横断ルールのこのコンペ版）

1. **公式メトリクスを再実装しない。** コストは必ず `data/verifiers/*` の未改変検証器から読む。
   高速モデル（`src/common/dfthc_fastscore.py`）は探索用で、採用判定には使わない。
   （biohub `cv-design.md` / 本リポ `validate_submission_format.py` が公式 metric kernel を再利用する流儀）
2. **draw 依存のコストは期待値で比較する。** dqi は 1 failure = 総合 −0.15 %、候補間差は 0.001–0.03 %。
   既定 salt 1 本の値で選ぶと「draw を選んでいる」だけになる。`tools/local_cv.py --salts s1 s2 ...`
   で複数 draw を取り、平均コストと失敗率で比較する（RezOps は 48 draw）。
3. **public LB は draw 1 本の観測であり、採用ゲートではない。** ただし「同一ファイルなら public は確定値」
   なので、ローカル既定 draw と public の一致は **採点パリティの検査**として使う
   （rogii の LB ノイズ床測定に相当。ここではノイズ床 = 0、不一致は環境差のシグナル）。
4. **コストは実行前に測る。** 新しい回路種は `--num-shots 4096`（dqi）/ `--max-x 20480`（invsqrt）で
   時間と RSS を実測してから全量を回す（memory: estimate-cost-before-running）。
5. **10 分を超える仕事は detach。** `setsid nohup ... > build/.../log 2>&1 < /dev/null &`、成果物は
   ファイルで判定（memory: long-jobs-detached）。
6. **全測定を台帳に残す。** `build/cv/ledger.jsonl` に 1 行/（ファイル, 課題, salt）。要約行も同じ台帳。
   手で書き写さない（rogii `.claude/rules/reporting.md`: スコアボードは台帳から生成）。
7. **提出は台帳経由のみ。** `submissions/registry.jsonl` に **提出前に**行を書く（tag, UTC, zip sha256, bytes,
   各部品の sha256 先頭, ローカル既定 draw スコア, 1 提出 = 1 帰属可能な変更）。採点後に public を追記し、
   ローカル値との差を見る（rogii `local/submit.py`, 前身 `notes/SUBMISSIONS.md`）。
8. **棄却にも証拠水準。** 数値なしで捨てない。駄目だった案は「parked + 復活条件」で `docs/IDEAS.md` に残す
   （kaggriculture `experiments/IDEAS.md`; memory: no-premature-rejection）。
9. **測定依存の編集は複数 salt で検査する。** 既定 seed で全入力 PASS でも、稀な測定枝の位相修正を
   消していると別 salt で落ちる（Lynn Langit の field notes: 0.5 turn 位相誤差で grader 失敗）。
10. **構造が同じ候補は同じ条件で比べる（CRN）。** 同じ salt 集合・同じ worker 数・同じ検証器版。
    salt 集合は用途で分ける: 探索用 `{s1,s2,s3}`、採用ゲート用 `{g1..g8}`、最終確認用 `{h1..h8}`
    （rogii のシード層分離）。探索中に見た salt でゲートを通さない。

---

## 2. 標準手順

### 2.1 候補ファイルを 1 つ評価する

```bash
# 既定 draw + 2 salt。dqi だけ salt ごとに繰り返す。
.venv/bin/python tools/local_cv.py <portfolio_dir_or_zip> --tag <name> --salts s1 s2
```

出力: 課題ごとの status / cost / ratio / Q、既定 draw スコア、平均 draw スコア、dqi 失敗 run 数。
全行が `build/cv/ledger.jsonl` に追記される。

### 2.2 採用ゲート（部品差し替えのとき）

- invsqrt / dfthcsos: コストが下がり、公式検証器 PASS（invsqrt は全数なので 1 run で十分。
  ただし測定依存の編集をしたら salt 3 本）。
- dqi: 8 salt で (a) 失敗 run 数が基準以下、(b) 平均コストが基準より低い、の両方。
  失敗率が同じなら T の小さい方。失敗率が上がる候補は、T の差 × 期待失敗コストで価格付けして判断。
- 形式: `src/common/validate_submission_format.py <zip>` が ACCEPT（`make_submission.py` が自動実行）。

### 2.3 提出

1. `submissions/registry.jsonl` に行を書く（status=submitting）。
2. `kaggle competitions submit -c google-verifiable-quantum-circuit-compilation -f <dir>/submission.zip -m "<tag>: <1行>"`
   **ファイル名は必ず `submission.zip`**。他の名前だと API が 400 を返し、CLI はエラー本文を出さない
   （本文は「Submission files must be named "submission.zip" for this Competition」。`requests.Session.send` をフックすると読める）。
3. 採点後（約 45 分）に `kaggle competitions submissions` から public を拾い、registry の行を更新。
4. ローカル既定 draw スコアと public の差を確認。dqi 以外で差があれば**環境差**として即調査。

### 2.4 1 日の運用

- 朝: RezOps briefing（ディスカッション最新）と公開ノートブック上位の差分を確認し、新しい部品ハッシュを
  `build/datasets/` に取得 → 2.1 で価格付け → 台帳。
- 提出 5 枠は「アンカー 1 + 手法差分 ≤3 + 自作 ≥1」。アンカーは既知 draw の構成で、採点パリティの確認用。
- 最終 2 提出の選定は **期待コスト（多 salt 平均）** で行う。public の良い draw に引かれない。

---

## 3. 道具

| 道具 | 役割 |
|---|---|
| `tools/local_cv.py` | ポートフォリオ → 3 検証器 → 幾何平均。多 salt、台帳追記。 |
| `src/common/make_submission.py` | 部品から zip を組み、形式検証（公式 metric の `find_one`/`read_circuit`/`read_npz`）まで実行。 |
| `src/common/validate_submission_format.py` | zip 単体の形式検証（`notes/metric_kernel_extracted.py` が必要: `kaggle kernels pull metric/google-quantum-ai-metric`）。 |
| `src/common/dfthc_fastscore.py` | 化学の閉形式コストモデル（シミュレーション無し、秒）。 |
| `tools/dqi_measure_first.py` | DQI の「測定先 Toffoli 恒等式」ピープホール（自作。77bace9d → 337cb049 を sha256 一致で再現）。 |
| `tools/dqi_salt_summary.py` | 塩スイープの集計（draw ごとの失敗数、平均コスト、ref/mean）。 |
| `tools/chem_refit.py` | 化学の固定 words 重みリフィット（torch、公式と 1e-11 一致、射影勾配）。 |
| `build/cv/ledger.jsonl` | 全測定の台帳。 |
| `submissions/registry.jsonl` | 提出台帳。 |

---

## 4. 既知の落とし穴

- **検証器 CLI の既定 shot で結論を出すな**: `--num-shots 256` などの smoke では ε+ の上界が 1/16 を
  超えて FAIL になる（統計的に当然）。smoke は時間見積り専用。
- **`--salt` 無し = 公開ノートブックが言う「default salt」**。他人の報告値と比べるときはこれ。
- **dqi の public draw はファイル固定**。public で失敗した dqi ファイルは再提出しても直らない。
  一方 private では全員新 draw なので、期待コスト最小のファイルが正解。
- **f277b054（heliosli live v89）は既定 draw で 1 failure** だった。T は 4432ec5f と同等。多 salt で再評価しないと優劣は言えない。
- **リポの `src/` は平坦化されていて `sys.path` が壊れている** → venv の `.pth` で解決済み（memory: google-quantum-ai-comp）。
- 複数 dqi 検証を同時に走らせると CPU を取り合い、所要が伸びる（各 25 s → 2 並列で 30 s）。順次でよい。

---

## 5. 出典（any-context の実践）

- rogii: `.claude/rules/cv-design.md`（相対権威のみ、シード層分離、CRN）、`docs/dev/validation-protocol.md`
  （証拠階層、LB 悪化時の巻き戻し規則）、`local/submit.py` + `submissions/registry.jsonl`、`experiments/cv-lb-points.csv` → `charter.py`。
- biohub: `.claude/rules/cv-design.md`「公式メトリクスを再実装しない」、`experiments/charter_config.json`「閾値は測るまで空欄」。
- gemma-agent: `docs/dev/20261003-cv-environment-notes.md`（採点環境パリティ構築）、`scripts/submit.py`（提出元の SHA-256 封印）。
- kaggriculture: `CLAUDE.md` シード帯規律、`experiments/IDEAS.md`（parked/rejected）。
- 本リポ前身: `notes/SUBMISSIONS.md`（id/UTC/sha256/bytes/score）、salt lottery の分離報告、`dfthc_fastscore.py`。

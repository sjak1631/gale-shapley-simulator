"""
大学院入試マッチングシミュレーション
Gale-Shapley アルゴリズム（学生提案型）による研究室配属シミュレーション

- 受験者の能力評価スコアにガウスノイズを加えてペーパーテストスコアを生成
- 研究室はペーパーテストスコアのみで受験者を評価
- 受験者は志望順位リストに従い研究室に提案

データ入力:
  applicants.csv  受験者データ（氏名, ability_score, 第1志望, 第2志望, ...）
  labs.csv        研究室データ（研究室名, 定員）
"""

import argparse
import csv
import numpy as np
from pathlib import Path
from typing import Optional

# ============================================================
#  設定 ── 必要に応じて変更してください
# ============================================================

# CSVファイルのパス（このスクリプトと同じディレクトリ）
_BASE = Path(__file__).parent
APPLICANTS_CSV: Path = _BASE / "applicants.csv"
LABS_CSV: Path = _BASE / "labs.csv"

# ノイズ設定
NOISE_STD: float = 1.6   # ペーパーテストスコアのノイズ標準偏差（能力スコアと同スケール）
RANDOM_SEED: Optional[int] = 42  # 乱数シード（再現性が不要なら None に変更）


# ============================================================
#  CSV 読み込み
# ============================================================

def load_applicants(path: Path) -> dict:
    """
    applicants.csv を読み込む。
    列: 氏名, ability_score, 第1志望, 第2志望, ... (志望列は可変)
    """
    applicants = {}
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["氏名"].strip()
            ability = float(row["ability_score"])
            prefs = [
                row[col].strip()
                for col in reader.fieldnames
                if col not in ("氏名", "ability_score") and row[col] and row[col].strip()
            ]
            applicants[name] = {"ability_score": ability, "preferences": prefs}
    return applicants


def load_labs(path: Path) -> dict:
    """
    labs.csv を読み込む。
    列: 研究室名, 定員
    """
    labs = {}
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labs[row["研究室名"].strip()] = int(row["定員"])
    return labs

# ============================================================
#  以下は変更不要です
# ============================================================


def generate_test_scores(
    applicants: dict,
    noise_std: float,
    seed: Optional[int],
) -> dict[str, float]:
    """能力スコアにガウスノイズを加えてペーパーテストスコアを生成する。"""
    rng = np.random.default_rng(seed)
    return {
        name: float(data["ability_score"] + rng.normal(0, noise_std))
        for name, data in applicants.items()
    }


def build_lab_rank_index(
    labs: dict,
    test_scores: dict[str, float],
) -> dict[str, dict[str, int]]:
    """
    研究室ごとに受験者をテストスコア降順でランク付けし、
    {研究室名: {受験者名: 順位インデックス}} を返す。
    インデックスが小さいほど高評価。
    """
    all_applicants = sorted(test_scores, key=lambda a: test_scores[a], reverse=True)
    # 全研究室が同じテストスコアで評価するため、ランキングは共通
    rank_index = {a: i for i, a in enumerate(all_applicants)}
    return {lab: rank_index for lab in labs}


def gale_shapley(
    applicants: dict,
    labs: dict,
    lab_rank_index: dict[str, dict[str, int]],
) -> dict[str, Optional[str]]:
    """
    学生提案型 Gale-Shapley アルゴリズム。

    Returns
    -------
    matching : {受験者名: マッチング先研究室名 or None}
    """
    next_proposal: dict[str, int] = {a: 0 for a in applicants}
    applicant_match: dict[str, Optional[str]] = {a: None for a in applicants}
    lab_tentative: dict[str, list[str]] = {lab: [] for lab in labs}

    free = list(applicants.keys())

    while free:
        applicant = free.pop(0)
        prefs = applicants[applicant]["preferences"]
        idx = next_proposal[applicant]

        if idx >= len(prefs):
            # すべての志望先に拒否された → マッチングなし
            continue

        lab = prefs[idx]
        next_proposal[applicant] += 1
        capacity = labs[lab]
        current = lab_tentative[lab]

        if len(current) < capacity:
            # 空きあり → 仮マッチング
            current.append(applicant)
            applicant_match[applicant] = lab
        else:
            # 満員 → 最低評価の仮マッチング者と比較
            worst = max(current, key=lambda a: lab_rank_index[lab][a])
            if lab_rank_index[lab][applicant] < lab_rank_index[lab][worst]:
                # 新受験者の方が高評価 → 入れ替え
                current.remove(worst)
                current.append(applicant)
                applicant_match[applicant] = lab
                applicant_match[worst] = None
                free.append(worst)  # 追い出された受験者は再提案へ
            else:
                # 拒否 → 同受験者が次の志望先に提案
                free.append(applicant)

    return applicant_match


def print_results(
    applicants: dict,
    labs: dict,
    test_scores: dict[str, float],
    matching: dict[str, Optional[str]],
) -> None:
    """マッチング結果を整形して表示する。"""
    SEP = "=" * 64

    # ── 受験者別 ──────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  受験者別マッチング結果")
    print(SEP)
    header = f"{'受験者':<12}  {'能力':>4}  {'試験スコア':>10}  {'配属先':>12}  {'志望順位':>6}"
    print(header)
    print("-" * 64)

    for applicant in applicants:
        lab = matching[applicant]
        ability = applicants[applicant]["ability_score"]
        test = test_scores[applicant]
        prefs = applicants[applicant]["preferences"]

        if lab is not None:
            rank = prefs.index(lab) + 1 if lab in prefs else "-"
            rank_str = f"第{rank}志望"
        else:
            lab = "(未マッチング)"
            rank_str = "-"

        print(f"{applicant:<12}  {ability:>4}  {test:>10.3f}  {lab:>12}  {rank_str:>6}")

    # ── 研究室別 ──────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  研究室別マッチング結果")
    print(SEP)

    lab_results: dict[str, list[str]] = {lab: [] for lab in labs}
    for applicant, lab in matching.items():
        if lab:
            lab_results[lab].append(applicant)

    for lab, capacity in labs.items():
        assigned = sorted(lab_results[lab], key=lambda a: test_scores[a], reverse=True)
        fill = len(assigned)
        print(f"\n  {lab}  (定員 {capacity} / 合格 {fill} 名)")
        if assigned:
            for a in assigned:
                prefs = applicants[a]["preferences"]
                rank = prefs.index(lab) + 1 if lab in prefs else "-"
                print(
                    f"    ✓ {a}  試験スコア: {test_scores[a]:.3f}"
                    f"  (第{rank}志望)"
                )
        else:
            print("    (合格者なし)")

    # ── 統計 ───────────────────────────────────────────────────
    total = len(matching)
    matched_count = sum(1 for v in matching.values() if v is not None)
    first_choice = sum(
        1
        for a, lab in matching.items()
        if lab is not None
        and applicants[a]["preferences"]
        and applicants[a]["preferences"][0] == lab
    )

    print(f"\n{SEP}")
    print("  統計")
    print(SEP)
    print(f"  受験者総数         : {total} 名")
    print(f"  マッチング成立     : {matched_count} 名  ({matched_count/total*100:.1f}%)")
    print(f"  第1志望マッチング  : {first_choice} 名  ({first_choice/total*100:.1f}%)")
    print(f"  未マッチング       : {total - matched_count} 名")
    print(SEP)


def run_simple_mode(
    noise_std: float,
    random_seed: Optional[int],
    labs_csv: Path,
    applicants_csv: Path,
) -> None:
    """単純なシミュレーションモード（1回の試行）。"""
    applicants = load_applicants(applicants_csv)
    labs = load_labs(labs_csv)

    print(f"【CSVファイル】")
    print(f"  受験者: {applicants_csv}")
    print(f"  研究室: {labs_csv}")
    print("【ノイズ設定】")
    print(f"  NOISE_STD  = {noise_std}")
    print(f"  RANDOM_SEED = {random_seed}")

    # 1. テストスコア生成
    test_scores = generate_test_scores(applicants, noise_std, random_seed)

    print("\n【生成されたペーパーテストスコア】")
    for name, score in sorted(test_scores.items(), key=lambda x: x[1], reverse=True):
        ability = applicants[name]["ability_score"]
        print(f"  {name}  能力スコア: {ability}  →  試験スコア: {score:.3f}")

    # 2. 研究室のランク付け
    lab_rank_index = build_lab_rank_index(labs, test_scores)

    # 3. Gale-Shapley マッチング
    matching = gale_shapley(applicants, labs, lab_rank_index)

    # 4. 結果表示
    print_results(applicants, labs, test_scores, matching)


def run_probability_mode(
    noise_std: float,
    labs_csv: Path,
    applicants_csv: Path,
    target_lab: str,
    target_name: str,
    ability_score_range: tuple[int, int],
    num_trials: int,
) -> None:
    """内定確率計算モード（複数回試行）。"""
    applicants = load_applicants(applicants_csv)
    labs = load_labs(labs_csv)

    min_score, max_score = ability_score_range

    print(f"【内定確率計算】")
    print(f"  対象者: {target_name}")
    print(f"  対象研究室: {target_lab}")
    print(f"  試行回数: {num_trials} 回/スコア")
    print(f"  スコア範囲: {min_score}～{max_score}")
    print(f"  ノイズ標準偏差: {noise_std}")
    print()

    print(f"スコア {min_score}～{max_score} での {target_lab} 内定確率計算")
    print("=" * 50)

    results = {}
    for test_score in range(min_score, max_score + 1):
        # 対象者のスコアを設定
        if target_name not in applicants:
            print(f"エラー: {target_name} が applicants.csv に見つかりません")
            return

        applicants[target_name]["ability_score"] = float(test_score)

        success_count = 0

        for seed in range(num_trials):
            test_scores = generate_test_scores(applicants, noise_std, seed)
            lab_rank_index = build_lab_rank_index(labs, test_scores)
            matching = gale_shapley(applicants, labs, lab_rank_index)

            if matching.get(target_name) == target_lab:
                success_count += 1

        prob = success_count / num_trials
        results[test_score] = prob
        print(f"スコア {test_score:2d}: {success_count:5d}/{num_trials} = {prob*100:6.2f}%")

    print()
    print("統計サマリー:")
    for score, prob in sorted(results.items()):
        bar_length = int(prob * 50)
        bar = "█" * bar_length + "░" * (50 - bar_length)
        print(f"{score:2d} |{bar}| {prob*100:6.2f}%")


def run_simulation() -> None:
    """シミュレーション全体を実行する。"""
    # CSV 読み込み
    APPLICANTS = load_applicants(APPLICANTS_CSV)
    LABS = load_labs(LABS_CSV)

    print(f"【CSVファイル】")
    print(f"  受験者: {APPLICANTS_CSV}")
    print(f"  研究室: {LABS_CSV}")
    print("【ノイズ設定】")
    print(f"  NOISE_STD  = {NOISE_STD}")
    print(f"  RANDOM_SEED = {RANDOM_SEED}")

    # 1. テストスコア生成
    test_scores = generate_test_scores(APPLICANTS, NOISE_STD, RANDOM_SEED)

    print("\n【生成されたペーパーテストスコア】")
    for name, score in sorted(test_scores.items(), key=lambda x: x[1], reverse=True):
        ability = APPLICANTS[name]["ability_score"]
        print(f"  {name}  能力スコア: {ability}  →  試験スコア: {score:.3f}")

    # 2. 研究室のランク付け
    lab_rank_index = build_lab_rank_index(LABS, test_scores)

    # 3. Gale-Shapley マッチング
    matching = gale_shapley(APPLICANTS, LABS, lab_rank_index)

    # 4. 結果表示
    print_results(APPLICANTS, LABS, test_scores, matching)


def main() -> None:
    """コマンドラインインターフェース。"""
    parser = argparse.ArgumentParser(
        description="Gale-Shapley マッチングシミュレーター",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 単純なシミュレーション
  python3 simulation.py --type simple

  # john への確率計算（スコア 0-10, me, 1000回）
  python3 simulation.py --type probability --target-lab john --target-name me --score-range 0 10 --trials 1000

  # カスタム CSV パスとノイズ
  python3 simulation.py --type simple --noise-std 2.0 --seed 123 \\
    --applicants custom_applicants.csv --labs custom_labs.csv
        """,
    )

    # 共通引数
    parser.add_argument(
        "--type",
        choices=["simple", "probability"],
        default="simple",
        help="実行モード: simple(単純なシミュレーション) or probability(確率計算)",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=NOISE_STD,
        help=f"ノイズ標準偏差 (default: {NOISE_STD})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"乱数シード (default: {RANDOM_SEED})",
    )
    parser.add_argument(
        "--applicants",
        type=Path,
        default=APPLICANTS_CSV,
        help=f"applicants.csv のパス (default: {APPLICANTS_CSV})",
    )
    parser.add_argument(
        "--labs",
        type=Path,
        default=LABS_CSV,
        help=f"labs.csv のパス (default: {LABS_CSV})",
    )

    # probability モード専用引数
    parser.add_argument(
        "--target-lab",
        default="john",
        help="確率計算対象の研究室 (default: john)",
    )
    parser.add_argument(
        "--target-name",
        default="me",
        help="確率計算対象の受験者 (default: me)",
    )
    parser.add_argument(
        "--score-range",
        type=int,
        nargs=2,
        default=[0, 10],
        metavar=("MIN", "MAX"),
        help="能力スコアの範囲 (default: 0 10)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=1000,
        help="各スコアでの試行回数 (default: 1000)",
    )

    args = parser.parse_args()

    if args.type == "simple":
        run_simple_mode(
            noise_std=args.noise_std,
            random_seed=args.seed,
            labs_csv=args.labs,
            applicants_csv=args.applicants,
        )
    elif args.type == "probability":
        run_probability_mode(
            noise_std=args.noise_std,
            labs_csv=args.labs,
            applicants_csv=args.applicants,
            target_lab=args.target_lab,
            target_name=args.target_name,
            ability_score_range=tuple(args.score_range),
            num_trials=args.trials,
        )


if __name__ == "__main__":
    main()

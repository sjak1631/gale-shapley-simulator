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
import itertools
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


# ラベル → ベーススコアの対応
LABEL_BASE_SCORES: dict[str, float] = {
    "HIGH":   20.0,
    "MEDIUM": 10.0,
    "LOW":     0.0,
}


def load_applicants_labeled(path: Path) -> dict:
    """
    ability_label 列（HIGH/MEDIUM/LOW）を持つ CSV を読み込む。
    内部では ability_score に変換せず、ラベルのまま保持する。
    列: 氏名, ability_label, 第1志望, 第2志望, ... (志望列は可変)
    """
    applicants = {}
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["氏名"].strip()
            label = row["ability_label"].strip().upper()
            if label not in LABEL_BASE_SCORES:
                raise ValueError(
                    f"{name} の ability_label '{label}' が無効です。"
                    f" HIGH / MEDIUM / LOW のいずれかを指定してください。"
                )
            prefs = [
                row[col].strip()
                for col in reader.fieldnames
                if col not in ("氏名", "ability_label") and row[col] and row[col].strip()
            ]
            applicants[name] = {"ability_label": label, "preferences": prefs}
    return applicants


def generate_test_scores_from_labels(
    applicants: dict,
    noise_std: float,
    rng: "np.random.Generator",
) -> dict[str, float]:
    """
    ability_label をベーススコアに変換し、ガウスノイズを加えてテストスコアを生成する。
    HIGH=20 / MEDIUM=10 / LOW=0 をベースにノイズを付与。
    """
    return {
        name: float(LABEL_BASE_SCORES[data["ability_label"]] + rng.normal(0, noise_std))
        for name, data in applicants.items()
    }


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


def run_permutation_mode(
    noise_std: float,
    labs_csv: Path,
    applicants_csv: Path,
    target_name: str,
    ability_score_range: tuple[int, int],
    num_trials: int,
) -> None:
    """
    志望リストの順列に対する内定確率計算モード。
    1位と2位は固定し、3位以降の4つの研究室の順列すべてについて
    スコアレンジでの内定確率を計算。配属先の研究室も記録。
    """
    from copy import deepcopy
    
    applicants = load_applicants(applicants_csv)
    labs = load_labs(labs_csv)

    if target_name not in applicants:
        print(f"エラー: {target_name} が applicants.csv に見つかりません")
        return

    # meの現在の志望
    original_prefs = applicants[target_name]["preferences"]
    
    # 1位と2位（固定）
    if len(original_prefs) < 2:
        print(f"エラー: {target_name} の志望が2位以上ありません")
        return
    
    fixed_labs = original_prefs[:2]
    
    # 3位以降（順列対象）
    remaining_labs = original_prefs[2:]
    
    if len(remaining_labs) < 4:
        print(f"エラー: {target_name} の3位以降の志望が4つ未満です")
        return

    remaining_labs = remaining_labs[:4]  # 最初の4つを使用

    min_score, max_score = ability_score_range

    print(f"【志望順列による内定確率分析】")
    print(f"  対象者: {target_name}")
    print(f"  固定志望: 1位={fixed_labs[0]}, 2位={fixed_labs[1]}")
    print(f"  順列対象: {', '.join(remaining_labs)}")
    print(f"  試行回数: {num_trials} 回/スコア")
    print(f"  スコア範囲: {min_score}～{max_score}")
    print(f"  ノイズ標準偏差: {noise_std}")
    print()

    # 24通りの順列を生成
    all_permutations = list(itertools.permutations(remaining_labs))
    print(f"総順列数: {len(all_permutations)}")
    print("=" * 100)

    # 結果を保存（ディクショナリ）
    # {(順列): {スコア: {配属先: 回数}}}
    all_results = {}

    for perm_idx, perm in enumerate(all_permutations, 1):
        # 志望リストを構築
        new_prefs = list(fixed_labs) + list(perm)
        
        # 計算用の applicants を深くコピー
        test_applicants = deepcopy(applicants)
        test_applicants[target_name]["preferences"] = new_prefs
        
        results_for_perm = {}
        
        for test_score in range(min_score, max_score + 1):
            test_applicants[target_name]["ability_score"] = float(test_score)
            lab_distribution = {}  # {配属先: 回数}

            for seed in range(num_trials):
                test_scores = generate_test_scores(test_applicants, noise_std, seed)
                lab_rank_index = build_lab_rank_index(labs, test_scores)
                matching = gale_shapley(test_applicants, labs, lab_rank_index)

                # 配属先を記録
                matched_lab = matching.get(target_name)
                if matched_lab is not None:
                    if matched_lab not in lab_distribution:
                        lab_distribution[matched_lab] = 0
                    lab_distribution[matched_lab] += 1
                else:
                    if "未配属" not in lab_distribution:
                        lab_distribution["未配属"] = 0
                    lab_distribution["未配属"] += 1

            results_for_perm[test_score] = lab_distribution

        all_results[tuple(new_prefs)] = results_for_perm
        
        # 進捗表示
        if perm_idx % 6 == 0 or perm_idx == len(all_permutations):
            print(f"  進捗: {perm_idx}/{len(all_permutations)} 完了")

    print()
    print("=" * 100)
    print("【結果詳細 - 配属先別集計】")
    print("=" * 100)

    # 結果をテーブル形式で表示（配属先情報付き）
    for perm_idx, (prefs, results) in enumerate(all_results.items(), 1):
        perm_str = " → ".join(prefs[2:])
        print(f"\n順列 {perm_idx}: {perm_str}")
        print("=" * 100)
        
        for test_score in range(min_score, max_score + 1):
            lab_dist = results.get(test_score, {})
            print(f"\n  スコア {test_score}:")
            print(f"  {'-' * 80}")
            
            if not lab_dist:
                print("    データなし")
                continue
            
            # 研究室ごとに確率を計算して表示
            for lab_name in sorted(lab_dist.keys()):
                count = lab_dist[lab_name]
                prob = count / num_trials * 100
                bar_length = int(prob / 2)
                bar = "█" * bar_length + "░" * (50 - bar_length)
                print(f"    {lab_name:<15} | {bar} | {count:4d}/{num_trials} ({prob:6.2f}%)")

    print()
    print("=" * 100)
    print("【スコア別 配属先一覧】")
    print("=" * 100)

    # スコア別に全順列の配属先を一覧表示
    for score in range(min_score, max_score + 1):
        print(f"\n【スコア {score}】")
        print("-" * 120)
        print(f"{'順列':<50}", end="")
        
        # 全研究室を列挙
        all_labs_set = set()
        for prefs, results in all_results.items():
            if score in results:
                all_labs_set.update(results[score].keys())
        all_labs_list = sorted(all_labs_set)
        
        for lab in all_labs_list:
            print(f" | {lab:<12}", end="")
        print(" |")
        print("-" * 120)
        
        for perm_idx, (prefs, results) in enumerate(all_results.items(), 1):
            perm_str = " → ".join(prefs[2:])
            print(f"{perm_str:<50}", end="")
            
            lab_dist = results.get(score, {})
            for lab in all_labs_list:
                count = lab_dist.get(lab, 0)
                prob = count / num_trials * 100
                print(f" | {prob:11.2f}%", end="")
            print(" |")

    print()
    print("=" * 100)
    print("【CSV出力用データ】")
    print("=" * 100)
    
    # CSV形式で出力（配属先ごと）
    for score in range(min_score, max_score + 1):
        print(f"\n【スコア {score} の詳細データ】")
        print("順列ID,志望順序,", end="")
        
        # 全研究室を列挙
        all_labs_set = set()
        for prefs, results in all_results.items():
            if score in results:
                all_labs_set.update(results[score].keys())
        all_labs_list = sorted(all_labs_set)
        
        for lab in all_labs_list:
            print(f"{lab}(%),", end="")
        print()
        
        for perm_idx, (prefs, results) in enumerate(all_results.items(), 1):
            perm_str = " → ".join(prefs[2:])
            print(f"{perm_idx},{perm_str},", end="")
            
            lab_dist = results.get(score, {})
            for lab in all_labs_list:
                count = lab_dist.get(lab, 0)
                prob = count / num_trials * 100
                print(f"{prob:.2f},", end="")
            print()


def build_random_lab_rank_index(
    labs: dict,
    applicants: dict,
    rng: "np.random.Generator",
) -> dict[str, dict[str, int]]:
    """
    全研究室で共通のランダム順位を生成する。
    能力スコア・テストスコアを一切使わない。
    """
    all_applicants = list(applicants.keys())
    shuffled = list(all_applicants)
    rng.shuffle(shuffled)
    rank_index = {a: i for i, a in enumerate(shuffled)}
    return {lab: rank_index for lab in labs}


def run_labeled_mode(
    labs_csv: Path,
    applicants_csv: Path,
    target_lab: str,
    num_trials: int,
    seed: Optional[int],
    noise_std: float,
    top_n: int,
) -> None:
    """
    ability_label（HIGH/MEDIUM/LOW）ベースの複数回シミュレーションモード。

    ラベルをベーススコアに変換しガウスノイズを加えてテストスコアを生成することで、
    「ラベル間の大まかな実力差を反映しつつ、現実的なランダム性」を再現する。
    指定した研究室の内定頻度が高い受験者上位を表示する。
    """
    applicants = load_applicants_labeled(applicants_csv)
    labs = load_labs(labs_csv)

    if target_lab not in labs:
        print(f"エラー: '{target_lab}' が labs.csv に見つかりません")
        print(f"利用可能な研究室: {', '.join(labs.keys())}")
        return

    capacity = labs[target_lab]
    rng = np.random.default_rng(seed)

    print("【ラベルスコアベース シミュレーション】")
    print(f"  対象研究室     : {target_lab}  (定員 {capacity} 名)")
    print(f"  試行回数       : {num_trials} 回")
    print(f"  ノイズ標準偏差 : {noise_std}  (HIGH=20 / MEDIUM=10 / LOW=0 にノイズを付与)")
    print(f"  乱数シード     : {seed}")
    print()

    acceptance_count: dict[str, int] = {a: 0 for a in applicants}

    for _ in range(num_trials):
        test_scores = generate_test_scores_from_labels(applicants, noise_std, rng)
        lab_rank_index = build_lab_rank_index(labs, test_scores)
        matching = gale_shapley(applicants, labs, lab_rank_index)
        for applicant, matched_lab in matching.items():
            if matched_lab == target_lab:
                acceptance_count[applicant] += 1

    sorted_results = sorted(
        acceptance_count.items(), key=lambda x: x[1], reverse=True
    )

    display = [(a, c) for a, c in sorted_results if c > 0]
    if top_n > 0:
        display = display[:top_n]

    SEP = "=" * 72
    print(f"\n{SEP}")
    print(f"  {target_lab}  最終内定状況ランキング  ({num_trials} 回シミュレーション)")
    print(SEP)
    header = f"{'順位':>4}  {'受験者':<12}  {'ラベル':>6}  {'内定回数':>8}  {'内定率':>7}  {'志望順位':>8}"
    print(header)
    print("-" * 72)

    for rank, (applicant, count) in enumerate(display, 1):
        prob = count / num_trials * 100
        label = applicants[applicant]["ability_label"]
        prefs = applicants[applicant]["preferences"]
        if target_lab in prefs:
            pref_rank = prefs.index(target_lab) + 1
            rank_str = f"第{pref_rank}志望"
        else:
            rank_str = "志望外"
        bar_len = int(prob / 2)
        bar = "█" * bar_len
        print(
            f"{rank:>4}  {applicant:<12}  {label:>6}  {count:>8}  {prob:>6.2f}%  {rank_str:>8}  {bar}"
        )

    if not display:
        print("  (内定者なし)")

    total_accepted = sum(acceptance_count.values())
    avg = total_accepted / num_trials
    print(SEP)
    print(f"  1試行あたり平均内定者数: {avg:.2f} 名  (定員: {capacity} 名)")
    print(SEP)


def run_random_mode(
    labs_csv: Path,
    applicants_csv: Path,
    target_lab: str,
    num_trials: int,
    seed: Optional[int],
    top_n: int,
) -> None:
    """
    完全ランダムランキングによる複数回シミュレーションモード。

    能力スコアを無視し、各試行・各研究室ごとに独立したランダム順位を生成。
    指定した研究室の内定頻度が高い受験者上位を表示する。
    """
    applicants = load_applicants(applicants_csv)
    labs = load_labs(labs_csv)

    if target_lab not in labs:
        print(f"エラー: '{target_lab}' が labs.csv に見つかりません")
        print(f"利用可能な研究室: {', '.join(labs.keys())}")
        return

    capacity = labs[target_lab]
    rng = np.random.default_rng(seed)

    print("【完全ランダムシミュレーション】")
    print(f"  対象研究室 : {target_lab}  (定員 {capacity} 名)")
    print(f"  試行回数   : {num_trials} 回")
    print(f"  乱数シード : {seed}")
    print()

    # 各受験者が target_lab に内定した回数
    acceptance_count: dict[str, int] = {a: 0 for a in applicants}

    for _ in range(num_trials):
        lab_rank_index = build_random_lab_rank_index(labs, applicants, rng)
        matching = gale_shapley(applicants, labs, lab_rank_index)
        for applicant, matched_lab in matching.items():
            if matched_lab == target_lab:
                acceptance_count[applicant] += 1

    # 内定頻度の高い順にソート
    sorted_results = sorted(
        acceptance_count.items(), key=lambda x: x[1], reverse=True
    )

    display = [(a, c) for a, c in sorted_results if c > 0]
    if top_n > 0:
        display = display[:top_n]

    SEP = "=" * 65
    print(f"\n{SEP}")
    print(f"  {target_lab}  最終内定状況ランキング  ({num_trials} 回シミュレーション)")
    print(SEP)
    header = f"{'順位':>4}  {'受験者':<12}  {'内定回数':>8}  {'内定率':>7}  {'志望順位':>8}"
    print(header)
    print("-" * 65)

    for rank, (applicant, count) in enumerate(display, 1):
        prob = count / num_trials * 100
        prefs = applicants[applicant]["preferences"]
        if target_lab in prefs:
            pref_rank = prefs.index(target_lab) + 1
            rank_str = f"第{pref_rank}志望"
        else:
            rank_str = "志望外"
        bar_len = int(prob / 2)
        bar = "█" * bar_len
        print(
            f"{rank:>4}  {applicant:<12}  {count:>8}  {prob:>6.2f}%  {rank_str:>8}  {bar}"
        )

    if not display:
        print("  (内定者なし)")

    total_accepted = sum(acceptance_count.values())
    avg = total_accepted / num_trials
    print(SEP)
    print(f"  1試行あたり平均内定者数: {avg:.2f} 名  (定員: {capacity} 名)")
    print(SEP)


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

  # 志望順列分析（meの志望順序の順列分析）
  python3 simulation.py --type permutation --target-name me --score-range 0 10 --trials 1000

  # 完全ランダムシミュレーション（lab_A研の内定頻度上位10名）
  python3 simulation.py --type random --target-lab lab_A --trials 10000 --top 10

  # ラベルスコアベースシミュレーション（lab_B研, labeled CSV使用）
  python3 simulation.py --type labeled --target-lab lab_B \\
    --applicants my_applicants_labeled.csv --labs my_labs.csv --trials 10000 --top 10

  # カスタム CSV パスとノイズ
  python3 simulation.py --type simple --noise-std 2.0 --seed 123 \\
    --applicants custom_applicants.csv --labs custom_labs.csv
        """,
    )

    # 共通引数
    parser.add_argument(
        "--type",
        choices=["simple", "probability", "permutation", "random", "labeled"],
        default="simple",
        help="実行モード: simple / probability / permutation / random / labeled(ラベルスコードシミュレーション)",
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

    # probability/permutation モード専用引数
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

    parser.add_argument(
        "--top",
        type=int,
        default=0,
        help="random モードで表示する上位 N 名 (0 = 全員, default: 0)",
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
    elif args.type == "permutation":
        run_permutation_mode(
            noise_std=args.noise_std,
            labs_csv=args.labs,
            applicants_csv=args.applicants,
            target_name=args.target_name,
            ability_score_range=tuple(args.score_range),
            num_trials=args.trials,
        )
    elif args.type == "random":
        run_random_mode(
            labs_csv=args.labs,
            applicants_csv=args.applicants,
            target_lab=args.target_lab,
            num_trials=args.trials,
            seed=args.seed,
            top_n=args.top,
        )
    elif args.type == "labeled":
        run_labeled_mode(
            labs_csv=args.labs,
            applicants_csv=args.applicants,
            target_lab=args.target_lab,
            num_trials=args.trials,
            seed=args.seed,
            noise_std=args.noise_std,
            top_n=args.top,
        )


if __name__ == "__main__":
    main()

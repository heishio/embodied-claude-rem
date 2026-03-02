"""Sleep-like replay and consolidation routines."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from math import ceil
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .graph import MemoryGraph
    from .memory import MemoryStore
    from .types import Memory

# ── Bias Update Constants ──────────────────────
BIAS_ACCUMULATION_RATE = 0.01    # バイアス蓄積レート（控えめ）
BIAS_MAX_CAP = 0.15              # 単一バイアスの上限
BIAS_DECAY_FACTOR = 0.95         # コンソリデーション毎の減衰率
BIAS_PRUNE_THRESHOLD = 0.001    # この値以下のバイアスを刈り取り
BIAS_APPLY_COEFFICIENT = 0.05   # recall時のノイズ適用係数


@dataclass(frozen=True)
class ConsolidationStats:
    """Summary statistics for replay execution."""

    replay_events: int
    coactivation_updates: int
    link_updates: int
    refreshed_memories: int

    def to_dict(self) -> dict[str, int]:
        return {
            "replay_events": self.replay_events,
            "coactivation_updates": self.coactivation_updates,
            "link_updates": self.link_updates,
            "refreshed_memories": self.refreshed_memories,
        }


class ConsolidationEngine:
    """Replay memories and update association strengths."""

    async def run(
        self,
        store: "MemoryStore",
        window_hours: int = 24,
        max_replay_events: int = 200,
        link_update_strength: float = 0.2,
    ) -> ConsolidationStats:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=max(1, window_hours))
        recent = await store.list_recent(limit=max(max_replay_events * 2, 50))
        recent = [m for m in recent if self._is_after(m, cutoff)]

        if len(recent) < 2:
            return ConsolidationStats(0, 0, 0, len(recent))

        replay_events = 0
        coactivation_updates = 0
        link_updates = 0
        refreshed_ids: set[str] = set()

        for idx in range(len(recent) - 1):
            if replay_events >= max_replay_events:
                break

            left = recent[idx]
            right = recent[idx + 1]

            delta = max(0.05, min(1.0, link_update_strength))
            await store.bump_coactivation(left.id, right.id, delta=delta)
            coactivation_updates += 2

            left_error = max(0.0, left.prediction_error * 0.9)
            right_error = max(0.0, right.prediction_error * 0.9)
            await store.record_activation(left.id, prediction_error=left_error)
            await store.record_activation(right.id, prediction_error=right_error)
            refreshed_ids.add(left.id)
            refreshed_ids.add(right.id)

            if await store.maybe_add_related_link(left.id, right.id, threshold=0.6):
                link_updates += 1

            replay_events += 1

        # Consolidate freshness — sleep creates temporal distance
        await store.consolidate_freshness()

        return ConsolidationStats(
            replay_events=replay_events,
            coactivation_updates=coactivation_updates,
            link_updates=link_updates,
            refreshed_memories=len(refreshed_ids),
        )

    async def synthesize_composites(
        self,
        store: "MemoryStore",
        similarity_threshold: float = 0.75,
        min_group_size: int = 2,
        max_group_size: int = 8,
        source_level: int = 0,
        target_level: int = 1,
    ) -> dict[str, int]:
        """指定 source_level の記憶をグループ化し、target_level の合成記憶を生成する。"""
        # 1. 対象取得
        mem_vecs = await store.fetch_memories_with_vectors_by_level(
            level=source_level, min_freshness=0.1,
        )
        if len(mem_vecs) < min_group_size:
            return {"composites_created": 0, "composites_skipped": 0}

        memories = [mv[0] for mv in mem_vecs]
        vectors = np.array([mv[1] for mv in mem_vecs])

        # 2. 類似度計算（コサイン類似度行列）
        norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10
        normalized = vectors / norms
        sim_matrix = normalized @ normalized.T

        # 3. Union-Find でグループ化
        n = len(memories)
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        for i in range(n):
            for j in range(i + 1, n):
                if sim_matrix[i, j] >= similarity_threshold:
                    union(i, j)

        # グループ収集
        groups: dict[int, list[int]] = {}
        for i in range(n):
            root = find(i)
            if root not in groups:
                groups[root] = []
            groups[root].append(i)

        # min_group_size 未満を除外
        groups = {k: v for k, v in groups.items() if len(v) >= min_group_size}

        # max_group_size を超えたら最も相互類似度が高いサブセットに絞る
        for root, indices in list(groups.items()):
            if len(indices) > max_group_size:
                # 各ペアの類似度合計でスコアを計算し、上位を残す
                scores: list[tuple[float, int]] = []
                for idx in indices:
                    s = sum(sim_matrix[idx, j] for j in indices if j != idx)
                    scores.append((s, idx))
                scores.sort(reverse=True)
                groups[root] = [idx for _, idx in scores[:max_group_size]]

        # 4. 既存チェック
        existing = await store.get_existing_composite_members()
        composites_created = 0
        composites_skipped = 0

        for indices in groups.values():
            member_ids = frozenset(memories[i].id for i in indices)
            if member_ids in existing:
                composites_skipped += 1
                continue

            # 5. 合成ベクトル生成（importance で重み付けた加重平均 → 正規化）
            weights = np.array([float(memories[i].importance) for i in indices])
            member_vectors = np.array([vectors[i] for i in indices])
            weighted_sum = (member_vectors.T @ weights).T
            norm = np.linalg.norm(weighted_sum) + 1e-10
            composite_vector = weighted_sum / norm

            # 6. メタ情報の集約
            emotions = [memories[i].emotion for i in indices]
            emotion_counter = Counter(emotions)
            composite_emotion = emotion_counter.most_common(1)[0][0]

            importances = [memories[i].importance for i in indices]
            composite_importance = min(5, ceil(sum(importances) / len(importances)))

            freshnesses = [memories[i].freshness for i in indices]
            composite_freshness = max(freshnesses)

            categories = [memories[i].category for i in indices]
            category_counter = Counter(categories)
            composite_category = category_counter.most_common(1)[0][0]

            # 7. PCA: 主成分軸の計算
            pca_result = self._compute_principal_axis(member_vectors)
            axis_vector = pca_result[0] if pca_result else None
            explained_var = pca_result[1] if pca_result else None

            # 8. 保存
            await store.save_composite(
                member_ids=list(member_ids),
                vector=composite_vector,
                emotion=composite_emotion,
                importance=composite_importance,
                freshness=composite_freshness,
                category=composite_category,
                axis_vector=axis_vector,
                explained_variance_ratio=explained_var,
                level=target_level,
            )
            composites_created += 1

        return {
            "composites_created": composites_created,
            "composites_skipped": composites_skipped,
        }

    async def compute_boundary_layers(
        self,
        store: "MemoryStore",
        graph: "MemoryGraph | None" = None,
        n_layers: int = 3,
        noise_scale: float = 0.1,
        max_template_strength: float = 0.3,
    ) -> dict[str, int]:
        """外縁検出 + ノイズレイヤー生成 + バイアス蓄積。

        各 composite について:
        - Layer 0: ノイズなしで edge/core を分類
        - Layer 1..n_layers: 体験チェーンベースのノイズで edge/core を再分類

        Returns:
            {"composites_processed": int, "total_layers": int,
             "biases_updated": int, "biases_decayed": int, "biases_pruned": int}
        """
        composite_ids = await store.fetch_all_composite_ids()
        if not composite_ids:
            return {
                "composites_processed": 0, "total_layers": 0,
                "biases_updated": 0, "biases_decayed": 0, "biases_pruned": 0,
            }

        # 1. バイアス減衰（蓄積より先に行う）
        decay_stats = await self._decay_biases(store)

        # 2. 既存バイアスを取得
        existing_biases = await store.fetch_template_biases()

        # テンプレート生成（全composite共通） — 3-tuple: (vec, transient_strength, bias_weight)
        templates: list[tuple[np.ndarray, float, float, str]] = []  # (vec, transient, bias, chain_id)
        if graph is not None:
            raw_templates = await store.fetch_verb_chain_templates()
            for chain_id, vec, verbs, nouns in raw_templates:
                raw_strength = await graph.get_path_strength(verbs, nouns)
                strength = min(raw_strength, max_template_strength)
                if strength > 0:
                    bias = existing_biases.get(chain_id, 0.0)
                    templates.append((vec, strength, bias, chain_id))

        await store.clear_boundary_layers()

        # 全compositeの主成分軸を取得（異方的エッジ判定用）
        all_axes = await store.fetch_all_composite_axes()

        composites_processed = 0
        total_layers = 0
        # 各compositeのlayer0分類を記録（バイアス更新用）
        composite_layer0_map: dict[str, list[int]] = {}

        for cid in composite_ids:
            members = await store.fetch_composite_with_vectors(cid)
            if len(members) < 2:
                continue

            centroid = await store.fetch_composite_centroid(cid)
            if centroid is None:
                continue

            member_ids = [m[0] for m in members]
            member_vecs = np.array([m[1] for m in members])
            axis_vec = all_axes.get(cid)  # None if no axis computed

            # ── Layer 0: ノイズなし（異方的判定） ──
            layer0 = self._classify_edge_core(member_vecs, centroid, axis_vec)
            composite_layer0_map[cid] = layer0
            all_layers: list[tuple[str, int, int]] = []
            for i, mid in enumerate(member_ids):
                all_layers.append((mid, 0, layer0[i]))

            # ── Layer 1..n_layers: ノイズ適用 ──
            layer_classifications: list[list[int]] = []
            for layer_idx in range(1, n_layers + 1):
                noised_vecs = self._apply_noise(
                    member_vecs, templates, noise_scale, layer_idx,
                )
                # 新しい重心を計算
                noised_centroid = noised_vecs.mean(axis=0)
                norm = np.linalg.norm(noised_centroid) + 1e-10
                noised_centroid = noised_centroid / norm

                layer_classes = self._classify_edge_core(noised_vecs, noised_centroid, axis_vec)
                layer_classifications.append(layer_classes)
                for i, mid in enumerate(member_ids):
                    all_layers.append((mid, layer_idx, layer_classes[i]))

            await store.save_boundary_layers(cid, all_layers)
            composites_processed += 1
            total_layers += 1 + n_layers

        # 3. バイアス蓄積
        biases_updated = await self._update_biases(
            store, templates, composite_layer0_map, n_layers, existing_biases,
        )

        return {
            "composites_processed": composites_processed,
            "total_layers": total_layers,
            "biases_updated": biases_updated,
            "biases_decayed": decay_stats.get("biases_decayed", 0),
            "biases_pruned": decay_stats.get("biases_pruned", 0),
        }

    def _classify_edge_core(
        self,
        member_vecs: np.ndarray,
        centroid: np.ndarray,
        axis_vector: np.ndarray | None = None,
    ) -> list[int]:
        """メンバーを重心からの距離で edge(1) / core(0) に分類。

        axis_vectorがある場合、異方的距離を使用:
        - 軸方向の距離は重み小（伸びてる方向なので自然）
        - 垂直方向の距離は重み大（はみ出してる＝本当のedge）
        """
        c_norm = centroid / (np.linalg.norm(centroid) + 1e-10)
        m_norms = member_vecs / (np.linalg.norm(member_vecs, axis=1, keepdims=True) + 1e-10)

        if axis_vector is None:
            # 等方的: コサイン距離
            similarities = m_norms @ c_norm
            distances = 1.0 - similarities
        else:
            # 異方的: 軸方向と垂直方向で重み分離
            a_norm = axis_vector / (np.linalg.norm(axis_vector) + 1e-10)

            # 各メンバーの重心からの差分ベクトル
            diffs = m_norms - c_norm
            # 軸方向成分
            axial_proj = np.outer(diffs @ a_norm, a_norm)  # (n, d)
            # 垂直成分
            perp = diffs - axial_proj

            # 軸方向の距離（重み小: 0.3）と垂直方向の距離（重み大: 1.0）
            axial_dist = np.linalg.norm(axial_proj, axis=1)
            perp_dist = np.linalg.norm(perp, axis=1)
            distances = 0.3 * axial_dist + 1.0 * perp_dist

        d_mean = float(np.mean(distances))
        return [1 if float(d) > d_mean else 0 for d in distances]

    def _apply_noise(
        self,
        member_vecs: np.ndarray,
        templates: list[tuple[np.ndarray, float, float, str]],
        noise_scale: float,
        seed: int,
        max_template_strength: float = 0.3,
    ) -> np.ndarray:
        """テンプレートベースのノイズを適用。テンプレートがなければランダムノイズ。

        templates: list of (vec, transient_strength, bias_weight, chain_id)
        effective_strength = min(transient + BIAS_APPLY_COEFFICIENT * bias, max_template_strength)
        """
        rng = np.random.default_rng(seed)
        n_members = member_vecs.shape[0]
        dim = member_vecs.shape[1]

        if not templates:
            # フォールバック: ランダムノイズ
            noise = rng.normal(0, noise_scale, size=(n_members, dim)).astype(np.float32)
            noised = member_vecs + noise
            norms = np.linalg.norm(noised, axis=1, keepdims=True) + 1e-10
            return noised / norms

        # テンプレートベースのノイズ
        noised = member_vecs.copy()
        m_norms = member_vecs / (np.linalg.norm(member_vecs, axis=1, keepdims=True) + 1e-10)

        for t_vec, transient_strength, bias_weight, _chain_id in templates:
            effective_strength = min(
                transient_strength + BIAS_APPLY_COEFFICIENT * bias_weight,
                max_template_strength,
            )
            alpha = rng.normal(0, 1)
            # Pad template vector to match member dimension if needed
            if len(t_vec) < dim:
                t_padded = np.zeros(dim, dtype=t_vec.dtype)
                t_padded[:len(t_vec)] = t_vec
                t_vec = t_padded
            elif len(t_vec) > dim:
                t_vec = t_vec[:dim]
            t_norm = t_vec / (np.linalg.norm(t_vec) + 1e-10)
            alignments = m_norms @ t_norm  # shape: (n_members,)
            for i in range(n_members):
                noised[i] += effective_strength * alpha * float(alignments[i]) * t_norm * noise_scale

        # 正規化
        norms = np.linalg.norm(noised, axis=1, keepdims=True) + 1e-10
        return noised / norms

    async def _decay_biases(self, store: "MemoryStore") -> dict[str, int]:
        """バイアスを減衰・刈り取り。"""
        return await store.decay_template_biases(BIAS_DECAY_FACTOR, BIAS_PRUNE_THRESHOLD)

    async def _update_biases(
        self,
        store: "MemoryStore",
        templates: list[tuple[np.ndarray, float, float, str]],
        composite_layer0_map: dict[str, list[int]],
        n_layers: int,
        existing_biases: dict[str, float],
    ) -> int:
        """各テンプレートのバイアスを蓄積する。

        composite毎にlayer 0 vs layer k の分類変化（ハミング距離 / メンバー数）を計算し、
        テンプレートのtransient_strengthの割合で按分して蓄積。
        """
        if not templates or not composite_layer0_map or n_layers == 0:
            return 0

        db = store._ensure_connected()

        # 全compositeの平均シフトを計算
        overall_shifts: list[float] = []
        for cid, layer0 in composite_layer0_map.items():
            n_members = len(layer0)
            if n_members == 0:
                continue
            # 各レイヤーのlayer0との分類差を取得
            for layer_idx in range(1, n_layers + 1):
                rows = db.execute(
                    """SELECT member_id, is_edge FROM boundary_layers
                       WHERE composite_id = ? AND layer_index = ?
                       ORDER BY member_id""",
                    (cid, layer_idx),
                ).fetchall()
                if len(rows) != n_members:
                    continue
                layer0_rows = db.execute(
                    """SELECT member_id, is_edge FROM boundary_layers
                       WHERE composite_id = ? AND layer_index = 0
                       ORDER BY member_id""",
                    (cid,),
                ).fetchall()
                # ハミング距離
                hamming = sum(
                    1 for l0, lk in zip(layer0_rows, rows)
                    if l0["is_edge"] != lk["is_edge"]
                )
                shift = hamming / n_members
                overall_shifts.append(shift)

        if not overall_shifts:
            return 0

        avg_shift = sum(overall_shifts) / len(overall_shifts)

        # テンプレート毎のバイアス更新
        sum_strengths = sum(t[1] for t in templates)
        if sum_strengths <= 0:
            return 0

        bias_updates: list[tuple[str, float, int]] = []
        for _vec, transient_strength, bias_weight, chain_id in templates:
            proportion = transient_strength / sum_strengths
            shift_contribution = proportion * avg_shift
            old_bias = existing_biases.get(chain_id, 0.0)
            new_bias = min(BIAS_MAX_CAP, old_bias + BIAS_ACCUMULATION_RATE * shift_contribution)
            # update_count を既存の値から取得
            row = db.execute(
                "SELECT update_count FROM template_biases WHERE chain_id = ?",
                (chain_id,),
            ).fetchone()
            old_count = row["update_count"] if row else 0
            bias_updates.append((chain_id, new_bias, old_count + 1))

        await store.save_template_biases(bias_updates)
        return len(bias_updates)

    def _compute_principal_axis(
        self, member_vecs: np.ndarray
    ) -> tuple[np.ndarray, float] | None:
        """メンバーベクトルの第一主成分と寄与率を返す。

        Returns:
            (axis_vector, explained_variance_ratio) or None if < 2 members.
        """
        if member_vecs.shape[0] < 2:
            return None

        # 中心化
        centroid = member_vecs.mean(axis=0)
        centered = member_vecs - centroid

        # SVD（経済的: 主成分1つだけ必要）
        try:
            _, s, vt = np.linalg.svd(centered, full_matrices=False)
        except np.linalg.LinAlgError:
            return None

        axis = vt[0]  # 第一主成分方向
        # 正規化
        norm = np.linalg.norm(axis)
        if norm < 1e-10:
            return None
        axis = axis / norm

        # 寄与率
        total_var = float(np.sum(s ** 2))
        if total_var < 1e-10:
            return None
        explained = float(s[0] ** 2) / total_var

        return axis.astype(np.float32), explained

    async def detect_overlap(
        self,
        store: "MemoryStore",
        overlap_threshold: float = 0.5,
    ) -> dict[str, int]:
        """composite 間の重なりを検出し、二重所属メンバーを追加する。

        1. 全 composite の重心ベクトルを取得
        2. 重心間コサイン類似度を計算
        3. 類似度 > overlap_threshold のペアについてメンバーの二重所属を検出・追加
        """
        composite_ids = await store.fetch_all_composite_ids()
        if len(composite_ids) < 2:
            return {"overlap_pairs": 0, "dual_members_added": 0}

        # 各 composite の重心とメンバーを取得
        centroids: dict[str, np.ndarray] = {}
        member_vecs_map: dict[str, list[tuple[str, np.ndarray]]] = {}
        existing_members: dict[str, set[str]] = {}

        member_sets = await store.fetch_all_composite_member_sets()

        for cid in composite_ids:
            centroid = await store.fetch_composite_centroid(cid)
            if centroid is None:
                continue
            centroids[cid] = centroid
            existing_members[cid] = member_sets.get(cid, set())

            members = await store.fetch_composite_with_vectors(cid)
            member_vecs_map[cid] = [(m[0], m[1]) for m in members]

        cids_with_centroid = list(centroids.keys())
        overlap_pairs = 0
        dual_members_added = 0

        db = store._ensure_connected()

        for i in range(len(cids_with_centroid)):
            for j in range(i + 1, len(cids_with_centroid)):
                cid_a = cids_with_centroid[i]
                cid_b = cids_with_centroid[j]
                centroid_a = centroids[cid_a]
                centroid_b = centroids[cid_b]

                sim = float(np.dot(centroid_a, centroid_b) / (
                    np.linalg.norm(centroid_a) * np.linalg.norm(centroid_b) + 1e-10
                ))
                if sim <= overlap_threshold:
                    continue

                overlap_pairs += 1

                # A のメンバーが B の重心に近いか確認
                for mid, mvec in member_vecs_map.get(cid_a, []):
                    if mid in existing_members.get(cid_b, set()):
                        continue
                    msim = float(np.dot(mvec, centroid_b) / (
                        np.linalg.norm(mvec) * np.linalg.norm(centroid_b) + 1e-10
                    ))
                    if msim > overlap_threshold:
                        db.execute(
                            """INSERT OR IGNORE INTO composite_members
                               (composite_id, member_id, contribution_weight)
                               VALUES (?, ?, ?)""",
                            (cid_b, mid, 0.5),
                        )
                        dual_members_added += 1

                # B のメンバーが A の重心に近いか確認
                for mid, mvec in member_vecs_map.get(cid_b, []):
                    if mid in existing_members.get(cid_a, set()):
                        continue
                    msim = float(np.dot(mvec, centroid_a) / (
                        np.linalg.norm(mvec) * np.linalg.norm(centroid_a) + 1e-10
                    ))
                    if msim > overlap_threshold:
                        db.execute(
                            """INSERT OR IGNORE INTO composite_members
                               (composite_id, member_id, contribution_weight)
                               VALUES (?, ?, ?)""",
                            (cid_a, mid, 0.5),
                        )
                        dual_members_added += 1

        if dual_members_added > 0:
            db.commit()

        return {"overlap_pairs": overlap_pairs, "dual_members_added": dual_members_added}

    async def rescue_orphans(
        self,
        store: "MemoryStore",
        rescue_threshold: float = 0.4,
        level: int = 0,
    ) -> dict[str, int]:
        """指定 level でどの composite にも属さない孤立記憶を最寄り composite に吸収する。"""
        orphans = await store.fetch_orphan_memories(level=level, min_freshness=0.1)
        if not orphans:
            return {"orphans_found": 0, "orphans_rescued": 0}

        composite_ids = await store.fetch_all_composite_ids()
        if not composite_ids:
            return {"orphans_found": len(orphans), "orphans_rescued": 0}

        # 各 composite の重心を取得
        centroids: dict[str, np.ndarray] = {}
        for cid in composite_ids:
            centroid = await store.fetch_composite_centroid(cid)
            if centroid is not None:
                centroids[cid] = centroid

        if not centroids:
            return {"orphans_found": len(orphans), "orphans_rescued": 0}

        db = store._ensure_connected()
        orphans_rescued = 0

        for mem, vec in orphans:
            best_cid = None
            best_sim = -1.0
            for cid, centroid in centroids.items():
                sim = float(np.dot(vec, centroid) / (
                    np.linalg.norm(vec) * np.linalg.norm(centroid) + 1e-10
                ))
                if sim > best_sim:
                    best_sim = sim
                    best_cid = cid

            if best_cid is not None and best_sim >= rescue_threshold:
                db.execute(
                    """INSERT OR IGNORE INTO composite_members
                       (composite_id, member_id, contribution_weight)
                       VALUES (?, ?, ?)""",
                    (best_cid, mem.id, 0.3),
                )
                orphans_rescued += 1

        if orphans_rescued > 0:
            db.commit()

        return {"orphans_found": len(orphans), "orphans_rescued": orphans_rescued}

    async def detect_intersections(
        self,
        store: "MemoryStore",
        parallel_threshold: float = 0.8,
        transversal_threshold: float = 0.3,
    ) -> dict[str, int]:
        """全compositeペアの主成分軸を比較し、交差を検出。

        - |cos(angle)| >= parallel_threshold → parallel（同次元重なり）
        - |cos(angle)| <= transversal_threshold かつ共有メンバーあり → transversal（横断的交差）

        Returns:
            {"parallel_found": int, "transversal_found": int, "intersection_nodes": int}
        """
        axes = await store.fetch_all_composite_axes()
        if len(axes) < 2:
            return {"parallel_found": 0, "transversal_found": 0, "intersection_nodes": 0}

        member_sets = await store.fetch_all_composite_member_sets()

        composite_ids = list(axes.keys())
        intersections: list[tuple[str, str, str, float, list[str]]] = []
        intersection_node_ids: set[str] = set()

        for i in range(len(composite_ids)):
            for j in range(i + 1, len(composite_ids)):
                cid_a = composite_ids[i]
                cid_b = composite_ids[j]

                axis_a = axes[cid_a]
                axis_b = axes[cid_b]

                # 軸間のコサイン（絶対値: 方向は問わない）
                cos_angle = float(np.abs(np.dot(axis_a, axis_b)))

                # 共有メンバー
                members_a = member_sets.get(cid_a, set())
                members_b = member_sets.get(cid_b, set())
                shared = members_a & members_b

                if cos_angle >= parallel_threshold:
                    intersections.append((
                        cid_a, cid_b, "parallel", cos_angle, list(shared),
                    ))
                    intersection_node_ids.update(shared)
                elif cos_angle <= transversal_threshold and shared:
                    intersections.append((
                        cid_a, cid_b, "transversal", cos_angle, list(shared),
                    ))
                    intersection_node_ids.update(shared)

        await store.save_intersections(intersections)

        parallel = sum(1 for x in intersections if x[2] == "parallel")
        transversal = sum(1 for x in intersections if x[2] == "transversal")

        return {
            "parallel_found": parallel,
            "transversal_found": transversal,
            "intersection_nodes": len(intersection_node_ids),
        }

    def _is_after(self, memory: "Memory", cutoff: datetime) -> bool:
        try:
            timestamp = datetime.fromisoformat(memory.timestamp)
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
        except ValueError:
            return False
        if cutoff.tzinfo is None:
            cutoff = cutoff.replace(tzinfo=timezone.utc)
        return timestamp >= cutoff

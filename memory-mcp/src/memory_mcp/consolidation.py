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
    refreshed_memories: int

    def to_dict(self) -> dict[str, int]:
        return {
            "replay_events": self.replay_events,
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
            return ConsolidationStats(0, len(recent))

        replay_events = 0
        refreshed_ids: set[str] = set()

        for idx in range(len(recent) - 1):
            if replay_events >= max_replay_events:
                break

            left = recent[idx]
            right = recent[idx + 1]

            left_error = max(0.0, left.prediction_error * 0.9)
            right_error = max(0.0, right.prediction_error * 0.9)
            await store.record_activation(left.id, prediction_error=left_error)
            await store.record_activation(right.id, prediction_error=right_error)
            refreshed_ids.add(left.id)
            refreshed_ids.add(right.id)

            replay_events += 1

        # Consolidate freshness — sleep creates temporal distance
        await store.consolidate_freshness()

        return ConsolidationStats(
            replay_events=replay_events,
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
                idx_arr = np.array(indices)
                group_sims = sim_matrix[np.ix_(idx_arr, idx_arr)]
                np.fill_diagonal(group_sims, 0)
                scores = np.sum(group_sims, axis=1)
                top_k = np.argsort(-scores)[:max_group_size]
                groups[root] = idx_arr[top_k].tolist()

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

    async def synthesize_image_composites(
        self,
        store: "MemoryStore",
        similarity_threshold: float = 0.75,
        min_group_size: int = 2,
        max_group_size: int = 8,
    ) -> dict[str, int]:
        """image_embeddings からクラスタリングし、image_composites を生成する。"""
        # 1. 対象取得（person_ratio >= 0.1）
        image_records = await store.fetch_image_embeddings_for_composites(
            min_person_ratio=0.1, min_freshness=0.1,
        )
        if len(image_records) < min_group_size:
            return {"image_composites_created": 0, "image_composites_skipped": 0}

        # delta_vectorが存在するレコードのみ
        valid = [r for r in image_records if "delta_vector" in r]
        if len(valid) < min_group_size:
            return {"image_composites_created": 0, "image_composites_skipped": 0}

        delta_vecs = np.array([r["delta_vector"] for r in valid])

        # 2. コサイン類似度行列（delta_vector）
        norms = np.linalg.norm(delta_vecs, axis=1, keepdims=True) + 1e-10
        normalized = delta_vecs / norms
        sim_matrix = normalized @ normalized.T

        # 3. Union-Find
        n = len(valid)
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
            groups.setdefault(root, []).append(i)

        groups = {k: v for k, v in groups.items() if len(v) >= min_group_size}

        # max_group_size を超えたら最も相互類似度が高いサブセットに絞る
        for root, indices in list(groups.items()):
            if len(indices) > max_group_size:
                idx_arr = np.array(indices)
                group_sims = sim_matrix[np.ix_(idx_arr, idx_arr)]
                np.fill_diagonal(group_sims, 0)
                scores = np.sum(group_sims, axis=1)
                top_k = np.argsort(-scores)[:max_group_size]
                groups[root] = idx_arr[top_k].tolist()

        # 4. 既存チェック
        existing = await store.get_existing_image_composite_members()
        composites_created = 0
        composites_skipped = 0

        for indices in groups.values():
            member_ids = frozenset(valid[i]["id"] for i in indices)
            if member_ids in existing:
                composites_skipped += 1
                continue

            # 5. delta_centroid = L2正規化した平均ベクトル
            member_deltas = np.array([delta_vecs[i] for i in indices])
            delta_mean = member_deltas.mean(axis=0)
            delta_norm = np.linalg.norm(delta_mean) + 1e-10
            delta_centroid = (delta_mean / delta_norm).astype(np.float32)

            # flow_centroid
            flow_vecs = [valid[i].get("flow_vector") for i in indices]
            flow_valid = [v for v in flow_vecs if v is not None]
            flow_centroid = None
            if flow_valid:
                flow_arr = np.array(flow_valid)
                flow_mean = flow_arr.mean(axis=0)
                flow_norm = np.linalg.norm(flow_mean) + 1e-10
                flow_centroid = (flow_mean / flow_norm).astype(np.float32)

            # face_centroid（face_vectorがあるメンバーのみ）
            face_vecs = [valid[i].get("face_vector") for i in indices]
            face_valid = [v for v in face_vecs if v is not None]
            face_centroid = None
            if face_valid:
                face_arr = np.array(face_valid)
                face_mean = face_arr.mean(axis=0)
                face_norm = np.linalg.norm(face_mean) + 1e-10
                face_centroid = (face_mean / face_norm).astype(np.float32)

            # tag: 多数決
            tags = [valid[i].get("tag") for i in indices]
            tag_counts = Counter(t for t in tags if t)
            tag = tag_counts.most_common(1)[0][0] if tag_counts else None

            # 6. 保存
            await store.save_image_composite(
                member_ids=list(member_ids),
                delta_centroid=delta_centroid,
                flow_centroid=flow_centroid,
                face_centroid=face_centroid,
                tag=tag,
            )
            composites_created += 1

        return {
            "image_composites_created": composites_created,
            "image_composites_skipped": composites_skipped,
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

        # テンプレート生成（composite重心ベース） — graph不要
        templates: list[tuple[np.ndarray, float, float, str]] = []  # (vec, transient, bias, template_id)
        raw_templates = await store.fetch_composite_templates()
        for template_id, vec, strength in raw_templates:
            strength = min(strength, max_template_strength)
            if strength > 0:
                bias = existing_biases.get(template_id, 0.0)
                templates.append((vec, strength, bias, template_id))

        await store.clear_boundary_layers()

        # 全compositeの主成分軸を取得（異方的エッジ判定用）
        all_axes = await store.fetch_all_composite_axes()

        # 全compositeのメンバーと重心を一括取得（N+1クエリ回避）
        all_members = await store.fetch_all_composites_with_vectors()
        all_centroids = await store.fetch_all_composite_centroids()

        composites_processed = 0
        total_layers = 0
        # 各compositeのlayer0分類を記録（バイアス更新用）
        composite_layer0_map: dict[str, list[int]] = {}

        for cid in composite_ids:
            members = all_members.get(cid, [])
            if len(members) < 2:
                continue

            centroid = all_centroids.get(cid)
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
        return (distances > d_mean).astype(int).tolist()

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

        # 正規化: effective_strength合計を上限0.5にスケール
        total_effective = sum(
            min(ts + BIAS_APPLY_COEFFICIENT * bw, max_template_strength)
            for _, ts, bw, _ in templates
        )
        scale_factor = min(1.0, 0.5 / total_effective) if total_effective > 0.5 else 1.0

        for t_vec, transient_strength, bias_weight, _template_id in templates:
            effective_strength = min(
                transient_strength + BIAS_APPLY_COEFFICIENT * bias_weight,
                max_template_strength,
            ) * scale_factor
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
            noised += (effective_strength * alpha * noise_scale) * alignments[:, np.newaxis] * t_norm

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
            # 全レイヤーを一括取得（layer0以外）
            all_rows = db.execute(
                """SELECT layer_index, member_id, is_edge FROM boundary_layers
                   WHERE composite_id = ? AND layer_index > 0
                   ORDER BY layer_index, member_id""",
                (cid,),
            ).fetchall()
            # レイヤー毎にグループ化
            layers_by_idx: dict[int, list] = {}
            for r in all_rows:
                li = r["layer_index"] if not isinstance(r, tuple) else r[0]
                layers_by_idx.setdefault(li, []).append(r)

            for layer_idx in range(1, n_layers + 1):
                rows = layers_by_idx.get(layer_idx, [])
                if len(rows) != n_members:
                    continue
                # layer0はメモリ上のcomposite_layer0_mapを使う（DB再読み込み不要）
                hamming = sum(
                    1 for l0_edge, lk in zip(layer0, rows)
                    if l0_edge != (lk["is_edge"] if not isinstance(lk, tuple) else lk[2])
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

        # update_countを一括取得
        template_ids = [t[3] for t in templates]
        placeholders = ",".join("?" for _ in template_ids)
        count_rows = db.execute(
            f"SELECT template_id, update_count FROM template_biases WHERE template_id IN ({placeholders})",
            template_ids,
        ).fetchall()
        count_map: dict[str, int] = {}
        for r in count_rows:
            tid_val = r["template_id"] if not isinstance(r, tuple) else r[0]
            cnt = r["update_count"] if not isinstance(r, tuple) else r[1]
            count_map[tid_val] = int(cnt)

        bias_updates: list[tuple[str, float, int]] = []
        for _vec, transient_strength, bias_weight, template_id in templates:
            proportion = transient_strength / sum_strengths
            shift_contribution = proportion * avg_shift
            old_bias = existing_biases.get(template_id, 0.0)
            new_bias = min(BIAS_MAX_CAP, old_bias + BIAS_ACCUMULATION_RATE * shift_contribution)
            old_count = count_map.get(template_id, 0)
            bias_updates.append((template_id, new_bias, old_count + 1))

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

        # 各 composite の重心とメンバーを一括取得（N+1クエリ回避）
        all_centroids = await store.fetch_all_composite_centroids()
        all_members = await store.fetch_all_composites_with_vectors()
        member_sets = await store.fetch_all_composite_member_sets()

        centroids: dict[str, np.ndarray] = {}
        member_vecs_map: dict[str, list[tuple[str, np.ndarray]]] = {}
        existing_members: dict[str, set[str]] = {}

        for cid in composite_ids:
            centroid = all_centroids.get(cid)
            if centroid is None:
                continue
            centroids[cid] = centroid
            existing_members[cid] = member_sets.get(cid, set())
            member_vecs_map[cid] = all_members.get(cid, [])

        cids_with_centroid = list(centroids.keys())
        overlap_pairs = 0

        # 重複排除用set: (composite_id, member_id)
        inserts_set: set[tuple[str, str]] = set()

        if not cids_with_centroid:
            return {"overlap_pairs": 0, "dual_members_added": 0}

        # 事前正規化で重複ノルム計算を排除
        centroid_arr = np.array([centroids[c] for c in cids_with_centroid])
        c_norms = np.linalg.norm(centroid_arr, axis=1, keepdims=True) + 1e-10
        centroid_normed = centroid_arr / c_norms

        # centroid間の類似度行列を一括計算
        centroid_sim_matrix = centroid_normed @ centroid_normed.T  # (C, C)

        # メンバーベクトルも事前正規化
        member_normed_map: dict[str, tuple[list[str], np.ndarray]] = {}
        for cid in cids_with_centroid:
            mvecs = member_vecs_map.get(cid, [])
            if mvecs:
                mids = [m[0] for m in mvecs]
                vecs = np.array([m[1] for m in mvecs])
                norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10
                member_normed_map[cid] = (mids, vecs / norms)

        for i in range(len(cids_with_centroid)):
            for j in range(i + 1, len(cids_with_centroid)):
                sim = float(centroid_sim_matrix[i, j])
                if sim <= overlap_threshold:
                    continue

                overlap_pairs += 1
                cid_a = cids_with_centroid[i]
                cid_b = cids_with_centroid[j]

                # A のメンバーが B の重心に近いか（行列演算）
                if cid_a in member_normed_map:
                    mids_a, vecs_a = member_normed_map[cid_a]
                    sims_a = vecs_a @ centroid_normed[j]  # (M_a,)
                    existing_b = existing_members.get(cid_b, set())
                    for k in range(len(mids_a)):
                        mid = mids_a[k]
                        if mid not in existing_b and float(sims_a[k]) > overlap_threshold:
                            inserts_set.add((cid_b, mid))
                            existing_b.add(mid)

                # B のメンバーが A の重心に近いか（行列演算）
                if cid_b in member_normed_map:
                    mids_b, vecs_b = member_normed_map[cid_b]
                    sims_b = vecs_b @ centroid_normed[i]  # (M_b,)
                    existing_a = existing_members.get(cid_a, set())
                    for k in range(len(mids_b)):
                        mid = mids_b[k]
                        if mid not in existing_a and float(sims_b[k]) > overlap_threshold:
                            inserts_set.add((cid_a, mid))
                            existing_a.add(mid)

        dual_members_added = len(inserts_set)
        if inserts_set:
            db = store._ensure_connected()
            db.executemany(
                """INSERT OR IGNORE INTO composite_members
                   (composite_id, member_id, contribution_weight)
                   VALUES (?, ?, ?)""",
                [(cid, mid, 0.5) for cid, mid in inserts_set],
            )
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

        # 全 composite の重心を一括取得（N+1クエリ回避）
        all_centroids = await store.fetch_all_composite_centroids()
        composite_id_set = set(composite_ids)
        centroids: dict[str, np.ndarray] = {
            cid: vec for cid, vec in all_centroids.items() if cid in composite_id_set
        }

        if not centroids:
            return {"orphans_found": len(orphans), "orphans_rescued": 0}

        db = store._ensure_connected()

        # 行列演算で全orphan × 全centroidの類似度を一括計算
        cid_list = list(centroids.keys())
        centroid_arr = np.array([centroids[c] for c in cid_list])
        c_norms = np.linalg.norm(centroid_arr, axis=1, keepdims=True) + 1e-10
        centroid_normed = centroid_arr / c_norms

        orphan_vecs = np.array([vec for _, vec in orphans])
        o_norms = np.linalg.norm(orphan_vecs, axis=1, keepdims=True) + 1e-10
        orphan_normed = orphan_vecs / o_norms

        # (O, C) 類似度行列
        sim_matrix = orphan_normed @ centroid_normed.T
        best_indices = np.argmax(sim_matrix, axis=1)  # (O,)
        best_sims = np.max(sim_matrix, axis=1)  # (O,)

        inserts: list[tuple[str, str, float]] = []
        for i in range(len(orphans)):
            if float(best_sims[i]) >= rescue_threshold:
                inserts.append((cid_list[int(best_indices[i])], orphans[i][0].id, 0.3))

        orphans_rescued = len(inserts)
        if inserts:
            db.executemany(
                """INSERT OR IGNORE INTO composite_members
                   (composite_id, member_id, contribution_weight)
                   VALUES (?, ?, ?)""",
                inserts,
            )
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

    async def synthesize_flow_composites(
        self,
        store: "MemoryStore",
        similarity_threshold: float = 0.75,
        min_group_size: int = 2,
        max_group_size: int = 8,
    ) -> dict[str, int]:
        """image_embeddings のflow_vectorからクラスタリングし、場所のimage_composites を生成する。

        IDは 'flow-' プレフィックス。delta_centroidカラムにflow centroidを格納（NOT NULL制約対応）。
        """
        image_records = await store.fetch_image_embeddings_for_composites(
            min_person_ratio=0.0, min_freshness=0.1,
        )
        if len(image_records) < min_group_size:
            return {"flow_composites_created": 0, "flow_composites_skipped": 0}

        valid = [r for r in image_records if "flow_vector" in r]
        if len(valid) < min_group_size:
            return {"flow_composites_created": 0, "flow_composites_skipped": 0}

        flow_vecs = np.array([r["flow_vector"] for r in valid])

        # コサイン類似度行列
        norms = np.linalg.norm(flow_vecs, axis=1, keepdims=True) + 1e-10
        normalized = flow_vecs / norms
        sim_matrix = normalized @ normalized.T

        # Union-Find
        n = len(valid)
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

        groups: dict[int, list[int]] = {}
        for i in range(n):
            root = find(i)
            groups.setdefault(root, []).append(i)

        groups = {k: v for k, v in groups.items() if len(v) >= min_group_size}

        for root, indices in list(groups.items()):
            if len(indices) > max_group_size:
                idx_arr = np.array(indices)
                group_sims = sim_matrix[np.ix_(idx_arr, idx_arr)]
                np.fill_diagonal(group_sims, 0)
                scores = np.sum(group_sims, axis=1)
                top_k = np.argsort(-scores)[:max_group_size]
                groups[root] = idx_arr[top_k].tolist()

        existing = await store.get_existing_flow_composite_members()
        composites_created = 0
        composites_skipped = 0

        for indices in groups.values():
            member_ids = frozenset(valid[i]["id"] for i in indices)
            if member_ids in existing:
                composites_skipped += 1
                continue

            # flow_centroid（delta_centroidカラムに格納）
            member_flows = np.array([flow_vecs[i] for i in indices])
            flow_mean = member_flows.mean(axis=0)
            flow_norm = np.linalg.norm(flow_mean) + 1e-10
            flow_centroid = (flow_mean / flow_norm).astype(np.float32)

            # tag: 多数決
            tags = [valid[i].get("tag") for i in indices]
            tag_counts = Counter(t for t in tags if t)
            tag = tag_counts.most_common(1)[0][0] if tag_counts else None

            await store.save_flow_composite(
                member_ids=list(member_ids),
                flow_centroid=flow_centroid,
                tag=tag,
            )
            composites_created += 1

        return {
            "flow_composites_created": composites_created,
            "flow_composites_skipped": composites_skipped,
        }

    async def strengthen_visual_graph_edges(
        self,
        store: "MemoryStore",
        graph: "MemoryGraph",
    ) -> dict[str, int]:
        """tag付きimage_compositesのタグでgraph vnエッジを強化。

        - img-* composites → 「見る → {tag}」（人物）
        - flow-* composites → 「いる → {tag}」（場所）

        member_countが多いほどエッジが強くなる。
        """
        composites = await store.fetch_image_composites(min_freshness=0.1)
        tagged = [c for c in composites if c.get("tag")]
        if not tagged:
            return {"visual_edges_strengthened": 0}

        max_member_count = max(c.get("member_count", 1) for c in tagged)
        if max_member_count < 1:
            max_member_count = 1

        strengthened = 0
        for c in tagged:
            tag = c["tag"]
            member_count = c.get("member_count", 1)
            vn_weight = 0.2 * member_count / max_member_count
            # img-* → 「見る」, flow-* → 「いる」
            verb = "いる" if c["id"].startswith("flow-") else "見る"
            await graph.register_chain(
                verbs=[verb],
                nouns_per_step=[[tag]],
                delta_override={"vv": 0.0, "vn": vn_weight, "nn": 0.0},
            )
            strengthened += 1

        return {"visual_edges_strengthened": strengthened}

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

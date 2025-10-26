#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import glob
import argparse
from pathlib import Path
from collections import defaultdict

import spacy
import networkx as nx
from sentence_transformers import SentenceTransformer, util
from rapidfuzz import process, fuzz

# --------- 1) 캡션 -> 씬그래프 파싱 (spaCy 규칙 기반) ---------
_NLP = None


def get_nlp():
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_sm")
    return _NLP


def caption_to_graph(caption: str) -> nx.MultiDiGraph:
    """
    간단한 규칙:
      - 노드: 명사(lemma lower)
      - 속성: amod 형용사
      - 엣지: (주어-동사-목적어), 전치사구(verb_prep -> pobj)
    """
    nlp = get_nlp()
    doc = nlp(caption)
    G = nx.MultiDiGraph()

    # 1) 명사 노드 + amod 속성
    for tok in doc:
        if tok.pos_ in ("NOUN", "PROPN", "PRON"):
            node_name = tok.lemma_.lower()
            if not G.has_node(node_name):
                G.add_node(node_name, attrs=set())
            for child in tok.children:
                if child.dep_ == "amod" and child.pos_ == "ADJ":
                    G.nodes[node_name]["attrs"].add(child.lemma_.lower())

    # 2) 동사 중심 SVO & prep 처리
    for tok in doc:
        if tok.pos_ == "VERB":
            subj = None
            obj = None
            for child in tok.children:
                if child.dep_ in ("nsubj", "nsubjpass"):
                    subj = child.lemma_.lower()
            for child in tok.children:
                if child.dep_ in ("dobj", "attr", "oprd"):
                    obj = child.lemma_.lower()
                if child.dep_ == "prep":
                    for pobj in child.children:
                        if pobj.dep_ == "pobj":
                            pobj_name = pobj.lemma_.lower()
                            if subj and pobj_name:
                                if not G.has_node(pobj_name):
                                    G.add_node(pobj_name, attrs=set())
                                G.add_edge(subj, pobj_name, rel=f"{tok.lemma_.lower()}_{child.lemma_.lower()}")
            if subj and obj:
                if not G.has_node(obj):
                    G.add_node(obj, attrs=set())
                G.add_edge(subj, obj, rel=tok.lemma_.lower())

    for node in G.nodes:
        if isinstance(G.nodes[node].get("attrs"), set):
            G.nodes[node]["attrs"] = sorted(list(G.nodes[node]["attrs"]))
    return G


# --------- 2) 그래프 통합기 ---------
class GraphConsolidator:
    def __init__(self, sim_threshold: float = 0.78, model_name: str = "all-MiniLM-L6-v2"):
        self.global_graph = nx.MultiDiGraph()
        self.encoder = SentenceTransformer(model_name)
        self.sim_threshold = sim_threshold
        self._cache = {}

    def _emb(self, text: str):
        if text not in self._cache:
            self._cache[text] = self.encoder.encode(text, normalize_embeddings=True)
        return self._cache[text]

    def _match_node(self, name: str) -> str:
        """글로벌 그래프 내 유사 노드 찾기 (문자 유사도 → 임베딩 유사도)."""
        if self.global_graph.number_of_nodes() == 0:
            return ""
        candidates = list(self.global_graph.nodes)
        best, score, _ = process.extractOne(name, candidates, scorer=fuzz.WRatio)
        if score >= 92:
            return best
        query = self._emb(name)
        matrix = self.encoder.encode(candidates, normalize_embeddings=True)
        sims = util.cos_sim(query, matrix).cpu().numpy()[0]
        idx = sims.argmax()
        if sims[idx] >= self.sim_threshold:
            return candidates[idx]
        return ""

    def add_graph(self, graph: nx.MultiDiGraph, shot_id: int):
        name_map = {}
        for node, data in graph.nodes(data=True):
            matched = self._match_node(node)
            if matched == "":
                attrs = data.get("attrs") or []
                self.global_graph.add_node(node, attrs=set(attrs), shots={shot_id})
                name_map[node] = node
            else:
                cur_attrs = set(self.global_graph.nodes[matched].get("attrs", []))
                cur_attrs.update(data.get("attrs") or [])
                self.global_graph.nodes[matched]["attrs"] = cur_attrs
                cur_shots = self.global_graph.nodes[matched].get("shots", set())
                cur_shots.add(shot_id)
                self.global_graph.nodes[matched]["shots"] = cur_shots
                name_map[node] = matched

        for u, v, edge_data in graph.edges(data=True):
            mapped_u, mapped_v = name_map[u], name_map[v]
            rel = edge_data.get("rel", "")
            existing = self.global_graph.get_edge_data(mapped_u, mapped_v, default={})
            found = False
            for key in existing.keys():
                if self.global_graph[mapped_u][mapped_v][key].get("rel") == rel:
                    shots = self.global_graph[mapped_u][mapped_v][key].get("shots", set())
                    shots.add(shot_id)
                    self.global_graph[mapped_u][mapped_v][key]["shots"] = shots
                    found = True
                    break
            if not found:
                self.global_graph.add_edge(mapped_u, mapped_v, rel=rel, shots={shot_id})

    def finalize(self):
        for node in self.global_graph.nodes:
            if isinstance(self.global_graph.nodes[node].get("attrs"), set):
                self.global_graph.nodes[node]["attrs"] = sorted(list(self.global_graph.nodes[node]["attrs"]))
            if isinstance(self.global_graph.nodes[node].get("shots"), set):
                self.global_graph.nodes[node]["shots"] = sorted(list(self.global_graph.nodes[node]["shots"]))
        for u, v, key in self.global_graph.edges(keys=True):
            data = self.global_graph[u][v][key]
            if isinstance(data.get("shots"), set):
                data["shots"] = sorted(list(data["shots"]))

    def to_dict(self):
        self.finalize()
        nodes = [{"id": name, **attrs} for name, attrs in self.global_graph.nodes(data=True)]
        edges = [{"source": u, "target": v, **attrs} for u, v, _, attrs in self.global_graph.edges(keys=True, data=True)]
        return {"nodes": nodes, "edges": edges}


# --------- 3) 메인: per-sample JSON 읽어 그래프 생성/통합 ---------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", required=True, help="inference 결과 json 디렉토리 (…/outputs)")
    parser.add_argument("--outdir", required=True, help="그래프 저장 디렉토리")
    parser.add_argument("--sim_threshold", type=float, default=0.78)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.inputs, "*.json")))
    if not files:
        raise SystemExit(f"No JSON files found in {args.inputs}")

    qaig_files = [fp for fp in files if fp.endswith("_qaig.json")]
    if qaig_files:
        for path in qaig_files:
            with open(path, "r", encoding="utf-8") as f:
                record = json.load(f)
            video_name = record.get("video") or record.get("id") or Path(path).stem
            base = Path(video_name).stem or str(record.get("id"))
            per_shot = record.get("shot_outputs") or []
            logs = record.get("logs") or []
            graph = record.get("graph") or {}
            choices = record.get("choices") or []
            summary = {
                "id": record.get("id"),
                "video": record.get("video"),
                "query": record.get("query"),
                "qaig_config": record.get("qaig_config"),
                "prediction": record.get("prediction"),
                "prediction_text": record.get("prediction_text"),
                "choices": choices,
                "logs": logs,
            }

            with open(os.path.join(args.outdir, f"{base}_per_shot.json"), "w", encoding="utf-8") as f:
                json.dump(per_shot, f, ensure_ascii=False, indent=2)
            with open(os.path.join(args.outdir, f"{base}_unified_graph.json"), "w", encoding="utf-8") as f:
                json.dump(graph, f, ensure_ascii=False, indent=2)
            with open(os.path.join(args.outdir, f"{base}_summary.json"), "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            print(f"[qaig] {video_name} -> saved to {args.outdir}")
        return

    # fallback to legacy caption-based pipeline
    per_video = defaultdict(list)
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            sample = json.load(f)
        video_name = sample.get("video") or "unknown_video"
        per_video[video_name].append(sample)

    for video_name, samples in per_video.items():
        try:
            samples.sort(key=lambda item: int(str(item.get("shot_id", 0))))
        except Exception:
            pass

        consolidator = GraphConsolidator(sim_threshold=args.sim_threshold)
        per_shot_out = []
        for idx, sample in enumerate(samples):
            caption = (sample.get("prediction") or "").strip()
            if not caption:
                continue
            graph = caption_to_graph(caption)
            consolidator.add_graph(graph, shot_id=sample.get("shot_id", idx))
            per_shot_out.append(
                {
                    "shot_id": sample.get("shot_id", idx),
                    "id": sample.get("id"),
                    "video": sample.get("video"),
                    "caption": caption,
                    "graph": {
                        "nodes": [{"id": name, **attrs} for name, attrs in graph.nodes(data=True)],
                        "edges": [{"source": u, "target": v, **attrs} for u, v, attrs in graph.edges(data=True)],
                    },
                }
            )

        unified = consolidator.to_dict()
        base_name = Path(video_name).stem
        with open(os.path.join(args.outdir, f"{base_name}_per_shot.json"), "w", encoding="utf-8") as f:
            json.dump(per_shot_out, f, ensure_ascii=False, indent=2)
        with open(os.path.join(args.outdir, f"{base_name}_unified_graph.json"), "w", encoding="utf-8") as f:
            json.dump(unified, f, ensure_ascii=False, indent=2)
        print(f"[graph] {video_name} -> saved to {args.outdir}")


if __name__ == "__main__":
    main()

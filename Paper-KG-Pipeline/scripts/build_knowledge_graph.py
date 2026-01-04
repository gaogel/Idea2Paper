"""
知识图谱构建脚本
将论文抽取结果和Pattern聚类结果组装成完整的知识图谱

输入:
  - data/{conference}/*_paper_node.json: 论文抽取结果
  - output/patterns_structured.json: Pattern聚类结果 (由generate_patterns.py生成)
  
输出:
  - output/knowledge_graph.gpickle: NetworkX图谱 (二进制格式)
  - output/knowledge_graph.json: JSON格式图谱
  - output/knowledge_graph_stats.json: 图谱统计信息
"""

import os
import sys
import json
import pickle
import hashlib
import networkx as nx
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
from collections import defaultdict
from dataclasses import dataclass, asdict


# ===================== 配置 =====================

# 获取项目根目录 (知识图谱Pipeline)
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

# 输入路径
DATA_DIR = PROJECT_ROOT / "data"
PATTERNS_FILE = PROJECT_ROOT / "output" / "patterns_structured.json"

# 输出路径
OUTPUT_DIR = PROJECT_ROOT / "output"
GRAPH_GPICKLE = OUTPUT_DIR / "knowledge_graph.gpickle"
GRAPH_JSON = OUTPUT_DIR / "knowledge_graph.json"
STATS_FILE = OUTPUT_DIR / "knowledge_graph_stats.json"

# 会议列表
CONFERENCES = ["ACL_2017", "ARR_2022", "COLING_2020"]


# ===================== 数据类 =====================

@dataclass
class GraphStats:
    """图谱统计信息"""
    total_nodes: int = 0
    total_edges: int = 0
    papers: int = 0
    domains: int = 0
    ideas: int = 0
    skeletons: int = 0
    tricks: int = 0
    patterns: int = 0
    reviews: int = 0
    
    # 边统计
    paper_domain_edges: int = 0
    paper_idea_edges: int = 0
    paper_skeleton_edges: int = 0
    paper_trick_edges: int = 0
    paper_pattern_edges: int = 0
    paper_review_edges: int = 0
    pattern_trick_edges: int = 0
    pattern_skeleton_edges: int = 0


class KnowledgeGraphBuilder:
    """知识图谱构建器"""
    
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.stats = GraphStats()
        
        # 节点映射表 (用于去重和关联)
        self.domain_map: Dict[str, str] = {}  # domain_text -> node_id
        self.idea_map: Dict[str, str] = {}     # idea_hash -> node_id
        self.trick_map: Dict[str, str] = {}    # trick_name -> node_id
        self.pattern_map: Dict[int, str] = {}  # pattern_id -> node_id
        self.paper_map: Dict[str, str] = {}    # paper_id -> node_id
        
    def build(self):
        """构建完整的知识图谱"""
        print("=" * 60)
        print("🚀 开始构建知识图谱")
        print("=" * 60)
        
        # Step 1: 加载所有论文数据
        papers = self._load_papers()
        print(f"\n📊 加载了 {len(papers)} 篇论文")
        
        # Step 2: 加载Pattern数据 (如果存在)
        patterns = self._load_patterns()
        print(f"📊 加载了 {len(patterns)} 个Patterns")
        
        # Step 2.5: 加载Review数据
        reviews = self._load_reviews()
        print(f"📊 加载了 {len(reviews)} 条Reviews")
        
        # Step 3: 构建节点
        print("\n" + "=" * 60)
        print("🔨 构建节点...")
        print("=" * 60)
        
        self._build_domain_nodes(papers)
        self._build_idea_nodes(papers)
        self._build_trick_nodes(papers)
        self._build_pattern_nodes(patterns)
        self._build_paper_nodes(papers)
        self._build_skeleton_nodes(papers)
        self._build_review_nodes(reviews)
        
        # Step 4: 构建边
        print("\n" + "=" * 60)
        print("🔗 构建边关系...")
        print("=" * 60)
        
        self._build_paper_edges(papers)
        self._build_pattern_edges(patterns)
        
        # Step 5: 更新统计
        self._update_stats()
        
        # Step 6: 保存图谱
        self._save_graph()
        
        print("\n" + "=" * 60)
        print("✅ 知识图谱构建完成!")
        print("=" * 60)
        self._print_stats()
        
        return self.graph
    
    # ===================== 数据加载 =====================
    
    def _load_papers(self) -> List[Dict]:
        """加载所有会议的论文数据"""
        papers = []
        
        for conference in CONFERENCES:
            conf_dir = DATA_DIR / conference
            if not conf_dir.exists():
                print(f"⚠️  会议目录不存在: {conf_dir}")
                continue
            
            # 优先加载合并文件
            all_papers_file = conf_dir / "_all_paper_nodes.json"
            if all_papers_file.exists():
                with open(all_papers_file, 'r', encoding='utf-8') as f:
                    conf_papers = json.load(f)
                    papers.extend(conf_papers)
                    print(f"  📂 {conference}: 加载 {len(conf_papers)} 篇 (from _all_paper_nodes.json)")
            else:
                # 加载单个文件
                count = 0
                for file in conf_dir.glob("*_paper_node.json"):
                    if file.name.startswith("_"):
                        continue
                    with open(file, 'r', encoding='utf-8') as f:
                        paper = json.load(f)
                        papers.append(paper)
                        count += 1
                print(f"  📂 {conference}: 加载 {count} 篇")
        
        return papers
    
    def _load_patterns(self) -> List[Dict]:
        """加载Pattern聚类结果"""
        if not PATTERNS_FILE.exists():
            print(f"⚠️  Pattern文件不存在: {PATTERNS_FILE}")
            print("   (可以先运行 generate_patterns.py 生成)")
            return []
        
        with open(PATTERNS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _load_reviews(self) -> List[Dict]:
        """加载所有会议的Review数据"""
        reviews = []
        
        for conference in CONFERENCES:
            conf_dir = DATA_DIR / conference
            if not conf_dir.exists():
                continue
            
            # 优先加载合并文件
            all_reviews_file = conf_dir / "_all_review_nodes.json"
            if all_reviews_file.exists():
                with open(all_reviews_file, 'r', encoding='utf-8') as f:
                    conf_reviews = json.load(f)
                    reviews.extend(conf_reviews)
                    print(f"  📝 {conference}: 加载 {len(conf_reviews)} 条review")
            else:
                # 加载单个文件
                count = 0
                for file in conf_dir.glob("*_reviews.json"):
                    if file.name.startswith("_"):
                        continue
                    with open(file, 'r', encoding='utf-8') as f:
                        file_reviews = json.load(f)
                        reviews.extend(file_reviews)
                        count += len(file_reviews)
                if count > 0:
                    print(f"  📝 {conference}: 加载 {count} 条review")
        
        return reviews
    
    # ===================== 构建节点 =====================
    
    def _build_domain_nodes(self, papers: List[Dict]):
        """构建Domain节点"""
        print("\n🌐 构建Domain节点...")
        
        for paper in papers:
            domain_info = paper.get('domain', {})
            
            # 从domains列表提取
            domains_list = domain_info.get('domains', [])
            for domain_text in domains_list:
                if domain_text and domain_text not in self.domain_map:
                    node_id = f"domain_{len(self.domain_map)}"
                    self.domain_map[domain_text] = node_id
                    
                    self.graph.add_node(
                        node_id,
                        node_type='Domain',
                        name=domain_text,
                        research_object=domain_info.get('research_object', ''),
                        core_technique=domain_info.get('core_technique', ''),
                        application=domain_info.get('application', '')
                    )
                    self.stats.domains += 1
        
        print(f"  ✓ 创建了 {self.stats.domains} 个Domain节点")
    
    def _build_idea_nodes(self, papers: List[Dict]):
        """构建Idea节点"""
        print("\n💡 构建Idea节点...")
        
        for paper in papers:
            ideal_info = paper.get('ideal', {})
            core_idea = ideal_info.get('core_idea', '')
            
            if core_idea:
                # 用核心想法的hash去重
                idea_hash = hashlib.md5(core_idea.encode()).hexdigest()[:16]
                
                if idea_hash not in self.idea_map:
                    node_id = f"idea_{len(self.idea_map)}"
                    self.idea_map[idea_hash] = node_id
                    
                    self.graph.add_node(
                        node_id,
                        node_type='Idea',
                        description=core_idea,
                        tech_stack=ideal_info.get('tech_stack', []),
                        input_type=ideal_info.get('input_type', ''),
                        output_type=ideal_info.get('output_type', '')
                    )
                    self.stats.ideas += 1
        
        print(f"  ✓ 创建了 {self.stats.ideas} 个Idea节点")
    
    def _build_trick_nodes(self, papers: List[Dict]):
        """构建Trick节点"""
        print("\n🎯 构建Trick节点...")
        
        for paper in papers:
            tricks = paper.get('tricks', [])
            
            for trick in tricks:
                trick_name = trick.get('name', '')
                if trick_name and trick_name not in self.trick_map:
                    node_id = f"trick_{len(self.trick_map)}"
                    self.trick_map[trick_name] = node_id
                    
                    self.graph.add_node(
                        node_id,
                        node_type='Trick',
                        name=trick_name,
                        trick_type=trick.get('type', 'unknown'),
                        purpose=trick.get('purpose', ''),
                        location=trick.get('location', ''),
                        description=trick.get('description', '')
                    )
                    self.stats.tricks += 1
        
        print(f"  ✓ 创建了 {self.stats.tricks} 个Trick节点")
    
    def _build_pattern_nodes(self, patterns: List[Dict]):
        """构建Pattern节点"""
        print("\n📋 构建Pattern节点...")
        
        for pattern in patterns:
            pattern_id = pattern.get('pattern_id')
            node_id = f"pattern_{pattern_id}"
            self.pattern_map[pattern_id] = node_id
            
            self.graph.add_node(
                node_id,
                node_type='Pattern',
                pattern_id=pattern_id,
                name=pattern.get('pattern_name', ''),
                summary=pattern.get('pattern_summary', ''),
                writing_guide=pattern.get('writing_guide', ''),
                paper_count=len(pattern.get('skeleton_examples', []))
            )
            self.stats.patterns += 1
        
        print(f"  ✓ 创建了 {self.stats.patterns} 个Pattern节点")
    
    def _build_paper_nodes(self, papers: List[Dict]):
        """构建Paper节点"""
        print("\n📄 构建Paper节点...")
        
        for paper in papers:
            paper_id = paper.get('paper_id', '')
            node_id = f"paper_{paper_id}"
            self.paper_map[paper_id] = node_id
            
            self.graph.add_node(
                node_id,
                node_type='Paper',
                paper_id=paper_id,
                title=paper.get('title', ''),
                conference=paper.get('conference', '')
            )
            self.stats.papers += 1
        
        print(f"  ✓ 创建了 {self.stats.papers} 个Paper节点")
    
    def _build_skeleton_nodes(self, papers: List[Dict]):
        """构建Skeleton节点"""
        print("\n🦴 构建Skeleton节点...")
        
        for paper in papers:
            paper_id = paper.get('paper_id', '')
            skeleton = paper.get('skeleton', {})
            
            if skeleton:
                node_id = f"skeleton_{paper_id}"
                
                self.graph.add_node(
                    node_id,
                    node_type='Skeleton',
                    paper_id=paper_id,
                    problem_framing=skeleton.get('problem_framing', ''),
                    gap_pattern=skeleton.get('gap_pattern', ''),
                    method_story=skeleton.get('method_story', ''),
                    experiments_story=skeleton.get('experiments_story', '')
                )
                self.stats.skeletons += 1
                
                # 建立 Paper -> Skeleton 边
                paper_node_id = self.paper_map.get(paper_id)
                if paper_node_id:
                    self.graph.add_edge(paper_node_id, node_id, relation='has_skeleton')
                    self.stats.paper_skeleton_edges += 1
        
        print(f"  ✓ 创建了 {self.stats.skeletons} 个Skeleton节点")
    
    def _build_review_nodes(self, reviews: List[Dict]):
        """构建Review节点"""
        print("\n📝 构建Review节点...")
        
        for review in reviews:
            review_id = review.get('review_id', '')
            paper_id = review.get('paper_id', '')
            node_id = f"review_{review_id}"
            
            self.graph.add_node(
                node_id,
                node_type='Review',
                review_id=review_id,
                paper_id=paper_id,
                reviewer=review.get('reviewer'),
                paper_summary=review.get('paper_summary', '')[:500],  # 截取前500字符
                strengths=review.get('strengths', '')[:500],
                weaknesses=review.get('weaknesses', '')[:500],
                comments=review.get('comments', '')[:500],
                overall_score=review.get('overall_score', ''),
                confidence=review.get('confidence', '')
            )
            self.stats.reviews += 1
            
            # 建立 Paper -> Review 边
            paper_node_id = self.paper_map.get(paper_id)
            if paper_node_id:
                self.graph.add_edge(paper_node_id, node_id, relation='has_review')
                self.stats.paper_review_edges += 1
        
        print(f"  ✓ 创建了 {self.stats.reviews} 个Review节点")
        print(f"  ✓ Paper->Review: {self.stats.paper_review_edges} 条")
    
    # ===================== 构建边 =====================
    
    def _build_paper_edges(self, papers: List[Dict]):
        """构建论文相关的边"""
        print("\n🔗 构建论文关联边...")
        
        for paper in papers:
            paper_id = paper.get('paper_id', '')
            paper_node_id = self.paper_map.get(paper_id)
            if not paper_node_id:
                continue
            
            # Paper -> Domain
            domain_info = paper.get('domain', {})
            for domain_text in domain_info.get('domains', []):
                domain_node_id = self.domain_map.get(domain_text)
                if domain_node_id:
                    self.graph.add_edge(paper_node_id, domain_node_id, relation='in_domain')
                    self.stats.paper_domain_edges += 1
            
            # Paper -> Idea
            ideal_info = paper.get('ideal', {})
            core_idea = ideal_info.get('core_idea', '')
            if core_idea:
                idea_hash = hashlib.md5(core_idea.encode()).hexdigest()[:16]
                idea_node_id = self.idea_map.get(idea_hash)
                if idea_node_id:
                    self.graph.add_edge(paper_node_id, idea_node_id, relation='implements')
                    self.stats.paper_idea_edges += 1
            
            # Paper -> Trick
            for trick in paper.get('tricks', []):
                trick_name = trick.get('name', '')
                trick_node_id = self.trick_map.get(trick_name)
                if trick_node_id:
                    self.graph.add_edge(
                        paper_node_id, 
                        trick_node_id, 
                        relation='uses_trick',
                        location=trick.get('location', ''),
                        purpose=trick.get('purpose', '')
                    )
                    self.stats.paper_trick_edges += 1
        
        print(f"  ✓ Paper->Domain: {self.stats.paper_domain_edges} 条")
        print(f"  ✓ Paper->Idea: {self.stats.paper_idea_edges} 条")
        print(f"  ✓ Paper->Skeleton: {self.stats.paper_skeleton_edges} 条")
        print(f"  ✓ Paper->Trick: {self.stats.paper_trick_edges} 条")
    
    def _build_pattern_edges(self, patterns: List[Dict]):
        """构建Pattern相关的边"""
        print("\n🔗 构建Pattern关联边...")
        
        for pattern in patterns:
            pattern_id = pattern.get('pattern_id')
            pattern_node_id = self.pattern_map.get(pattern_id)
            if not pattern_node_id:
                continue
            
            # Pattern -> Paper (through skeleton_examples)
            for example in pattern.get('skeleton_examples', []):
                example_paper_id = example.get('paper_id', '')
                paper_node_id = self.paper_map.get(example_paper_id)
                if paper_node_id:
                    self.graph.add_edge(
                        pattern_node_id, 
                        paper_node_id, 
                        relation='exemplified_by'
                    )
                    self.stats.paper_pattern_edges += 1
                
                # Pattern -> Skeleton
                skeleton_node_id = f"skeleton_{example_paper_id}"
                if self.graph.has_node(skeleton_node_id):
                    self.graph.add_edge(
                        pattern_node_id,
                        skeleton_node_id,
                        relation='has_skeleton_example'
                    )
                    self.stats.pattern_skeleton_edges += 1
            
            # Pattern -> Trick (through common_tricks)
            for trick_info in pattern.get('common_tricks', []):
                trick_name = trick_info.get('trick_name', '')
                trick_node_id = self.trick_map.get(trick_name)
                if trick_node_id:
                    self.graph.add_edge(
                        pattern_node_id,
                        trick_node_id,
                        relation='commonly_uses',
                        frequency=trick_info.get('frequency', 0),
                        percentage=trick_info.get('percentage', '')
                    )
                    self.stats.pattern_trick_edges += 1
        
        print(f"  ✓ Pattern->Paper: {self.stats.paper_pattern_edges} 条")
        print(f"  ✓ Pattern->Skeleton: {self.stats.pattern_skeleton_edges} 条")
        print(f"  ✓ Pattern->Trick: {self.stats.pattern_trick_edges} 条")
    
    # ===================== 保存和统计 =====================
    
    def _update_stats(self):
        """更新统计信息"""
        self.stats.total_nodes = self.graph.number_of_nodes()
        self.stats.total_edges = self.graph.number_of_edges()
    
    def _save_graph(self):
        """保存图谱"""
        print("\n💾 保存知识图谱...")
        
        # 确保输出目录存在
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        # 保存 gpickle 格式
        with open(GRAPH_GPICKLE, 'wb') as f:
            pickle.dump(self.graph, f)
        print(f"  ✓ 保存到: {GRAPH_GPICKLE}")
        
        # 保存 JSON 格式
        graph_data = {
            'nodes': [],
            'edges': [],
            'stats': asdict(self.stats)
        }
        
        for node, data in self.graph.nodes(data=True):
            node_info = {'id': node}
            node_info.update(data)
            graph_data['nodes'].append(node_info)
        
        for u, v, data in self.graph.edges(data=True):
            edge_info = {'source': u, 'target': v}
            edge_info.update(data)
            graph_data['edges'].append(edge_info)
        
        with open(GRAPH_JSON, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, ensure_ascii=False, indent=2)
        print(f"  ✓ 保存到: {GRAPH_JSON}")
        
        # 保存统计信息
        with open(STATS_FILE, 'w', encoding='utf-8') as f:
            json.dump(asdict(self.stats), f, ensure_ascii=False, indent=2)
        print(f"  ✓ 保存到: {STATS_FILE}")
    
    def _print_stats(self):
        """打印统计信息"""
        print("\n📊 知识图谱统计:")
        print("-" * 40)
        print(f"  总节点数: {self.stats.total_nodes}")
        print(f"  总边数:   {self.stats.total_edges}")
        print("-" * 40)
        print("  节点类型:")
        print(f"    Paper:    {self.stats.papers}")
        print(f"    Domain:   {self.stats.domains}")
        print(f"    Idea:     {self.stats.ideas}")
        print(f"    Skeleton: {self.stats.skeletons}")
        print(f"    Trick:    {self.stats.tricks}")
        print(f"    Pattern:  {self.stats.patterns}")
        print("-" * 40)


def main():
    """主函数"""
    builder = KnowledgeGraphBuilder()
    graph = builder.build()
    return graph


if __name__ == '__main__':
    main()

# 论文知识图谱构建 Pipeline

从NLP研究论文构建知识图谱：信息抽取、Pattern聚类、图谱构建。

[English](README.md) | 中文

---

## 快速开始

### 安装依赖

```bash
cd Paper-KG-Pipeline
pip install -r requirements.txt
```

### 一键运行

```bash
cd Paper-KG-Pipeline/scripts
python run_pipeline.py
```

### 分步运行

```bash
cd Paper-KG-Pipeline/scripts

# Step 1: 数据抽取 (需要 OpenAI API，已有结果可跳过)
# python extract_paper_review.py

# Step 2: Pattern聚类 (需要 Embedding API，已有结果可跳过)
# python generate_patterns.py

# Step 3: 构建知识图谱
python build_knowledge_graph.py
```

### 输出文件

运行完成后，在 `output/` 目录下生成：
- `knowledge_graph.gpickle` - NetworkX图谱 (推荐使用)
- `knowledge_graph.json` - JSON格式图谱
- `knowledge_graph_stats.json` - 统计信息

---

## 目录结构

```
Paper-KG-Pipeline/
├── data/                              # 论文抽取结果数据
│   ├── ACL_2017/                      # 135篇论文
│   │   ├── ACL_2017_*_paper_node.json # 单篇论文抽取结果
│   │   └── _all_paper_nodes.json      # 合并文件
│   ├── ARR_2022/                      # 323篇论文
│   │   ├── ARR_2022_*_paper_node.json # 单篇论文抽取结果
│   │   └── _all_paper_nodes.json      # 合并文件
│   └── COLING_2020/                   # 87篇论文
│       ├── COLING_2020_*_paper_node.json
│       └── _all_paper_nodes.json
│
├── scripts/                           # 核心脚本
│   ├── extract_paper_review.py        # Step1: 信息抽取
│   ├── generate_patterns.py           # Step2: 聚类+Pattern生成
│   └── build_knowledge_graph.py       # Step3: 知识图谱构建
│
├── output/                            # 输出结果
│   ├── patterns_structured.json       # Pattern聚类结果
│   ├── knowledge_graph.gpickle        # 知识图谱 (NetworkX格式)
│   ├── knowledge_graph.json           # 知识图谱 (JSON格式)
│   └── knowledge_graph_stats.json     # 图谱统计信息
│
└── README.md
```

## 知识图谱节点类型

| 节点类型 | 数量 | 说明 | 关键属性 |
|---------|------|------|--------|
| **Paper** | 545 | 论文 | paper_id, title, conference |
| **Domain** | 257 | 研究领域 | name, research_object, core_technique |
| **Idea** | 545 | 核心创新点 | description, tech_stack, input_type, output_type |
| **Skeleton** | 545 | 论文骨架 | problem_framing, gap_pattern, method_story, experiments_story |
| **Trick** | 4550 | 写作技巧 | name, type, purpose, location, description |
| **Pattern** | 29 | 写作套路 | name, summary, writing_guide |
| **Review** | 989 | 审稿意见 | reviewer, strengths, weaknesses, overall_score |

## 边关系类型

| 关系 | 起点 → 终点 | 说明 |
|-----|-------------|------|
| `in_domain` | Paper → Domain | 论文所属领域 |
| `implements` | Paper → Idea | 论文实现的创新点 |
| `has_skeleton` | Paper → Skeleton | 论文的结构骨架 |
| `uses_trick` | Paper → Trick | 论文使用的技巧 |
| `has_review` | Paper → Review | 论文的审稿意见 |
| `exemplified_by` | Pattern → Paper | Pattern的示例论文 |
| `commonly_uses` | Pattern → Trick | Pattern常用的技巧 |
| `has_skeleton_example` | Pattern → Skeleton | Pattern的骨架示例 |

## 使用方法

### 前置依赖

```bash
pip install -r requirements.txt
```

### 环境变量配置 (可选)

如需重新运行数据抽取或Pattern生成，需要配置 API Token：

```bash
# Linux/Mac
export LLM_AUTH_TOKEN='Bearer your_token_here'

# Windows PowerShell
$env:LLM_AUTH_TOKEN='Bearer your_token_here'
```

> 注意：已提供预处理数据，可直接运行 `build_knowledge_graph.py` 构建图谱，无需API。

### Step 1: 信息抽取 (已完成)

从论文中抽取四层结构信息：
- **domain**: 研究对象、核心技术、应用场景
- **ideal**: 核心创新点、技术栈、输入输出
- **skeleton**: 问题定位、研究缺口、方法叙述、实验设计
- **tricks**: 写作技巧列表

```bash
cd scripts
python extract_paper_review.py
```

输入: 原始论文数据 (ACL_2017, ARR_2022, COLING_2020)
输出: `data/{conference}/*_paper_node.json`

### Step 2: Pattern聚类 (已完成)

使用层次聚类将相似的论文骨架聚合成Pattern：
- Embedding: Qwen3-Embedding-8B (4096维)
- 融合权重: skeleton 40% + tricks 60%
- 聚类算法: AgglomerativeClustering (余弦距离, threshold=0.35)

```bash
cd scripts
python generate_patterns.py
```

输入: `data/{conference}/*_paper_node.json`
输出: `output/patterns_structured.json`

### Step 3: 构建知识图谱

整合抽取结果和Pattern聚类结果，构建完整的知识图谱：

```bash
cd scripts
python build_knowledge_graph.py
```

输入:
- `data/{conference}/*_paper_node.json`
- `output/patterns_structured.json`

输出:
- `output/knowledge_graph.gpickle` (NetworkX二进制格式)
- `output/knowledge_graph.json` (JSON格式)
- `output/knowledge_graph_stats.json` (统计信息)

## 数据来源

| 会议 | 年份 | 论文数 | Review数 | 说明 |
|-----|------|--------|---------|------|
| ACL | 2017 | 135 | 272 | 自然语言处理顶会 |
| ARR | 2022 | 323 | 606 | ACL Rolling Review |
| COLING | 2020 | 87 | 111 | 计算语言学会议 |

## 示例代码

### 加载知识图谱

```python
import json
import pickle
import networkx as nx

# 方式1: 加载gpickle格式 (推荐)
with open('output/knowledge_graph.gpickle', 'rb') as f:
    G = pickle.load(f)

# 方式2: 加载JSON格式
with open('output/knowledge_graph.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    G = nx.MultiDiGraph()
    for node in data['nodes']:
        G.add_node(node['id'], **node)
    for edge in data['edges']:
        G.add_edge(edge['source'], edge['target'], **edge)

# 统计信息
print(f"节点数: {G.number_of_nodes()}")
print(f"边数: {G.number_of_edges()}")
```

### 查询示例

```python
# 查询所有Pattern节点
patterns = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'Pattern']
for p in patterns:
    print(f"{p}: {G.nodes[p].get('name')}")

# 查询某个Pattern的示例论文
pattern_id = 'pattern_1'
papers = [v for u, v, d in G.edges(data=True) 
          if u == pattern_id and d.get('relation') == 'exemplified_by']

# 查询某篇论文使用的技巧
paper_id = 'paper_ARR_2022_0'
tricks = [v for u, v, d in G.edges(data=True) 
          if u == paper_id and d.get('relation') == 'uses_trick']
```

## License

MIT

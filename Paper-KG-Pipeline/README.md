# Paper Knowledge Graph Pipeline

Build a knowledge graph from NLP research papers: information extraction, pattern clustering, and graph construction.

English | [中文](README_CN.md)

---

## Quick Start

### Install Dependencies

```bash
cd Paper-KG-Pipeline
pip install -r requirements.txt
```

### One-Click Run

```bash
cd Paper-KG-Pipeline/scripts
python run_pipeline.py
```

### Step-by-Step

```bash
cd Paper-KG-Pipeline/scripts

# Step 1: Data extraction (requires OpenAI API, skip if results exist)
# python extract_paper_review.py

# Step 2: Pattern clustering (requires Embedding API, skip if results exist)
# python generate_patterns.py

# Step 3: Build knowledge graph
python build_knowledge_graph.py
```

### Output Files

After running, the following files will be generated in `output/`:
- `knowledge_graph.gpickle` - NetworkX graph (recommended)
- `knowledge_graph.json` - JSON format graph
- `knowledge_graph_stats.json` - Statistics

---

## Directory Structure

```
Paper-KG-Pipeline/
├── data/                              # Paper extraction results
│   ├── ACL_2017/                      # 135 papers
│   │   ├── ACL_2017_*_paper_node.json # Single paper extraction
│   │   └── _all_paper_nodes.json      # Merged file
│   ├── ARR_2022/                      # 323 papers
│   │   ├── ARR_2022_*_paper_node.json # Single paper extraction
│   │   └── _all_paper_nodes.json      # Merged file
│   └── COLING_2020/                   # 87 papers
│       ├── COLING_2020_*_paper_node.json
│       └── _all_paper_nodes.json
│
├── scripts/                           # Core scripts
│   ├── extract_paper_review.py        # Step1: Information extraction
│   ├── generate_patterns.py           # Step2: Clustering + Pattern generation
│   └── build_knowledge_graph.py       # Step3: Knowledge graph construction
│
├── output/                            # Output results
│   ├── patterns_structured.json       # Pattern clustering results
│   ├── knowledge_graph.gpickle        # Knowledge graph (NetworkX)
│   ├── knowledge_graph.json           # Knowledge graph (JSON)
│   └── knowledge_graph_stats.json     # Graph statistics
│
└── README.md
```

## Node Types

| Node Type | Count | Description | Key Attributes |
|---------|------|------|--------|
| **Paper** | 545 | Paper | paper_id, title, conference |
| **Domain** | 257 | Research Domain | name, research_object, core_technique |
| **Idea** | 545 | Core Innovation | description, tech_stack, input_type, output_type |
| **Skeleton** | 545 | Paper Structure | problem_framing, gap_pattern, method_story, experiments_story |
| **Trick** | 4550 | Writing Techniques | name, type, purpose, location, description |
| **Pattern** | 29 | Writing Patterns | name, summary, writing_guide |
| **Review** | 989 | Peer Reviews | reviewer, strengths, weaknesses, overall_score |

## Edge Types

| Relation | Source → Target | Description |
|-----|-------------|------|
| `in_domain` | Paper → Domain | Paper's research domain |
| `implements` | Paper → Idea | Paper's core innovation |
| `has_skeleton` | Paper → Skeleton | Paper's structure |
| `uses_trick` | Paper → Trick | Writing techniques used |
| `has_review` | Paper → Review | Peer review comments |
| `exemplified_by` | Pattern → Paper | Example papers for pattern |
| `commonly_uses` | Pattern → Trick | Common tricks in pattern |
| `has_skeleton_example` | Pattern → Skeleton | Skeleton examples |

## Usage

### Prerequisites

```bash
pip install -r requirements.txt
```

### Environment Variables (Optional)

If you need to re-run data extraction or pattern generation, configure the API Token:

```bash
# Linux/Mac
export LLM_AUTH_TOKEN='Bearer your_token_here'

# Windows PowerShell
$env:LLM_AUTH_TOKEN='Bearer your_token_here'
```

> Note: Pre-processed data is provided. You can directly run `build_knowledge_graph.py` without API access.

### Step 1: Information Extraction (Completed)

Extract four-layer structured information from papers:
- **domain**: Research object, core techniques, applications
- **ideal**: Core innovation, tech stack, input/output
- **skeleton**: Problem framing, research gap, method narrative, experiment design
- **tricks**: Writing technique list

```bash
cd scripts
python extract_paper_review.py
```

Input: Raw paper data (ACL_2017, ARR_2022, COLING_2020)
Output: `data/{conference}/*_paper_node.json`

### Step 2: Pattern Clustering (Completed)

Cluster similar paper structures using hierarchical clustering:
- Embedding: Qwen3-Embedding-8B (4096-dim)
- Fusion weights: skeleton 40% + tricks 60%
- Clustering: AgglomerativeClustering (cosine distance, threshold=0.35)

```bash
cd scripts
python generate_patterns.py
```

Input: `data/{conference}/*_paper_node.json`
Output: `output/patterns_structured.json`

### Step 3: Build Knowledge Graph

Integrate extraction results and pattern clustering to build the complete knowledge graph:

```bash
cd scripts
python build_knowledge_graph.py
```

Input:
- `data/{conference}/*_paper_node.json`
- `output/patterns_structured.json`

Output:
- `output/knowledge_graph.gpickle` (NetworkX binary format)
- `output/knowledge_graph.json` (JSON format)
- `output/knowledge_graph_stats.json` (Statistics)

## Data Sources

| Conference | Year | Papers | Reviews | Description |
|-----|------|--------|---------|------|
| ACL | 2017 | 135 | 272 | Top NLP conference |
| ARR | 2022 | 323 | 606 | ACL Rolling Review |
| COLING | 2020 | 87 | 111 | Computational Linguistics |

## Example Code

### Load Knowledge Graph

```python
import json
import pickle
import networkx as nx

# Method 1: Load gpickle format (recommended)
with open('output/knowledge_graph.gpickle', 'rb') as f:
    G = pickle.load(f)

# Method 2: Load JSON format
with open('output/knowledge_graph.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    G = nx.MultiDiGraph()
    for node in data['nodes']:
        G.add_node(node['id'], **node)
    for edge in data['edges']:
        G.add_edge(edge['source'], edge['target'], **edge)

# Statistics
print(f"Nodes: {G.number_of_nodes()}")
print(f"Edges: {G.number_of_edges()}")
```

### Query Examples

```python
# Query all Pattern nodes
patterns = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'Pattern']
for p in patterns:
    print(f"{p}: {G.nodes[p].get('name')}")

# Query example papers for a Pattern
pattern_id = 'pattern_1'
papers = [v for u, v, d in G.edges(data=True) 
          if u == pattern_id and d.get('relation') == 'exemplified_by']

# Query tricks used by a paper
paper_id = 'paper_ARR_2022_0'
tricks = [v for u, v, d in G.edges(data=True) 
          if u == paper_id and d.get('relation') == 'uses_trick']
```



"""
将知识图谱导出为JSON格式
"""

import json
import networkx as nx

print("加载知识图谱...")
G = nx.read_gpickle('knowledge_graph.gpickle')

print(f"节点数: {G.number_of_nodes()}")
print(f"边数: {G.number_of_edges()}")

# ============= 导出为JSON格式 =============
print("\n转换为JSON格式...")

# 方式1: node-link格式（标准格式，易于可视化）
graph_data = nx.node_link_data(G)

# 保存为JSON
output_path = 'knowledge_graph.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(graph_data, f, indent=2, ensure_ascii=False)

print(f"✓ 已保存至: {output_path}")

# ============= 导出为简化版本（只包含核心信息） =============
print("\n生成简化版本...")

simplified_graph = {
    'nodes': [],
    'edges': []
}

# 添加节点
for node_id, data in G.nodes(data=True):
    node_info = {
        'id': node_id,
        'type': data.get('type', 'Unknown')
    }
    
    # 根据节点类型添加关键属性
    if data.get('type') == 'Paper':
        node_info['title'] = data.get('title', '')
        node_info['conference'] = data.get('conference', '')
    elif data.get('type') == 'Pattern':
        node_info['name'] = data.get('pattern_name', '')
        node_info['size'] = data.get('cluster_size', 0)
    elif data.get('type') == 'Trick':
        node_info['name'] = data.get('name', '')
        node_info['trick_type'] = data.get('trick_type', '')
    elif data.get('type') == 'Domain':
        node_info['name'] = data.get('name', '')
        node_info['paper_count'] = data.get('paper_count', 0)
    elif data.get('type') == 'Review':
        node_info['score'] = data.get('overall_score', '')
    
    simplified_graph['nodes'].append(node_info)

# 添加边
for source, target, data in G.edges(data=True):
    edge_info = {
        'source': source,
        'target': target,
        'relation': data.get('relation', 'Unknown')
    }
    
    # 添加额外属性（如频率）
    if 'frequency' in data:
        edge_info['frequency'] = data['frequency']
    if 'percentage' in data:
        edge_info['percentage'] = data['percentage']
    
    simplified_graph['edges'].append(edge_info)

# 保存简化版本
simplified_path = 'knowledge_graph_simplified.json'
with open(simplified_path, 'w', encoding='utf-8') as f:
    json.dump(simplified_graph, f, indent=2, ensure_ascii=False)

print(f"✓ 简化版已保存至: {simplified_path}")

# ============= 统计信息 =============
print("\n【文件大小】")
import os
print(f"  完整版: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
print(f"  简化版: {os.path.getsize(simplified_path) / 1024 / 1024:.2f} MB")

print("\n✅ JSON导出完成！")

"""
论文与审稿意见综合抽取脚本
整合 paper 和 review 数据，输出适合知识图谱的节点格式
"""

import json
import re
import os
import requests
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict, field
from datetime import datetime


# ============================================================
# 配置 - 请配置环境变量或修改此处
# ============================================================
API_URL = os.environ.get("LLM_API_URL", "https://api.openai.com/v1/chat/completions")
AUTH_TOKEN = os.environ.get("LLM_AUTH_TOKEN", "")
MODEL = os.environ.get("LLM_MODEL", "gpt-4")

if not AUTH_TOKEN:
    print("⚠️  警告: 未设置 LLM_AUTH_TOKEN 环境变量，抽取功能将不可用")
    print("   请设置环境变量: export LLM_AUTH_TOKEN='Bearer your_token_here'")


# ============================================================
# 数据结构 - 论文节点
# ============================================================
@dataclass
class PaperSections:
    """论文分段结果"""
    title: str
    abstract: str
    keywords: List[str]
    introduction: str
    related_work: str
    method: str
    experiments: str
    conclusion: str


@dataclass
class ReviewNode:
    """单条审稿知识图谱节点"""
    review_id: str
    paper_id: str
    reviewer: Optional[str]
    paper_summary: str
    strengths: str
    weaknesses: str
    comments: str
    overall_score: str
    confidence: str


@dataclass
class PaperNode:
    """论文知识图谱节点"""
    # 基础信息
    paper_id: str
    title: str
    conference: str
    
    # 抽取的四类信息
    domain: Dict
    ideal: Dict
    skeleton: Dict
    tricks: List[Dict]


# ============================================================
# JSON 解析
# ============================================================
def parse_json_paper(json_path: str) -> PaperSections:
    """从 JSON 文件解析论文章节"""
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    metadata = data.get('metadata', {})
    title = metadata.get('title', '')
    sections_data = metadata.get('sections', [])
    
    abstract = ""
    keywords = []
    introduction = ""
    related_work = ""
    method = ""
    experiments = ""
    conclusion = ""
    
    for section in sections_data:
        heading = section.get('heading', '').lower()
        text = section.get('text', '')
        
        if not text:
            continue
        
        if 'abstract' in heading:
            abstract = text
        elif 'introduction' in heading or heading.startswith('1 '):
            introduction += text + " "
        elif any(kw in heading for kw in ['related', 'background', 'preliminar']):
            related_work += text + " "
        elif any(kw in heading for kw in ['method', 'approach', 'model', 'proposed', 'framework']):
            method += text + " "
        elif any(kw in heading for kw in ['experiment', 'evaluation', 'result', 'empirical']):
            experiments += text + " "
        elif any(kw in heading for kw in ['conclusion', 'discussion', 'future']):
            conclusion += text + " "
    
    if not abstract and sections_data:
        first_text = sections_data[0].get('text', '')
        if len(first_text) < 2000:
            abstract = first_text
    
    # 清理文本但不截断
    def clean_text(text):
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
        return text
    
    return PaperSections(
        title=title,
        abstract=clean_text(abstract),
        keywords=keywords,
        introduction=clean_text(introduction),
        related_work=clean_text(related_work),
        method=clean_text(method),
        experiments=clean_text(experiments),
        conclusion=clean_text(conclusion)
    )


def parse_reviews(review_path: str, paper_id: str) -> List[ReviewNode]:
    """从 JSON 文件解析审稿意见，转为 ReviewNode"""
    
    if not os.path.exists(review_path):
        return []
    
    with open(review_path, 'r', encoding='utf-8') as f:
        reviews_data = json.load(f)
    
    reviews = []
    for idx, review in enumerate(reviews_data):
        report = review.get('report', {})
        scores = review.get('scores', {})
        meta = review.get('meta', {})
        
        rid = review.get('rid', '')[:16] if review.get('rid') else f"{paper_id}_review_{idx}"
        
        reviews.append(ReviewNode(
            review_id=rid,
            paper_id=paper_id,
            reviewer=review.get('reviewer'),
            paper_summary=report.get('paper_summary', ''),
            strengths=report.get('summary_of_strengths', ''),
            weaknesses=report.get('summary_of_weaknesses', ''),
            comments=report.get('comments,_suggestions_and_typos', ''),
            overall_score=str(scores.get('overall', '')),
            confidence=str(meta.get('confidence', ''))
        ))
    
    return reviews


def parse_metadata(metadata_path: str) -> Dict:
    """解析元数据文件"""
    if not os.path.exists(metadata_path):
        return {}
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ============================================================
# LLM 调用
# ============================================================
def call_llm(prompt: str, system_prompt: str = "", temperature: float = 0.3, max_retries: int = 3) -> str:
    """调用 LLM API"""
    headers = {
        "Authorization": AUTH_TOKEN,
        "Content-Type": "application/json"
    }
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 2000
    }
    
    for attempt in range(max_retries):
        try:
            print(f"  [API] 正在调用 LLM... (尝试 {attempt + 1}/{max_retries})")
            response = requests.post(API_URL, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
        except Exception as e:
            print(f"  [API] 调用失败: {e}")
            if attempt < max_retries - 1:
                import time
                time.sleep(3)
            else:
                raise
    return ""


def parse_json_response(response: str) -> Dict:
    """解析 LLM 返回的 JSON"""
    json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', response, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        json_str = response
    
    json_str = json_str.strip()
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"  [解析] JSON 解析失败: {e}")
        return {}


# ============================================================
# Paper 抽取任务
# ============================================================
def extract_domain(title: str, keywords: List[str], abstract: str) -> Dict:
    """任务 1：抽取 domain"""
    print("\n📌 任务 1：抽取 domain")
    
    prompt = f"""【输入】
- 论文标题: {title}
- 关键词: {', '.join(keywords) if keywords else '无'}
- 摘要: {abstract}

【任务】
请从以下三个维度分析这篇论文的研究领域：

1. 研究对象：这篇论文主要研究什么类型的数据或问题？（如图像、文本、图结构、时序数据、多模态等）
2. 核心技术：论文使用或改进了什么类型的技术方法？（如 Transformer、扩散模型、强化学习、图神经网络等）
3. 应用场景：论文的成果可以应用于什么实际场景？（如机器翻译、目标检测、推荐系统、对话系统等）

基于以上分析，归纳出这篇论文所属的研究领域（可以有多个）。

【输出格式】
请严格输出 JSON 格式：
{{
  "research_object": "研究对象描述",
  "core_technique": "核心技术描述", 
  "application": "应用场景描述",
  "domains": ["领域1", "领域2"]
}}"""

    system_prompt = "你是一位资深的学术论文分析专家。请严格按照要求的格式输出 JSON。"
    response = call_llm(prompt, system_prompt)
    result = parse_json_response(response)
    print(f"  ✅ domain 抽取完成: {result.get('domains', [])}")
    return result


def extract_ideal(abstract: str, keywords: List[str], introduction: str, method_first_para: str) -> Dict:
    """任务 2：抽取 ideal"""
    print("\n📌 任务 2：抽取 ideal")
    
    # 提取 method 第一段
    method_intro = method_first_para[:1500] if method_first_para else ""
    
    prompt = f"""【输入】
- 摘要: {abstract}
- 关键词: {', '.join(keywords) if keywords else '无'}
- 引言: {introduction[:3000]}
- 方法部分开头: {method_intro}

【任务】
提取这篇论文的核心创新信息。

请回答以下问题：
1. core_idea：这篇论文的核心创新点是什么？用一句话描述（不超过50字）
2. tech_stack：论文使用了哪些关键技术？列出技术名词
3. input_type：论文方法的输入是什么类型的数据或问题？
4. output_type：论文方法的输出是什么类型的结果？

【输出格式】
请严格输出 JSON 格式：
{{
  "core_idea": "一句话描述核心创新点",
  "tech_stack": ["技术1", "技术2"],
  "input_type": "输入类型描述",
  "output_type": "输出类型描述"
}}"""

    system_prompt = "你是一位资深的学术论文分析专家。请严格按照要求的格式输出 JSON。"
    response = call_llm(prompt, system_prompt)
    result = parse_json_response(response)
    print(f"  ✅ ideal 抽取完成: {result.get('core_idea', '')[:50]}...")
    return result


def extract_skeleton(introduction: str, related_work: str, method: str, experiments: str) -> Dict:
    """任务 3：抽取 skeleton"""
    print("\n📌 任务 3：抽取 skeleton")
    
    # 提取各部分的结构性信息
    intro_summary = introduction[:3000] if introduction else ""
    related_summary = related_work[:2000] if related_work else ""
    method_summary = method[:3000] if method else ""
    exp_summary = experiments[:3000] if experiments else ""
    
    prompt = f"""【输入】
- 引言: {intro_summary}
- 相关工作: {related_summary}
- 方法: {method_summary}
- 实验: {exp_summary}

【任务】
分析这篇论文的叙事结构策略，回答以下问题：

1. problem_framing：论文如何引出问题？用了什么开篇策略？（如：从实际痛点出发、从学术gap出发、从应用需求出发等）
2. gap_pattern：论文如何批评现有方法？用了什么逻辑或句式？（如：现有方法忽视了X、现有方法在Y场景下失效等）
3. method_story：方法部分的叙述顺序和策略是什么？（如：先整体后局部、分模块介绍、从简单到复杂等）
4. experiments_story：实验部分的叙述策略是什么？包含哪些类型的实验？（如：主实验+消融+可视化、多数据集验证等）

【输出格式】
请严格输出 JSON 格式：
{{
  "problem_framing": "问题引出策略描述",
  "gap_pattern": "Gap 批评策略描述",
  "method_story": "方法叙述策略描述",
  "experiments_story": "实验叙述策略描述"
}}"""

    system_prompt = "你是一位资深的学术论文分析专家。请严格按照要求的格式输出 JSON。"
    response = call_llm(prompt, system_prompt)
    result = parse_json_response(response)
    print(f"  ✅ skeleton 抽取完成")
    return result


def extract_tricks(introduction: str, method: str, experiments: str) -> List[Dict]:
    """任务 4：抽取 tricks"""
    print("\n📌 任务 4：抽取 tricks")
    
    intro_text = introduction[:3000] if introduction else ""
    method_text = method[:4000] if method else ""
    exp_text = experiments[:4000] if experiments else ""
    
    prompt = f"""【输入】
- 引言部分: {intro_text}
- 方法部分: {method_text}
- 实验部分: {exp_text}

【任务】
请像一位经验丰富的审稿人一样，分析这篇论文在“引言”、“方法描述”和“实验设计”中使用了哪些**写作技巧或包装策略**。

从以下几个维度思考：
1. 说服力：作者用了什么方法让读者相信这个方法有效？
2. 新颖性：作者如何展示这个工作的创新点？
3. 可解释性：作者如何帮助读者理解方法的原理？
4. 完备性：作者如何证明实验是充分的、结论是可靠的？
5. 对比性：作者如何与现有方法进行比较？
6. 叙事结构：作者如何组织全文的逻辑流？如何引入问题、铺垫方法、呼应结论？

请识别出论文中使用的具体技巧，每个技巧需要说明：
- name：技巧名称（用简洁的词语命名）
- type：所属类型（method-level / experiment-level / writing-level）
- purpose：达成目的（这个技巧帮助作者达成什么效果）
- location：具体位置（出现在论文的哪个部分）
- description：简要描述（用一句话描述作者是怎么做的）

【输出格式】
请严格输出 JSON 数组格式：
[
  {{
    "name": "技巧名称",
    "type": "method-level / experiment-level / writing-level",
    "purpose": "目的描述",
    "location": "introduction / method / experiments / 其他",
    "description": "简要描述"
  }}
]"""

    system_prompt = "你是一位资深的学术论文分析专家。请严格按照要求的格式输出 JSON 数组。"
    response = call_llm(prompt, system_prompt)
    result = parse_json_response(response)
    
    if isinstance(result, list):
        print(f"  ✅ tricks 抽取完成: 发现 {len(result)} 个技巧")
        return result
    else:
        print(f"  ⚠️ tricks 解析异常，返回空列表")
        return []



# ============================================================
# 主流程
# ============================================================
def extract_paper_with_reviews(
    paper_path: str, 
    review_path: str, 
    metadata_path: str = None
):
    """抽取单篇论文及其审稿信息，返回 (PaperNode, List[ReviewNode])"""
    
    # 获取 paper_id
    filename = os.path.basename(paper_path)
    paper_id = filename.replace('_paper.json', '')
    
    print("=" * 60)
    print(f"📄 开始抽取: {paper_id}")
    print("=" * 60)
    
    # Step 1: 解析论文
    print("\n🔍 Step 1: 解析论文 JSON")
    sections = parse_json_paper(paper_path)
    print(f"  标题: {sections.title[:60]}...")
    
    # Step 2: 解析审稿
    print("\n🔍 Step 2: 解析审稿意见")
    review_nodes = parse_reviews(review_path, paper_id)
    print(f"  审稿数量: {len(review_nodes)} 条")
    
    # Step 3: 解析元数据
    metadata = {}
    if metadata_path:
        metadata = parse_metadata(metadata_path)
    conference = metadata.get('conference', 'ARR')
    
    # Step 4: LLM 抽取论文信息
    print("\n🤖 Step 3: LLM 抽取论文信息")
    
    domain_result = extract_domain(
        sections.title,
        sections.keywords,
        sections.abstract
    )
    
    ideal_result = extract_ideal(
        sections.abstract,
        sections.keywords,
        sections.introduction,
        sections.method
    )
    
    skeleton_result = extract_skeleton(
        sections.introduction,
        sections.related_work,
        sections.method,
        sections.experiments
    )
    
    tricks_result = extract_tricks(
        sections.introduction,
        sections.method,
        sections.experiments
    )
    
    # Step 5: 构建论文节点
    print("\n📦 Step 4: 构建知识图谱节点")
    
    paper_node = PaperNode(
        paper_id=paper_id,
        title=sections.title,
        conference=conference,
        domain=domain_result,
        ideal=ideal_result,
        skeleton=skeleton_result,
        tricks=tricks_result
    )
    
    print(f"\n✅ 抽取完成")
    print(f"  论文节点: {paper_id}")
    print(f"  审稿节点: {len(review_nodes)} 个")
    
    return paper_node, review_nodes


def batch_extract(
    paper_dir: str, 
    review_dir: str, 
    metadata_dir: str,
    output_dir: str, 
    limit: int = None
):
    """批量抽取，生成 paper 节点和 review 节点"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有 paper 文件
    paper_files = [f for f in os.listdir(paper_dir) if f.endswith('_paper.json')]
    paper_files.sort()
    
    if limit:
        paper_files = paper_files[:limit]
    
    print(f"\n🚀 批量抽取开始")
    print(f"   Paper 目录: {paper_dir}")
    print(f"   Review 目录: {review_dir}")
    print(f"   输出目录: {output_dir}")
    print(f"   待处理: {len(paper_files)} 篇")
    print("=" * 60)
    
    success_count = 0
    fail_count = 0
    all_paper_nodes = []
    all_review_nodes = []
    
    for i, filename in enumerate(paper_files, 1):
        paper_id = filename.replace('_paper.json', '')
        
        paper_path = os.path.join(paper_dir, filename)
        review_path = os.path.join(review_dir, f"{paper_id}_review.json")
        metadata_path = os.path.join(metadata_dir, f"{paper_id}_metadata.json")
        
        # 输出路径
        paper_output_file = os.path.join(output_dir, f"{paper_id}_paper_node.json")
        
        # 跳过已处理
        if os.path.exists(paper_output_file):
            print(f"\n[{i}/{len(paper_files)}] ⏭️ 跳过已处理: {paper_id}")
            continue
        
        print(f"\n[{i}/{len(paper_files)}] 处理中...")
        
        try:
            paper_node, review_nodes = extract_paper_with_reviews(
                paper_path, review_path, metadata_path
            )
            
            # 保存 paper 节点
            with open(paper_output_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(paper_node), f, ensure_ascii=False, indent=2)
            print(f"💾 已保存 paper 节点: {paper_output_file}")
            
            # 保存所有 review 节点到一个文件
            if review_nodes:
                review_output_file = os.path.join(output_dir, f"{paper_id}_reviews.json")
                with open(review_output_file, 'w', encoding='utf-8') as f:
                    json.dump([asdict(r) for r in review_nodes], f, ensure_ascii=False, indent=2)
                print(f"💾 已保存 {len(review_nodes)} 个 review 节点: {review_output_file}")
            
            all_paper_nodes.append(asdict(paper_node))
            all_review_nodes.extend([asdict(r) for r in review_nodes])
            success_count += 1
            
        except Exception as e:
            print(f"❌ 处理失败: {paper_id}, 错误: {e}")
            import traceback
            traceback.print_exc()
            fail_count += 1
            continue
    
    # 保存汇总文件
    if all_paper_nodes:
        paper_summary_file = os.path.join(output_dir, "_all_paper_nodes.json")
        with open(paper_summary_file, 'w', encoding='utf-8') as f:
            json.dump(all_paper_nodes, f, ensure_ascii=False, indent=2)
        print(f"\n💾 Paper 汇总文件: {paper_summary_file}")
    
    if all_review_nodes:
        review_summary_file = os.path.join(output_dir, "_all_review_nodes.json")
        with open(review_summary_file, 'w', encoding='utf-8') as f:
            json.dump(all_review_nodes, f, ensure_ascii=False, indent=2)
        print(f"💾 Review 汇总文件: {review_summary_file}")
    
    print("\n" + "=" * 60)
    print(f"🎉 批量抽取完成！")
    print(f"   成功: {success_count} 篇")
    print(f"   失败: {fail_count} 篇")
    print(f"   Paper 节点: {len(all_paper_nodes)} 个")
    print(f"   Review 节点: {len(all_review_nodes)} 个")
    print("=" * 60)


def main():
    """主函数"""
    import sys
    
    # 默认配置 - ARR_2022
    base_dir = "ARR/ARR_2022"
    paper_dir = f"{base_dir}/ARR_2022_paper"
    review_dir = f"{base_dir}/ARR_2022_review"
    metadata_dir = f"{base_dir}/ARR_2022_metadata"
    output_dir = "extract_result/ARR_2022"  # 统一输出到 extract_result 目录
    limit = 3  # 默认只处理前3篇测试
    
    # 命令行参数
    if len(sys.argv) > 1:
        base_dir = sys.argv[1]
        conf_name = os.path.basename(base_dir)
        paper_dir = f"{base_dir}/{conf_name}_paper"
        review_dir = f"{base_dir}/{conf_name}_review"
        metadata_dir = f"{base_dir}/{conf_name}_metadata"
        output_dir = f"extract_result/{conf_name}"  # 统一输出到 extract_result 目录
    
    if len(sys.argv) > 2:
        limit = int(sys.argv[2]) if sys.argv[2] != 'all' else None
    
    # 单文件模式
    if len(sys.argv) > 1 and sys.argv[1].endswith('.json'):
        paper_path = sys.argv[1]
        paper_id = os.path.basename(paper_path).replace('_paper.json', '')
        
        # 构建 review 和 metadata 路径
        paper_dir = os.path.dirname(paper_path)
        base_dir = os.path.dirname(paper_dir)
        review_path = os.path.join(base_dir, os.path.basename(paper_dir).replace('paper', 'review'), f"{paper_id}_review.json")
        metadata_path = os.path.join(base_dir, os.path.basename(paper_dir).replace('paper', 'metadata'), f"{paper_id}_metadata.json")
        
        paper_node, review_nodes = extract_paper_with_reviews(paper_path, review_path, metadata_path)
        
        # 统一输出到 extract_result 目录
        os.makedirs('extract_result', exist_ok=True)
        
        # 保存 paper 节点
        paper_output_file = os.path.join('extract_result', f"{paper_id}_paper_node.json")
        with open(paper_output_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(paper_node), f, indent=2, ensure_ascii=False)
        print(f"\n💾 Paper 节点已保存到: {paper_output_file}")
        
        # 保存 review 节点到一个文件
        if review_nodes:
            review_output_file = os.path.join('extract_result', f"{paper_id}_reviews.json")
            with open(review_output_file, 'w', encoding='utf-8') as f:
                json.dump([asdict(r) for r in review_nodes], f, indent=2, ensure_ascii=False)
            print(f"💾 {len(review_nodes)} 个 Review 节点已保存到: {review_output_file}")
    else:
        # 批量模式
        batch_extract(paper_dir, review_dir, metadata_dir, output_dir, limit)


if __name__ == "__main__":
    main()

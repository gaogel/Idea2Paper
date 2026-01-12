"""
Idea2Story Pipeline - 从用户 Idea 到可发表的 Paper Story

实现流程:
  Phase 1: Pattern Selection (策略选择)
  Phase 2: Story Generation (结构化生成)
  Phase 3: Multi-Agent Critic & Refine (评审与修正)
  Phase 4: RAG Verification & Pivot (查重与规避)

使用方法:
  python scripts/idea2story_pipeline.py "你的Idea描述"
"""

import json
import os
import re
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# 抑制 urllib3 的 OpenSSL 警告
warnings.filterwarnings("ignore", category=UserWarning, module='urllib3')

# ===================== 配置 =====================
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "output"

# LLM API 配置 (需要配置环境变量或直接设置)
LLM_API_KEY = os.getenv("SILICONFLOW_API_KEY", "")
LLM_API_URL = os.getenv("LLM_API_URL", "https://api.siliconflow.cn/v1/chat/completions")
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")

# Pipeline 配置
class PipelineConfig:
    """Pipeline 配置参数"""
    # Pattern 选择
    SELECT_PATTERN_COUNT = 3  # 选择 3 个不同策略的 Pattern
    CONSERVATIVE_RANK_RANGE = (0, 2)  # 稳健型: Rank 1-3
    INNOVATIVE_CLUSTER_SIZE_THRESHOLD = 10  # 创新型: Cluster Size < 10

    # Critic 阈值
    PASS_SCORE = 7.0  # 评分 >= 7 为通过
    MAX_REFINE_ITERATIONS = 3  # 最多修正 3 轮

    # RAG 查重阈值
    COLLISION_THRESHOLD = 0.75  # 相似度 > 0.75 认为撞车

    # Refinement 策略
    TAIL_INJECTION_RANK_RANGE = (4, 9)  # 长尾注入: Rank 5-10
    HEAD_INJECTION_RANK_RANGE = (0, 2)  # 头部注入: Rank 1-3
    HEAD_INJECTION_CLUSTER_THRESHOLD = 15  # 头部注入: Cluster Size > 15


# ===================== LLM 调用工具 =====================
def call_llm(prompt: str, temperature: float = 0.7, max_tokens: int = 2000) -> str:
    """调用 LLM API"""
    if not LLM_API_KEY:
        print("⚠️  警告: LLM_API_KEY 未配置，使用模拟输出")
        return f"[模拟LLM输出] Prompt: {prompt[:100]}..."

    import requests

    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    try:
        response = requests.post(LLM_API_URL, headers=headers, json=data, timeout=60)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"❌ LLM 调用失败: {e}")
        return ""


# ===================== Phase 1: Pattern Selection =====================
class PatternSelector:
    """Pattern 选择器: 选择多样化的 Pattern"""

    def __init__(self, recalled_patterns: List[Tuple[str, Dict, float]]):
        """
        Args:
            recalled_patterns: [(pattern_id, pattern_info, score), ...]
        """
        self.recalled_patterns = recalled_patterns

    def select(self) -> Dict[str, Tuple[str, Dict]]:
        """选择 3 个不同策略的 Pattern

        Returns:
            {
                'conservative': (pattern_id, pattern_info),
                'innovative': (pattern_id, pattern_info),
                'cross_domain': (pattern_id, pattern_info)
            }
        """
        print("\n" + "=" * 80)
        print("📋 Phase 1: Pattern Selection (策略选择)")
        print("=" * 80)

        selected = {}

        # 1. Conservative (稳健型): 最高分
        conservative = self._select_conservative()
        if conservative:
            selected['conservative'] = conservative
            print(f"\n✅ [稳健型] {conservative[0]}")
            print(f"   名称: {conservative[1].get('name', 'N/A')}")
            print(f"   聚类大小: {conservative[1].get('cluster_size', 0)} 篇")
            print(f"   策略: Score 最高，最符合直觉")

        # 2. Innovative (创新型): Cluster Size 小
        innovative = self._select_innovative(exclude=[conservative[0]] if conservative else [])
        if innovative:
            selected['innovative'] = innovative
            print(f"\n✅ [创新型] {innovative[0]}")
            print(f"   名称: {innovative[1].get('name', 'N/A')}")
            print(f"   聚类大小: {innovative[1].get('cluster_size', 0)} 篇")
            print(f"   策略: Cluster Size < {PipelineConfig.INNOVATIVE_CLUSTER_SIZE_THRESHOLD}，容易产生新颖结合")

        # 3. Cross-Domain (跨域型): 来自路径2或路径3
        cross_domain = self._select_cross_domain(
            exclude=[conservative[0] if conservative else None,
                    innovative[0] if innovative else None]
        )
        if cross_domain:
            selected['cross_domain'] = cross_domain
            print(f"\n✅ [跨域型] {cross_domain[0]}")
            print(f"   名称: {cross_domain[1].get('name', 'N/A')}")
            print(f"   聚类大小: {cross_domain[1].get('cluster_size', 0)} 篇")
            print(f"   策略: 来自领域相关或Paper相似路径")

        print("\n" + "-" * 80)
        print(f"✅ 共选择 {len(selected)} 个 Pattern")
        print("=" * 80)

        return selected

    def _select_conservative(self) -> Optional[Tuple[str, Dict]]:
        """选择稳健型: Score 最高"""
        if not self.recalled_patterns:
            return None

        # 已经按分数排序，选择第一个
        pattern_id, pattern_info, score = self.recalled_patterns[0]
        return (pattern_id, pattern_info)

    def _select_innovative(self, exclude: List[str]) -> Optional[Tuple[str, Dict]]:
        """选择创新型: Cluster Size 最小"""
        candidates = [
            (pid, pinfo, score)
            for pid, pinfo, score in self.recalled_patterns
            if pid not in exclude and
               pinfo.get('cluster_size', 999) < PipelineConfig.INNOVATIVE_CLUSTER_SIZE_THRESHOLD
        ]

        if not candidates:
            # 如果没有符合条件的，选择 Cluster Size 最小的
            candidates = [
                (pid, pinfo, score)
                for pid, pinfo, score in self.recalled_patterns
                if pid not in exclude
            ]
            candidates.sort(key=lambda x: x[1].get('cluster_size', 999))

        if candidates:
            return (candidates[0][0], candidates[0][1])
        return None

    def _select_cross_domain(self, exclude: List[str]) -> Optional[Tuple[str, Dict]]:
        """选择跨域型: 从剩余的中选择"""
        candidates = [
            (pid, pinfo, score)
            for pid, pinfo, score in self.recalled_patterns
            if pid not in exclude
        ]

        if candidates:
            # 选择得分第二高的（不同于 conservative）
            return (candidates[0][0], candidates[0][1])
        return None


# ===================== Phase 2: Story Generation =====================
class StoryGenerator:
    """Story 生成器: 基于 Idea + Pattern 生成结构化 Story"""

    def __init__(self, user_idea: str):
        self.user_idea = user_idea

    def generate(self, pattern_id: str, pattern_info: Dict,
                 constraints: Optional[List[str]] = None,
                 injected_tricks: Optional[List[str]] = None,
                 previous_story: Optional[Dict] = None,
                 review_feedback: Optional[Dict] = None,
                 new_tricks_only: Optional[List[str]] = None) -> Dict:
        """生成 Story (支持初次生成和增量修正)

        Args:
            ...
            previous_story: 上一轮生成的 Story (用于增量修正)
            review_feedback: 上一轮的评审反馈 (用于增量修正)
            new_tricks_only: 本轮新注入的 Trick (用于增量修正的精准注入)
        """

        # 模式判断：如果有上一轮 Story 和反馈，进入【增量修正模式】
        if previous_story and review_feedback:
            print(f"\n📝 修正 Story (基于上一轮反馈 + 新注入技巧)")
            prompt = self._build_refinement_prompt(
                previous_story, review_feedback, new_tricks_only, pattern_info
            )
        else:
            # 【初次生成模式】
            print(f"\n📝 生成 Story (基于 {pattern_id})")

            # 打印调试信息
            if injected_tricks:
                print(f"   🔧 已注入 {len(injected_tricks)} 个 Trick:")
                for trick in injected_tricks:
                    print(f"      - {trick}")
            else:
                print(f"   🔧 本轮无 Trick 注入（首次生成）")

            if constraints:
                print(f"   📌 应用 {len(constraints)} 个约束条件:")
                for constraint in constraints:
                    print(f"      - {constraint}")

            # 构建 Prompt
            prompt = self._build_generation_prompt(
                pattern_info, constraints, injected_tricks
            )

        # 调用 LLM 生成
        print("   ⏳ 调用 LLM 生成...")
        response = call_llm(prompt, temperature=0.7, max_tokens=1500) # 稍微降低温度以保持稳定性

        # 解析输出
        story = self._parse_story_response(response)

        # 如果是修正模式，合并旧 Story 的未修改部分（保底策略）
        if previous_story:
            for key in ['title', 'abstract', 'problem_definition', 'method_skeleton', 'innovation_claims', 'experiments_plan']:
                if not story.get(key) or story.get(key) == "":
                    story[key] = previous_story.get(key)
                    print(f"   ⚠️  字段 '{key}' 为空，已从上一版本恢复")

            # 特殊处理 method_skeleton：如果是字典，尝试转换为字符串
            if isinstance(story.get('method_skeleton'), dict):
                method_dict = story['method_skeleton']
                story['method_skeleton'] = '；'.join(str(v) for v in method_dict.values() if v)
                print(f"   ⚠️  method_skeleton 是字典，已转换为字符串")

            # 特殊处理 innovation_claims：如果不是列表或内容异常，恢复
            if not isinstance(story.get('innovation_claims'), list) or \
               len(story.get('innovation_claims', [])) == 0 or \
               any(claim in ['novelty', 'specific_contributions', 'innovative_points']
                   for claim in story.get('innovation_claims', [])):
                story['innovation_claims'] = previous_story.get('innovation_claims', [])
                print(f"   ⚠️  innovation_claims 异常，已从上一版本恢复")

        # 打印生成的 Story
        self._print_story(story)

        return story

    def _build_refinement_prompt(self, previous_story: Dict,
                               review_feedback: Dict,
                               new_tricks: List[str],
                               pattern_info: Dict) -> str:
        """构建增量修正 Prompt (Editor Mode) - 强调深度方法论融合"""

        # 提取评审意见摘要
        critique_summary = ""
        main_issue = ""
        for review in review_feedback.get('reviews', []):
            critique_summary += f"- {review['reviewer']} ({review['role']}): {review['score']}分. 反馈: {review['feedback'][:250]}...\n"
            if review['role'] == 'Novelty' and review['score'] < 7.0:
                main_issue = "novelty"
            elif review['role'] == 'Methodology' and review['score'] < 7.0 and not main_issue:
                main_issue = "stability"

        # 提取新注入的技术（强调深度融合）
        tricks_instruction = ""
        if new_tricks:
            if "核心技术" in str(new_tricks) or "方法论" in str(new_tricks):
                # 针对方法论注入的特殊指令
                tricks_instruction = "【核心任务：方法论深度重构】\n"
                tricks_instruction += "评审指出当前方法存在问题，需要引入新的技术路线来解决。请参考以下注入的技术和方法论，对核心方法进行**深度改造**：\n\n"
                for trick in new_tricks:
                    tricks_instruction += f"  🔧 {trick}\n"
                tricks_instruction += "\n【重构要求】\n"
                tricks_instruction += "1. **方法论融合**：不要只是在 method_skeleton 末尾添加新步骤，而是要将新技术**深度嵌入**到现有方法的核心逻辑中。\n"
                tricks_instruction += "   - 例如：如果注入\"课程学习\"，应该是\"设计基于难度的课程学习调度器，让模型从易到难学习\"，而不是\"添加课程学习\"。\n"
                tricks_instruction += "   - 例如：如果注入\"对抗训练\"，应该是\"在优化目标中加入对抗扰动正则项，并采用混合训练策略\"，而不是\"使用对抗训练\"。\n"
                tricks_instruction += "2. **技术组合创新**：将注入的技术与现有方法结合，形成新的技术组合，产生 1+1>2 的效果。\n"
                tricks_instruction += "3. **贡献点更新**：在 innovation_claims 中明确指出新技术如何解决了评审指出的问题。\n"
            else:
                tricks_instruction = "【本次修正核心任务】\n请将以下新技巧深度融合到 Method 和 Contribution 中，解决上述评审指出的问题：\n"
                for trick in new_tricks:
                    tricks_instruction += f"  👉 注入: {trick}\n"

        # 根据主要问题添加针对性指导
        specific_guidance = ""
        if main_issue == "novelty":
            specific_guidance = "\n【针对创新性问题的特别指导】\n"
            specific_guidance += "当前方法被评审认为\"创新性不足\"或\"技术组合常见\"。你需要：\n"
            specific_guidance += "1. 在 method_skeleton 中，突出新注入技术的**独特应用方式**，形成与众不同的技术路线。\n"
            specific_guidance += "2. 在 innovation_claims 中，明确指出你的技术组合与现有工作的**本质区别**。\n"
            specific_guidance += "3. 避免使用\"提升性能\"、\"增强效果\"等泛泛而谈的描述，要具体说明技术创新点。\n"
        elif main_issue == "stability":
            specific_guidance = "\n【针对稳定性问题的特别指导】\n"
            specific_guidance += "当前方法被评审认为\"技术细节不足\"或\"稳定性有待验证\"。你需要：\n"
            specific_guidance += "1. 在 method_skeleton 中，添加具体的稳定性保障机制（如正则化、混合策略、鲁棒性设计）。\n"
            specific_guidance += "2. 强调方法的可靠性和实用性，而不仅仅是理论创新。\n"

        prompt = f"""
你是一位顶级 NLP 会议的资深论文作者，擅长将新技术深度融合到现有方法中，形成创新的技术组合。

【当前 Story 版本】
Title: {previous_story.get('title')}
Abstract: {previous_story.get('abstract')}
Problem: {previous_story.get('problem_definition')}
Method: {previous_story.get('method_skeleton')}
Claims: {json.dumps(previous_story.get('innovation_claims', []), ensure_ascii=False)}

【评审专家反馈】(请仔细阅读，保留好评部分，深度改造差评部分)
{critique_summary}

{tricks_instruction}
{specific_guidance}

【修正原则】
1. **保留精华**：评审中得分较高或未被批评的维度（如问题定义、实验计划等），请尽量保留原样。
2. **深度融合**：将新注入的技术**有机地嵌入**到 method_skeleton 的核心逻辑中，形成**统一的技术路线**，而不是逐个罗列技术。
3. **重构而非堆砌**：不要简单地在原有方法后追加新技术，而是要**改造现有步骤**，让新技术成为方法论的有机组成部分。
4. **具体描述**：避免抽象的描述，要具体说明技术如何实现、如何组合、解决什么问题。

【核心要求】：将多个新注入的技术**整合成一个连贯的方法论框架**，而不是分别描述每个技术

【输出要求】
请输出修正后的完整 Story JSON（必须严格遵循以下格式，不要省略任何字段）：

输出格式（纯JSON，不要包含其他文本）：
{{
  "title": "...",
  "abstract": "...",
  "problem_definition": "...",
  "method_skeleton": "步骤1；步骤2；步骤3（必须是字符串，用分号分隔各步骤）",
  "innovation_claims": ["贡献点1", "贡献点2", "贡献点3"],
  "experiments_plan": "..."
}}

注意：
- method_skeleton 必须是字符串类型，描述3-5个方法步骤，用分号分隔，**每个步骤要具体描述技术实现细节**
- innovation_claims 必须是字符串数组，包含3个具体的贡献点，**要突出技术组合的独特性**
- 所有字段都必须填写，不能为空
"""
        return prompt


    def _build_generation_prompt(self, pattern_info: Dict,
                                  constraints: Optional[List[str]],
                                  injected_tricks: Optional[List[str]]) -> str:
        """构建生成 Prompt"""

        # 提取 Pattern 信息
        pattern_name = pattern_info.get('name', '')
        pattern_summary = pattern_info.get('summary', '')
        skeleton_examples = pattern_info.get('skeleton_examples', [])[:2]  # 取前2个示例
        top_tricks = pattern_info.get('top_tricks', [])[:5]  # 取前5个高频技巧

        # 构建 Skeleton 示例文本
        skeleton_text = ""
        for i, sk in enumerate(skeleton_examples, 1):
            skeleton_text += f"\n示例 {i}:\n"
            skeleton_text += f"  标题: {sk.get('title', '')}\n"
            skeleton_text += f"  问题定位: {sk.get('problem_framing', '')[:100]}...\n"
            skeleton_text += f"  方法概述: {sk.get('method_story', '')[:100]}...\n"

        # 构建 Tricks 文本
        tricks_text = ""
        for trick in top_tricks:
            tricks_text += f"  - {trick.get('name', '')} (使用率 {trick.get('percentage', '')})\n"

        # 构建约束文本
        constraints_text = ""
        if constraints:
            constraints_text = "\n【约束条件】\n"
            for constraint in constraints:
                constraints_text += f"  - {constraint}\n"

        # 构建注入 Trick 文本
        injection_text = ""
        if injected_tricks:
            injection_text = "\n【必须融合的技巧】\n"
            for trick in injected_tricks:
                injection_text += f"  - {trick}\n"
            injection_text += "\n注意: 必须将这些技巧自然地融合到方法中，不是简单拼接。\n"

        # 构建注入提示（针对 Novelty 问题强化重构引导）
        emphasis_text = ""
        if injected_tricks:
            if "novelty" in str(injected_tricks).lower() or len(injected_tricks) > 3:
                emphasis_text = "\n⚠️  【极重要：技术重构指令】\n"
                emphasis_text += "当前方案被评审指出“创新性不足”。你必须利用下列注入的技巧对核心方法进行**颠覆性重构**：\n"
                emphasis_text += "1. 不要只是在原有框架上修补，要将这些技巧作为方法论的第一优先级。\n"
                emphasis_text += "2. 在 method_skeleton 中，前两个步骤必须直接体现这些新技巧的应用。\n"
                emphasis_text += "3. 必须在 innovation_claims 中明确指出这些技巧如何解决了原有“平庸组合”的问题。\n"
            else:
                emphasis_text = "\n⚠️  【重要】请务必在方法中充分融合下列技巧，使其成为核心内容，而非简单堆砌：\n"

            for i, trick in enumerate(injected_tricks, 1):
                emphasis_text += f"   {i}. {trick}\n"

        prompt = f"""
你是一位顶级 NLP 会议的论文作者。请基于以下用户 Idea 和写作模板，生成一个结构化的论文 Story。

【用户 Idea】
{self.user_idea}

【写作模板】{pattern_name}
{pattern_summary}

【模板示例】
{skeleton_text}

【高频技巧】
{tricks_text}
{constraints_text}
{injection_text}
{emphasis_text}

【任务要求】
请生成以下结构化内容（JSON格式）。注意：如果提供了【必须融合的技巧】或【重要】部分，你生成的方法必须清晰体现这些要素，使其成为整个方案的核心组成部分。

1. title: 论文标题（简洁、专业、要体现关键创新点）
2. abstract: 摘要（150-200字，概括问题、方法、贡献）
3. problem_definition: 明确的问题定义（50-80字）
4. method_skeleton: 核心方法的步骤（3-5个步骤，每步用分号分隔，必须清晰体现已注入的技巧）
5. innovation_claims: 3个核心贡献点（列表格式，应包含已注入技巧带来的新创新）
6. experiments_plan: 实验设计（50-80字）

输出格式（纯JSON，不要包含其他文本）：
{{
  "title": "...",
  "abstract": "...",
  "problem_definition": "...",
  "method_skeleton": "...",
  "innovation_claims": ["...", "...", "..."],
  "experiments_plan": "..."
}}
"""
        return prompt

    def _parse_story_response(self, response: str) -> Dict:
        """解析 LLM 输出的 Story"""
        try:
            # 1. 尝试清理 Markdown 代码块标记
            clean_response = response.strip()
            if clean_response.startswith("```json"):
                clean_response = clean_response[7:]
            if clean_response.startswith("```"):
                clean_response = clean_response[3:]
            if clean_response.endswith("```"):
                clean_response = clean_response[:-3]

            clean_response = clean_response.strip()

            # 2. 提取 JSON 部分 (寻找最外层的 {})
            start = clean_response.find('{')
            end = clean_response.rfind('}') + 1

            if start >= 0 and end > start:
                json_str = clean_response[start:end]

                # 2.1 预处理：处理非法控制字符（如未转义的换行符）
                # 将字符串内的换行符替换为 \n
                def replace_control_chars(match):
                    s = match.group(0)
                    return s.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')

                # 匹配双引号包裹的内容 (更健壮的正则，处理转义引号)
                json_str = re.sub(r'"((?:[^"\\]|\\.)*)"', replace_control_chars, json_str, flags=re.DOTALL)

                # 2.2 尝试修复常见的 JSON 错误
                try:
                    story = json.loads(json_str)
                    print(f"   ✅ JSON 直接解析成功")
                    return story
                except json.JSONDecodeError as e:
                    # 打印出错位置附近的文本以便调试
                    if hasattr(e, 'pos'):
                        start_pos = max(0, e.pos - 20)
                        end_pos = min(len(json_str), e.pos + 20)
                        print(f"      出错位置上下文: ...{json_str[start_pos:end_pos]}...")

                    # 尝试修复逻辑
                    repaired = json_str

                    # 移除尾部逗号
                    repaired = re.sub(r',(\s*[}\]])', r'\1', repaired)
                    # 修复字段间缺失逗号 (如 "val" "key")
                    repaired = re.sub(r'("\s*)\n?\s*"', r'\1,\n"', repaired)
                    # 修复结构间缺失逗号 (如 } "key" 或 ] "key")
                    repaired = re.sub(r'(}|])\s*\n?\s*"', r'\1,\n"', repaired)

                    try:
                        story = json.loads(repaired)
                        print(f"   ✅ JSON 修复后成功解析")
                        return story
                    except:
                        pass

                # 如果修复失败，抛出异常进入 fallback
                raise json.JSONDecodeError("Failed to parse even after repairs", json_str, 0)
            else:
                print(f"⚠️  无法找到 JSON 结构")
                return self._fallback_parse_story(response)

        except Exception as e:
            print(f"   ⚠️  JSON 解析失败: {e}，尝试 Fallback 解析")
            return self._fallback_parse_story(response)

    def _fallback_parse_story(self, text: str) -> Dict:
        """Fallback: 使用正则提取 Story 字段 (更加健壮)"""
        story = self._default_story()

        # 辅助函数：提取字符串值 (处理复杂情况)
        def extract_str(key):
            # 更加健壮的正则：允许换行、特殊字符、嵌套引号
            # 匹配模式: "key": "value..." 其中 value 可以跨多行，直到遇到未转义的引号后跟逗号或}
            pattern = r'"' + re.escape(key) + r'"\s*:\s*"((?:[^"\\]|\\["\\/bfnrt]|\\u[0-9a-fA-F]{4})*)"'
            match = re.search(pattern, text, re.DOTALL)
            if match:
                val = match.group(1)
                # 处理转义字符
                val = val.replace('\\"', '"')
                val = val.replace('\\n', '\n')
                val = val.replace('\\r', '\r')
                val = val.replace('\\t', '\t')
                val = val.replace('\\\\', '\\')
                return val

            # 尝试另一种提取方式: 寻找 key 之后的首个引号，然后提取到最后一个合理的引号
            alt_pattern = r'"' + re.escape(key) + r'"\s*:\s*"([^"]*(?:\\.[^"]*)*)"'
            match = re.search(alt_pattern, text, re.DOTALL)
            if match:
                val = match.group(1)
                val = val.replace('\\"', '"')
                val = val.replace('\\n', '\n')
                return val

            return None

        # 辅助函数：提取列表
        def extract_list(key):
            pattern = r'"' + re.escape(key) + r'"\s*:\s*\[(.*?)\]'
            match = re.search(pattern, text, re.DOTALL)
            if match:
                content = match.group(1)
                items = []
                # 更加精确地提取列表项
                for m in re.finditer(r'"((?:[^"\\]|\\["\\/bfnrt]|\\u[0-9a-fA-F]{4})*)"', content):
                    item = m.group(1)
                    item = item.replace('\\"', '"')
                    item = item.replace('\\n', '\n')
                    items.append(item)
                return items if items else None
            return None

        # 打印调试信息
        print(f"   📋 使用 Fallback 解析，原始长度: {len(text)} 字符")

        # 尝试提取各字段
        val = extract_str('title')
        if val:
            story['title'] = val
            print(f"      ✓ 提取 title: {val[:60]}...")

        val = extract_str('abstract')
        if val:
            story['abstract'] = val
            print(f"      ✓ 提取 abstract: {val[:60]}...")

        val = extract_str('problem_definition')
        if val:
            story['problem_definition'] = val
            print(f"      ✓ 提取 problem_definition: {val[:60]}...")

        val = extract_str('method_skeleton')
        if val:
            story['method_skeleton'] = val
            print(f"      ✓ 提取 method_skeleton: {val[:60]}...")

        val = extract_str('experiments_plan')
        if val:
            story['experiments_plan'] = val
            print(f"      ✓ 提取 experiments_plan: {val[:60]}...")

        val = extract_list('innovation_claims')
        if val:
            story['innovation_claims'] = val
            print(f"      ✓ 提取 innovation_claims: {len(val)} 项")

        return story

    def _default_story(self) -> Dict:
        """默认 Story 结构"""
        return {
            'title': f"基于 {self.user_idea[:20]} 的创新方法",
            'abstract': f"我们提出了一个新的框架来解决 {self.user_idea}。实验表明有效性。",
            'problem_definition': f"现有方法在 {self.user_idea} 上存在性能不足的问题。",
            'method_skeleton': "第一步：构建基础框架；第二步：设计核心算法；第三步：优化性能。",
            'innovation_claims': [
                "提出新的方法框架",
                "设计高效的算法",
                "在多个数据集上验证有效性"
            ],
            'experiments_plan': "在标准数据集上对比基线方法，验证各组件的有效性。"
        }

    def _print_story(self, story: Dict):
        """打印生成的 Story"""
        print("\n   📄 生成的 Story:")
        print(f"   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"   标题: {story.get('title', '')}")
        print(f"   摘要: {story.get('abstract', '')}")
        print(f"   问题: {story.get('problem_definition', '')}")
        print(f"   方法: {story.get('method_skeleton', '')}")
        print(f"   贡献:")
        for claim in story.get('innovation_claims', []):
            print(f"     - {claim}")
        print(f"   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


# ===================== Phase 3: Multi-Agent Critic =====================
class MultiAgentCritic:
    """多智能体评审团: 三个角色评审 Story"""

    def __init__(self):
        self.reviewers = [
            {'name': 'Reviewer A', 'role': 'Methodology', 'focus': '技术合理性'},
            {'name': 'Reviewer B', 'role': 'Novelty', 'focus': '创新性'},
            {'name': 'Reviewer C', 'role': 'Storyteller', 'focus': '叙事完整性'}
        ]

    def review(self, story: Dict) -> Dict:
        """评审 Story

        Returns:
            {
                'pass': bool,
                'avg_score': float,
                'reviews': [
                    {'reviewer': str, 'role': str, 'score': float, 'feedback': str},
                    ...
                ],
                'main_issue': str,  # 'novelty' | 'stability' | 'interpretability' | 'domain_mismatch'
                'suggestions': List[str]
            }
        """
        print("\n" + "=" * 80)
        print("🔍 Phase 3: Multi-Agent Critic (多智能体评审)")
        print("=" * 80)

        reviews = []
        scores = []

        for reviewer in self.reviewers:
            print(f"\n📝 {reviewer['name']} ({reviewer['role']}) 评审中...")

            review_result = self._single_review(story, reviewer)
            reviews.append(review_result)
            scores.append(review_result['score'])

            print(f"   评分: {review_result['score']:.1f}/10")
            print(f"   反馈: {review_result['feedback']}")

        # 计算平均分
        avg_score = sum(scores) / len(scores)
        passed = avg_score >= PipelineConfig.PASS_SCORE

        # 诊断主要问题
        main_issue, suggestions = self._diagnose_issue(reviews, scores)

        print("\n" + "-" * 80)
        print(f"📊 评审结果: 平均分 {avg_score:.2f}/10 - {'✅ PASS' if passed else '❌ FAIL'}")
        if not passed:
            print(f"🔧 主要问题: {main_issue}")
            print(f"💡 建议: {', '.join(suggestions)}")
        print("=" * 80)

        return {
            'pass': passed,
            'avg_score': avg_score,
            'reviews': reviews,
            'main_issue': main_issue,
            'suggestions': suggestions
        }

    def _single_review(self, story: Dict, reviewer: Dict) -> Dict:
        """单个评审员评审"""

        # 针对 Novelty 角色的特殊指令
        special_instructions = ""
        if reviewer['role'] == 'Novelty':
            special_instructions = """
【特别注意】
作为 Novelty 评审，你需要比较严格，不要被表面的“新颖”词汇迷惑。
1. **批判性评估组合**：仔细思考作者提出的技术是否在近两年的 NLP/CV 顶会中已经泛滥。如果是常见的“A+B”堆砌且缺乏深层理论创新，请给出低分（4-5分）。
2. **拒绝平庸**：如果 Story 只是将现有技术应用到新领域（如“用 BERT 做 X 任务”），而没有针对该领域的独特适配或理论贡献，这不叫创新。
3. **直言不讳**：如果发现是常见套路，请在反馈中明确指出“这种组合已经很常见”或“缺乏实质性创新”。
4. **高分门槛**：只有真正的范式创新、极具启发性的反直觉发现，或对现有方法的根本性改进，才能得到 8 分以上。
"""

        # 构建 Prompt
        prompt = f"""
你是顶级 NLP 会议（如 ACL/ICLR）的**严厉评审专家** {reviewer['name']}，专注于评估{reviewer['focus']}。
你的打分标准非常严格，满分 10 分。6 分以下为不及格（Reject），8 分以上为优秀（Accept）。
{special_instructions}
请评审以下论文 Story：

【标题】{story.get('title', '')}

【摘要】{story.get('abstract', '')}

【问题定义】{story.get('problem_definition', '')}

【方法概述】{story.get('method_skeleton', '')}

【贡献点】
{chr(10).join([f"  - {claim}" for claim in story.get('innovation_claims', [])])}

【实验计划】{story.get('experiments_plan', '')}

请从{reviewer['focus']}的角度进行评审。

【评审要求】
1. 请列出 3 个具体的评估维度。
2. **对每个维度进行打分（1-10分）**，并给出理由。
3. **最终总分（score）必须是各维度分数的综合评估，严禁出现细项分低但总分高的情况。**
4. 如果发现明显缺陷（如创新性不足、方法不合理），请给出低分（<6分）。

输出格式（JSON）：
{{
  "score": 6.5,
  "feedback": "1. 维度A (6.0分): 理由...\\n2. 维度B (7.0分): 理由...\\n\\n总结: ..."
}}
"""

        response = call_llm(prompt, temperature=0.3, max_tokens=800)  # 降低 temperature 提高逻辑一致性

        # 1. 尝试标准 JSON 解析
        try:
            # 尝试清理 Markdown 代码块标记
            clean_response = response.strip()
            if clean_response.startswith("```json"):
                clean_response = clean_response[7:]
            if clean_response.startswith("```"):
                clean_response = clean_response[3:]
            if clean_response.endswith("```"):
                clean_response = clean_response[:-3]

            clean_response = clean_response.strip()

            start = clean_response.find('{')
            end = clean_response.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = clean_response[start:end]

                # 预处理：处理非法控制字符
                def replace_control_chars(match):
                    s = match.group(0)
                    return s.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
                # 更健壮的正则
                json_str = re.sub(r'"((?:[^"\\]|\\.)*)"', replace_control_chars, json_str, flags=re.DOTALL)

                try:
                    result = json.loads(json_str)
                    return {
                        'reviewer': reviewer['name'],
                        'role': reviewer['role'],
                        'score': float(result.get('score', 5.0)),
                        'feedback': result.get('feedback', '')
                    }
                except:
                    # 尝试修复逻辑
                    repaired = json_str
                    repaired = re.sub(r',(\s*[}\]])', r'\1', repaired)
                    repaired = re.sub(r'("\s*)\n?\s*"', r'\1,\n"', repaired)
                    repaired = re.sub(r'(}|])\s*\n?\s*"', r'\1,\n"', repaired)

                    result = json.loads(repaired)
                    return {
                        'reviewer': reviewer['name'],
                        'role': reviewer['role'],
                        'score': float(result.get('score', 5.0)),
                        'feedback': result.get('feedback', '')
                    }
        except Exception as e:
            print(f"   ⚠️  JSON 解析失败: {e}，尝试 Fallback 解析")

        # 2. Fallback: 正则提取分数和反馈
        score = 5.0
        feedback = "评审意见解析失败，请查看原始输出"

        # 尝试匹配分数 "score": 7.5 或 score: 7.5
        score_match = re.search(r'(?:\"|\')?score(?:\"|\')?\s*:\s*([\d\.]+)', response)
        if score_match:
            try:
                score = float(score_match.group(1))
                print(f"      📊 从响应中提取分数: {score}")
            except:
                pass

        # 尝试提取 feedback 字段（更加健壮）
        # 方法1: 匹配 "feedback": "..."
        feedback_match = re.search(
            r'(?:\"|\')?feedback(?:\"|\')?\s*:\s*"((?:[^"\\]|\\["\\/bfnrt]|\\u[0-9a-fA-F]{4})*)"',
            response,
            re.DOTALL
        )
        if feedback_match:
            feedback = feedback_match.group(1)
            feedback = feedback.replace('\\"', '"')
            feedback = feedback.replace('\\n', '\n')
            print(f"      💬 从响应中提取 feedback（模式1）")
        else:
            # 方法2: 更宽松的匹配
            feedback_match = re.search(
                r'(?:\"|\')?feedback(?:\"|\')?\s*:\s*"([^"]*(?:\\.[^"]*)*)"',
                response,
                re.DOTALL
            )
            if feedback_match:
                feedback = feedback_match.group(1)
                feedback = feedback.replace('\\"', '"')
                feedback = feedback.replace('\\n', '\n')
                print(f"      💬 从响应中提取 feedback（模式2）")
            else:
                # 方法3: 如果还是失败，尝试找到所有冒号后的内容，取最长的
                content_matches = list(re.finditer(r':\s*"([^"]*(?:\\.[^"]*)*)"', response))
                if len(content_matches) >= 2:
                    # 假设 score 是第一个，feedback 是第二个
                    feedback = content_matches[1].group(1)
                    feedback = feedback.replace('\\"', '"')
                    feedback = feedback.replace('\\n', '\n')
                    print(f"      💬 从响应中提取 feedback（模式3-启发式）")
                else:
                    # 最后的尝试：使用原始响应的部分内容
                    print(f"      ⚠️  无法精确提取 feedback，使用原始响应摘录")

        return {
            'reviewer': reviewer['name'],
            'role': reviewer['role'],
            'score': score,
            'feedback': feedback
        }

    def _diagnose_issue(self, reviews: List[Dict], scores: List[float]) -> Tuple[str, List[str]]:
        """诊断主要问题

        Returns:
            (main_issue, suggestions)
        """
        # 找出分数最低的评审员
        min_idx = scores.index(min(scores))
        worst_review = reviews[min_idx]

        role = worst_review['role']

        # 打印诊断信息
        print(f"\n   📊 诊断信息:")
        print(f"      分数分布: {scores}")
        print(f"      最低分评审员: {worst_review['reviewer']} ({role}), 分数: {scores[min_idx]}")

        # 根据角色诊断问题
        if role == 'Novelty':
            return 'novelty', ['注入冷门 Trick 提升新颖性', '寻找长尾 Pattern']
        elif role == 'Methodology':
            return 'stability', ['注入成熟稳健的 Trick', '增加鲁棒性验证']
        elif role == 'Storyteller':
            return 'interpretability', ['增加可视化分析', '补充 Case Study']
        else:
            return 'domain_mismatch', ['调整领域适配方法', '增加预处理步骤']


# ===================== Phase 3.5: Refinement Engine =====================
class RefinementEngine:
    """修正引擎: 根据 Critic 反馈进行 Pattern Injection"""

    # 通用/实验性 Trick 列表，这些 Trick 不足以提升技术新颖性
    GENERIC_TRICKS = [
        "消融实验", "多数据集验证", "对比实验", "Case Study", "案例分析",
        "可视化", "Attention 可视化", "参数敏感性分析", "鲁棒性测试",
        "现有方法局限性", "逻辑递进", "叙事结构", "性能提升", "实验验证"
    ]

    def __init__(self, recalled_patterns: List[Tuple[str, Dict, float]]):
        self.recalled_patterns = recalled_patterns
        self.used_patterns = set()  # 追踪已使用过的 Pattern，避免重复

    def refine(self, main_issue: str, suggestions: List[str]) -> List[str]:
        """根据问题类型注入 Trick

        Args:
            main_issue: 'novelty' | 'stability' | 'interpretability' | 'domain_mismatch'
            suggestions: 建议列表

        Returns:
            injected_tricks: List[str] - 注入的 Trick 描述
        """
        print("\n" + "=" * 80)
        print("🔧 Phase 3.5: Refinement (修正注入)")
        print("=" * 80)
        print(f"📌 诊断问题: {main_issue}")
        print(f"💡 建议策略: {', '.join(suggestions)}")

        if main_issue == 'novelty':
            return self._inject_tail_tricks()
        elif main_issue == 'stability':
            return self._inject_head_tricks()
        elif main_issue == 'interpretability':
            return self._inject_explanation_tricks()
        elif main_issue == 'domain_mismatch':
            return self._inject_domain_tricks()
        else:
            return []

    def _inject_tail_tricks(self) -> List[str]:
        """长尾注入: 选择冷门但有特色的 Trick - 注入核心方法论"""
        print("\n🎯 策略: Tail Injection (长尾注入 - 深度方法论融合)")
        print("   目标: 从 Rank 5-10 中选择 Cluster Size < 10 的冷门 Pattern，提取核心方法论")

        # 筛选候选 Pattern
        start, end = PipelineConfig.TAIL_INJECTION_RANK_RANGE
        candidates = []

        for i in range(start, min(end + 1, len(self.recalled_patterns))):
            pattern_id, pattern_info, score = self.recalled_patterns[i]
            # 避免重复使用已使用过的 Pattern
            if pattern_id in self.used_patterns:
                continue
            cluster_size = pattern_info.get('cluster_size', 999)

            if cluster_size < PipelineConfig.INNOVATIVE_CLUSTER_SIZE_THRESHOLD:
                candidates.append((pattern_id, pattern_info, cluster_size))

        if not candidates:
            print("   ⚠️  未找到符合条件的长尾 Pattern，尝试放宽条件...")
            # 放宽条件：在所有召回中找未使用的、聚类最小的
            candidates = [
                (pid, pinfo, pinfo.get('cluster_size', 999))
                for pid, pinfo, _ in self.recalled_patterns
                if pid not in self.used_patterns
            ]
            candidates.sort(key=lambda x: x[2])

        if not candidates:
            print("   ⚠️  所有召回 Pattern 已用尽，注入通用创新算子")
            return ["引入对比学习负采样优化策略", "设计多尺度特征融合机制", "添加自适应动态权重分配"]

        # 选择 Cluster Size 最小的
        candidates.sort(key=lambda x: x[2])
        selected_pattern = candidates[0]

        pattern_id, pattern_info, cluster_size = selected_pattern
        # 记录已使用的 Pattern
        self.used_patterns.add(pattern_id)

        pattern_name = pattern_info.get('name', '')
        pattern_summary = pattern_info.get('summary', '')
        skeleton_examples = pattern_info.get('skeleton_examples', [])

        print(f"\n   ✅ 选择 Pattern: {pattern_id}")
        print(f"      名称: {pattern_name}")
        print(f"      聚类大小: {cluster_size} 篇（冷门）")
        print(f"      已使用 Pattern 数: {len(self.used_patterns)}")

        # 【关键改进】提取 Pattern 的核心方法论，而不是表层 trick
        method_insights = []

        # 1. 从 skeleton_examples 中提取核心方法步骤
        if skeleton_examples:
            for ex in skeleton_examples[:2]:  # 取前2个示例
                method_story = ex.get('method_story', '')
                if method_story:
                    # 提取关键短语（去除通用描述）
                    method_insights.append(method_story[:150])

        # 2. 从 top_tricks 中提取技术性 trick（过滤通用实验 trick）
        tech_tricks = []
        for trick in pattern_info.get('top_tricks', [])[:5]:
            trick_name = trick.get('name', '')
            # 过滤通用 Trick
            is_generic = any(gt in trick_name for gt in self.GENERIC_TRICKS)
            if is_generic:
                continue
            tech_tricks.append(trick_name)
            if len(tech_tricks) >= 2:
                break

        # 3. 构建注入描述（强调方法论融合）
        injection_instructions = []

        if method_insights:
            # 【核心改进】直接注入方法论的具体描述
            for i, insight in enumerate(method_insights[:1], 1):  # 取最相关的一个
                injection_instructions.append(
                    f"【方法论重构】参考 {pattern_name} 的核心技术路线：{insight}"
                )
                print(f"      注入方法论示例 {i}: {insight[:80]}...")

        if tech_tricks:
            # 补充具体技术名称
            injection_instructions.append(
                f"【核心技术】融合 {pattern_name} 的关键技术点：{' + '.join(tech_tricks)}"
            )
            for trick in tech_tricks:
                print(f"      注入核心技术: {trick}")

        if not injection_instructions:
            injection_instructions.append(f"融合 {pattern_name} 的核心思路，重构现有方法论")

        return injection_instructions

    def _inject_head_tricks(self) -> List[str]:
        """头部注入: 选择成熟稳健的 Trick - 注入稳定性方法论"""
        print("\n🎯 策略: Head Injection (头部注入 - 稳定性方法论融合)")
        print(f"   目标: 从 Rank 1-3 中选择 Cluster Size > {PipelineConfig.HEAD_INJECTION_CLUSTER_THRESHOLD} 的成熟 Pattern，提取稳定性技术")

        # 筛选候选 Pattern
        start, end = PipelineConfig.HEAD_INJECTION_RANK_RANGE
        candidates = []

        for i in range(start, min(end + 1, len(self.recalled_patterns))):
            pattern_id, pattern_info, score = self.recalled_patterns[i]
            # 避免重复使用已使用过的 Pattern
            if pattern_id in self.used_patterns:
                continue
            cluster_size = pattern_info.get('cluster_size', 0)

            if cluster_size > PipelineConfig.HEAD_INJECTION_CLUSTER_THRESHOLD:
                candidates.append((pattern_id, pattern_info, cluster_size))

        if not candidates:
            # 如果没有符合条件的，选择 Cluster Size 最大的（且未使用过）
            candidates = [
                (pid, pinfo, pinfo.get('cluster_size', 0))
                for i, (pid, pinfo, _) in enumerate(self.recalled_patterns[:3])
                if pid not in self.used_patterns
            ]
            candidates.sort(key=lambda x: x[2], reverse=True)

        if not candidates:
            # 如果所有头部 Pattern 都用过了，从中间范围选择
            print("   ⚠️  头部 Pattern 已用完，尝试中间范围...")
            candidates = [
                (pid, pinfo, pinfo.get('cluster_size', 0))
                for i, (pid, pinfo, _) in enumerate(self.recalled_patterns[3:6])
                if pid not in self.used_patterns
            ]
            candidates.sort(key=lambda x: x[2], reverse=True)

        if not candidates:
            print("   ⚠️  未找到符合条件的头部 Pattern")
            return []

        selected_pattern = candidates[0]
        pattern_id, pattern_info, cluster_size = selected_pattern
        # 记录已使用的 Pattern
        self.used_patterns.add(pattern_id)

        pattern_name = pattern_info.get('name', '')
        skeleton_examples = pattern_info.get('skeleton_examples', [])

        print(f"\n   ✅ 选择 Pattern: {pattern_id}")
        print(f"      名称: {pattern_name}")
        print(f"      聚类大小: {cluster_size} 篇（成熟）")
        print(f"      已使用 Pattern 数: {len(self.used_patterns)}")

        # 【关键改进】提取稳定性相关的核心技术和方法论
        injection_instructions = []

        # 1. 从 top_tricks 中提取技术性 trick（过滤通用实验 trick）
        tech_tricks = []
        for trick in pattern_info.get('top_tricks', [])[:5]:
            trick_name = trick.get('name', '')
            # 过滤通用 Trick
            is_generic = any(gt in trick_name for gt in self.GENERIC_TRICKS)
            if is_generic:
                continue
            tech_tricks.append(trick_name)
            if len(tech_tricks) >= 2:
                break

        # 2. 从 skeleton_examples 中提取稳定性方法
        stability_methods = []
        if skeleton_examples:
            # 优先提取包含稳定性关键词的方法
            for ex in skeleton_examples[:3]:
                method_story = ex.get('method_story', '')
                if method_story and any(kw in method_story.lower() for kw in ['稳定', '鲁棒', '一致', '对抗', '正则', '混合']):
                    stability_methods.append(method_story[:150])
                    if len(stability_methods) >= 2:
                        break
            # 如果没有匹配到，直接提取前2个示例
            if not stability_methods and skeleton_examples:
                for ex in skeleton_examples[:2]:
                    method_story = ex.get('method_story', '')
                    if method_story:
                        stability_methods.append(method_story[:150])

        # 3. 构建注入指令（直接注入方法论细节）
        if stability_methods:
            # 【核心改进】直接注入稳定性方法的具体描述
            for i, method in enumerate(stability_methods[:1], 1):  # 取最相关的一个
                injection_instructions.append(
                    f"【稳定性方法论】参考 {pattern_name} 的鲁棒性设计：{method}"
                )
                print(f"      注入稳定性方法论 {i}: {method[:80]}...")

        if tech_tricks:
            # 补充具体技术名称
            injection_instructions.append(
                f"【稳定性技术】融合 {pattern_name} 的成熟技术：{' + '.join(tech_tricks)}"
            )
            for trick in tech_tricks:
                print(f"      注入稳定性技术: {trick}")

        if not injection_instructions:
            injection_instructions.append(f"融合 {pattern_name} 的成熟方法，增强技术稳定性")

        return injection_instructions

    def _inject_explanation_tricks(self) -> List[str]:
        """解释性注入: 增加可视化和分析"""
        print("\n🎯 策略: Explanation Injection (解释性注入)")
        print("   目标: 增加可视化和 Case Study 模块")

        tricks = [
            "增加 Attention 权重可视化分析",
            "设计代表性样本的 Case Study",
            "添加消融实验说明各组件贡献"
        ]

        for trick in tricks:
            print(f"      注入 Trick: {trick}")

        return tricks

    def _inject_domain_tricks(self) -> List[str]:
        """领域适配注入: 调整领域相关方法"""
        print("\n🎯 策略: Domain Adaptation Injection (领域适配注入)")
        print("   目标: 增加领域特定的预处理或特征工程")

        tricks = [
            "增加领域特定的数据预处理步骤",
            "设计领域相关的特征提取方法",
            "调整评估指标以适配目标领域"
        ]

        for trick in tricks:
            print(f"      注入 Trick: {trick}")

        return tricks


# ===================== Phase 4: RAG Verification =====================
class RAGVerifier:
    """RAG 查重验证器"""

    def __init__(self, papers: List[Dict]):
        self.papers = papers

    def verify(self, story: Dict) -> Dict:
        """查重验证

        Returns:
            {
                'pass': bool,
                'collision_detected': bool,
                'similar_papers': List[Dict],
                'max_similarity': float
            }
        """
        print("\n" + "=" * 80)
        print("🔎 Phase 4: RAG Verification (查重验证)")
        print("=" * 80)

        # 简单的相似度计算（基于 Method Skeleton）
        method_skeleton = story.get('method_skeleton', '')

        # 处理 method_skeleton 可能是字典的情况
        if isinstance(method_skeleton, dict):
            # 如果是字典，提取所有值并拼接成字符串
            method_skeleton = ' '.join(str(v) for v in method_skeleton.values() if v)
            print(f"   ⚠️  method_skeleton 是字典类型，已转换为字符串")
        elif not isinstance(method_skeleton, str):
            # 如果不是字符串也不是字典，转换为字符串
            method_skeleton = str(method_skeleton)
            print(f"   ⚠️  method_skeleton 类型异常，已转换为字符串")

        similar_papers = []
        max_similarity = 0.0

        print(f"🔍 检索与当前 Story 相似的论文...")
        print(f"   查询: {method_skeleton[:80]}...")

        for paper in self.papers[:50]:  # 仅检查前 50 篇（演示用）
            paper_method = paper.get('skeleton', {}).get('method_story', '')
            if not paper_method:
                continue

            similarity = self._compute_similarity(method_skeleton, paper_method)

            if similarity > 0.3:  # 过滤低相似度
                similar_papers.append({
                    'paper_id': paper.get('paper_id', ''),
                    'title': paper.get('title', ''),
                    'similarity': similarity,
                    'method': paper_method[:100]
                })
                max_similarity = max(max_similarity, similarity)

        # 排序
        similar_papers.sort(key=lambda x: x['similarity'], reverse=True)
        top_similar = similar_papers[:3]

        # 判断是否撞车
        collision_detected = max_similarity > PipelineConfig.COLLISION_THRESHOLD

        print(f"\n📊 查重结果:")
        print(f"   找到 {len(similar_papers)} 篇相似论文")
        print(f"   最高相似度: {max_similarity:.2f}")

        if top_similar:
            print(f"\n   Top-3 相似论文:")
            for i, paper in enumerate(top_similar, 1):
                print(f"   {i}. {paper['title']}")
                print(f"      相似度: {paper['similarity']:.2f}")
                print(f"      方法: {paper['method'][:60]}...")

        if collision_detected:
            print(f"\n   ⚠️  检测到撞车 (相似度 > {PipelineConfig.COLLISION_THRESHOLD})")
        else:
            print(f"\n   ✅ 未检测到撞车")

        print("=" * 80)

        return {
            'pass': not collision_detected,
            'collision_detected': collision_detected,
            'similar_papers': top_similar,
            'max_similarity': max_similarity
        }

    def _compute_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度（Jaccard）"""
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())

        if not tokens1 or not tokens2:
            return 0.0

        intersection = tokens1 & tokens2
        union = tokens1 | tokens2

        return len(intersection) / len(union)

    def generate_pivot_constraints(self, story: Dict, similar_papers: List[Dict]) -> List[str]:
        """生成 Pivot 约束"""
        print("\n🔄 生成 Pivot 约束...")

        constraints = []

        if similar_papers:
            most_similar = similar_papers[0]
            constraints.append(f"禁止使用与《{most_similar['title']}》相同的核心技术")
            constraints.append("将应用场景迁移到新领域（如法律、金融、医疗等）")
            constraints.append("增加额外的约束条件（如无监督、少样本等设定）")

        for constraint in constraints:
            print(f"   - {constraint}")

        return constraints


# ===================== Pipeline Orchestrator =====================
class Idea2StoryPipeline:
    """Idea2Story 主流程编排器"""

    def __init__(self, user_idea: str, recalled_patterns: List[Tuple[str, Dict, float]],
                 papers: List[Dict]):
        self.user_idea = user_idea
        self.recalled_patterns = recalled_patterns
        self.papers = papers

        # 初始化各模块
        self.pattern_selector = PatternSelector(recalled_patterns)
        self.story_generator = StoryGenerator(user_idea)
        self.critic = MultiAgentCritic()
        self.refinement_engine = RefinementEngine(recalled_patterns)
        self.verifier = RAGVerifier(papers)

    def run(self) -> Dict:
        """运行完整 Pipeline

        Returns:
            {
                'success': bool,
                'final_story': Dict,
                'iterations': int,
                'selected_patterns': Dict,
                'review_history': List,
                'refinement_history': List
            }
        """
        print("\n" + "=" * 80)
        print("🚀 Idea2Story Pipeline 启动")
        print("=" * 80)
        print(f"\n【用户 Idea】\n{self.user_idea}\n")

        # Phase 1: Pattern Selection
        selected_patterns = self.pattern_selector.select()

        if not selected_patterns:
            print("❌ 未选择到 Pattern，流程终止")
            return {'success': False}

        # 选择第一个 Pattern 进行生成（优先使用 conservative）
        pattern_type = 'conservative' if 'conservative' in selected_patterns else list(selected_patterns.keys())[0]
        pattern_id, pattern_info = selected_patterns[pattern_type]

        print(f"\n🎯 使用 Pattern: {pattern_type} - {pattern_id}")

        # 初始化迭代变量（必须在第一次生成前初始化）
        iterations = 0
        constraints = None
        injected_tricks = []  # 初始生成时无注入
        review_history = []
        refinement_history = []

        # Phase 2: Initial Story Generation (初始生成)
        current_story = self.story_generator.generate(
            pattern_id, pattern_info, constraints, injected_tricks
        )

        while iterations < PipelineConfig.MAX_REFINE_ITERATIONS:
            iterations += 1
            print(f"\n" + "=" * 80)
            print(f"🔄 迭代轮次: {iterations}/{PipelineConfig.MAX_REFINE_ITERATIONS}")
            print("=" * 80)

            # Phase 3: Multi-Agent Critic
            critic_result = self.critic.review(current_story)
            review_history.append(critic_result)

            if critic_result['pass']:
                print("\n✅ 评审通过，进入查重验证阶段")
                break

            # Phase 3.5: Refinement
            print(f"\n❌ 评审未通过 (平均分: {critic_result['avg_score']:.2f})")

            main_issue = critic_result['main_issue']
            suggestions = critic_result['suggestions']

            # 检查分数是否停滞 (针对 novelty)
            if iterations >= 1 and main_issue == 'novelty':
                # 获取当前和上一次的 Novelty 分数
                curr_novelty_score = next((r['score'] for r in critic_result['reviews'] if r['role'] == 'Novelty'), 0)
                prev_novelty_score = 0
                if len(review_history) >= 2:
                    prev_novelty_score = next((r['score'] for r in review_history[-2]['reviews'] if r['role'] == 'Novelty'), 0)

                if iterations >= 2 and curr_novelty_score <= prev_novelty_score + 0.5:
                    print(f"\n⚠️  检测到新颖性评分停滞或提升缓慢 ({curr_novelty_score:.1f} <= {prev_novelty_score:.1f} + 0.5)")

                    # 全局寻找未使用的、最创新的 Pattern (不再局限于 Phase 1 的 3 个)
                    all_unused = [
                        (pid, pinfo) for pid, pinfo, _ in self.recalled_patterns
                        if pid not in self.refinement_engine.used_patterns
                    ]
                    # 按聚类大小升序排列，优先选冷门的
                    all_unused.sort(key=lambda x: x[1].get('cluster_size', 999))

                    if all_unused:
                        alt_pattern = all_unused[0]
                        pattern_id, pattern_info = alt_pattern
                        print(f"🚀 强制切换到全局最创新 Pattern: {pattern_id} (聚类大小: {pattern_info.get('cluster_size')})")

                        # 切换 Pattern 后，清空之前的注入，重新开始
                        injected_tricks = []
                        print("   已重置注入技巧，基于新 Pattern 重新构建")
                    else:
                        print("   ⚠️  已无更多可用 Pattern，继续在当前路径修正")

            new_tricks = self.refinement_engine.refine(main_issue, suggestions)


            # 累积 Tricks (去重)
            if new_tricks:
                for trick in new_tricks:
                    if trick not in injected_tricks:
                        injected_tricks.append(trick)

            refinement_history.append({
                'iteration': iterations,
                'issue': main_issue,
                'injected_tricks': new_tricks
            })

            print(f"\n🔄 准备重新生成 Story（迭代 {iterations + 1}）...\n")
            time.sleep(1)  # 短暂延迟

            # 判断是否发生了 Pattern 强制切换
            # 如果发生了切换，则视为重新生成（previous_story=None）
            # 否则，视为增量修正
            is_pattern_switch = False
            if iterations >= 2 and main_issue == 'novelty':
                 # 简单的启发式判断：如果 injected_tricks 被清空了，说明发生了切换
                 if not injected_tricks and new_tricks:
                     is_pattern_switch = True

            # 注意：上面的判断逻辑可能不够严谨，更准确的是检查 pattern_id 是否变化
            # 但由于 pattern_id 在循环外定义，这里我们直接根据上下文传递逻辑来处理

            if is_pattern_switch:
                 # 强制切换模式：重新生成
                 current_story = self.story_generator.generate(
                    pattern_id, pattern_info, constraints, injected_tricks
                )
            else:
                # 增量修正模式：传入旧 Story、评审反馈、以及本轮新增的 Trick
                current_story = self.story_generator.generate(
                    pattern_id, pattern_info, constraints, injected_tricks,
                    previous_story=current_story,
                    review_feedback=critic_result,
                    new_tricks_only=new_tricks
                )

        # 检查是否达到最大迭代次数
        if iterations >= PipelineConfig.MAX_REFINE_ITERATIONS and not review_history[-1]['pass']:
            print("\n⚠️  达到最大迭代次数，但评审仍未通过")
            print("   将使用当前版本进入查重验证阶段\n")

        # Phase 4: RAG Verification
        verification_result = self.verifier.verify(current_story)

        if verification_result['collision_detected']:
            print("\n❌ 检测到撞车，触发 Pivot 策略")

            # 生成 Pivot 约束
            constraints = self.verifier.generate_pivot_constraints(
                current_story, verification_result['similar_papers']
            )

            # 重新生成（使用 innovative 或 cross_domain Pattern）
            if 'innovative' in selected_patterns:
                pattern_id, pattern_info = selected_patterns['innovative']
                print(f"\n🔄 切换到创新型 Pattern: {pattern_id}")
            elif 'cross_domain' in selected_patterns:
                pattern_id, pattern_info = selected_patterns['cross_domain']
                print(f"\n🔄 切换到跨域型 Pattern: {pattern_id}")

            current_story = self.story_generator.generate(
                pattern_id, pattern_info, constraints, injected_tricks
            )

            # 重新查重
            verification_result = self.verifier.verify(current_story)

        # 输出最终结果
        success = verification_result['pass']

        print("\n" + "=" * 80)
        print("🎉 Pipeline 完成!")
        print("=" * 80)
        print(f"✅ 状态: {'成功' if success else '需人工审核'}")
        print(f"📊 迭代次数: {iterations}")
        print(f"📝 最终 Story:")
        print(f"   标题: {current_story.get('title', '')}")
        print(f"   摘要: {current_story.get('abstract', '')[:100]}...")
        print("=" * 80)

        return {
            'success': success,
            'final_story': current_story,
            'iterations': iterations,
            'selected_patterns': {k: v[0] for k, v in selected_patterns.items()},
            'review_history': review_history,
            'refinement_history': refinement_history,
            'verification_result': verification_result
        }


# ===================== 主函数 =====================
def main():
    """主函数"""
    # 获取用户输入
    if len(sys.argv) > 1:
        user_idea = " ".join(sys.argv[1:])
    else:
        user_idea = "使用蒸馏技术做Transformer跨领域文本分类任务"

    # 加载召回结果（调用 simple_recall_demo 的结果）
    print("📂 加载数据...")

    try:
        # 加载节点数据
        with open(OUTPUT_DIR / "nodes_pattern.json", 'r', encoding='utf-8') as f:
            patterns = json.load(f)
        with open(OUTPUT_DIR / "nodes_paper.json", 'r', encoding='utf-8') as f:
            papers = json.load(f)

        print(f"  ✓ 加载 {len(patterns)} 个 Pattern")
        print(f"  ✓ 加载 {len(papers)} 个 Paper")

        # 运行召回（复用 simple_recall_demo 的逻辑）
        from simple_recall_demo import main as recall_main
        import io
        from contextlib import redirect_stdout

        # 临时保存原始 argv
        original_argv = sys.argv.copy()
        sys.argv = ['simple_recall_demo.py', user_idea]

        # 运行召回（捕获输出）
        print("\n🔍 运行召回系统...")
        print("-" * 80)

        # 直接导入召回逻辑
        from simple_recall_demo import (
            NODES_IDEA, NODES_PATTERN, NODES_DOMAIN, NODES_PAPER, GRAPH_FILE,
            compute_similarity, TOP_K_IDEAS, TOP_K_DOMAINS, TOP_K_PAPERS,
            FINAL_TOP_K, PATH1_WEIGHT, PATH2_WEIGHT, PATH3_WEIGHT
        )
        import pickle
        import numpy as np

        # 加载数据
        with open(NODES_IDEA, 'r', encoding='utf-8') as f:
            ideas = json.load(f)
        with open(NODES_PATTERN, 'r', encoding='utf-8') as f:
            patterns_data = json.load(f)
        with open(NODES_DOMAIN, 'r', encoding='utf-8') as f:
            domains = json.load(f)
        with open(NODES_PAPER, 'r', encoding='utf-8') as f:
            papers_data = json.load(f)
        with open(GRAPH_FILE, 'rb') as f:
            G = pickle.load(f)

        # 【关键修复】加载完整的 patterns_structured.json 以获取 skeleton_examples
        patterns_structured_file = OUTPUT_DIR / "patterns_structured.json"
        with open(patterns_structured_file, 'r', encoding='utf-8') as f:
            patterns_structured = json.load(f)

        # 构建 pattern_id -> structured_data 的映射
        structured_map = {}
        for p in patterns_structured:
            pattern_id = f"pattern_{p.get('pattern_id')}"
            structured_map[pattern_id] = p

        # 构建索引并合并完整的 skeleton_examples
        idea_map = {i['idea_id']: i for i in ideas}
        pattern_map = {}
        for p in patterns_data:
            pattern_id = p['pattern_id']
            # 合并 nodes_pattern 和 patterns_structured 的数据
            merged_pattern = dict(p)  # 复制基础数据
            if pattern_id in structured_map:
                # 补充完整的 skeleton_examples 和 common_tricks
                merged_pattern['skeleton_examples'] = structured_map[pattern_id].get('skeleton_examples', [])
                merged_pattern['common_tricks'] = structured_map[pattern_id].get('common_tricks', [])
            pattern_map[pattern_id] = merged_pattern

        domain_map = {d['domain_id']: d for d in domains}
        paper_map = {p['paper_id']: p for p in papers_data}

        # 路径1
        path1_scores = defaultdict(float)
        similarities = [(idea['idea_id'], compute_similarity(user_idea, idea['description']))
                       for idea in ideas if compute_similarity(user_idea, idea['description']) > 0]
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_ideas = similarities[:TOP_K_IDEAS]

        for idea_id, similarity in top_ideas:
            idea = idea_map[idea_id]
            pattern_ids = idea.get('pattern_ids', [])
            for pid in pattern_ids:
                path1_scores[pid] += similarity

        # 路径2
        path2_scores = defaultdict(float)
        top_idea = idea_map[top_ideas[0][0]] if top_ideas else None
        domain_scores = []

        if top_idea and G.has_node(top_idea['idea_id']):
            for successor in G.successors(top_idea['idea_id']):
                edge_data = G[top_idea['idea_id']][successor]
                if edge_data.get('relation') == 'belongs_to':
                    domain_id = successor
                    weight = edge_data.get('weight', 0.5)
                    domain_scores.append((domain_id, weight))

        domain_scores.sort(key=lambda x: x[1], reverse=True)
        top_domains = domain_scores[:TOP_K_DOMAINS]

        for domain_id, domain_weight in top_domains:
            for predecessor in G.predecessors(domain_id):
                edge_data = G[predecessor][domain_id]
                if edge_data.get('relation') == 'works_well_in':
                    pattern_id = predecessor
                    effectiveness = edge_data.get('effectiveness', 0.0)
                    confidence = edge_data.get('confidence', 0.0)
                    path2_scores[pattern_id] += domain_weight * max(effectiveness, 0.1) * confidence

        # 路径3
        path3_scores = defaultdict(float)
        similarities = []
        for paper in papers_data:
            paper_idea = paper.get('idea', {}).get('core_idea', '') or paper.get('abstract', '')[:100]
            if not paper_idea:
                continue

            sim = compute_similarity(user_idea, paper_idea)
            if sim > 0.1 and G.has_node(paper['paper_id']):
                reviews = paper.get('reviews', [])
                if reviews:
                    scores = [r.get('rating', 5) for r in reviews]
                    avg_score = np.mean(scores)
                    quality = (avg_score - 1) / 9
                else:
                    quality = 0.5

                combined = sim * quality
                similarities.append((paper['paper_id'], sim, quality, combined))

        similarities.sort(key=lambda x: x[3], reverse=True)
        top_papers = similarities[:TOP_K_PAPERS]

        for paper_id, similarity, quality, combined_weight in top_papers:
            if not G.has_node(paper_id):
                continue
            for successor in G.successors(paper_id):
                edge_data = G[paper_id][successor]
                if edge_data.get('relation') == 'uses_pattern':
                    pattern_id = successor
                    pattern_quality = edge_data.get('quality', 0.5)
                    path3_scores[pattern_id] += combined_weight * pattern_quality

        # 融合
        all_patterns = set(path1_scores.keys()) | set(path2_scores.keys()) | set(path3_scores.keys())
        final_scores = {}
        for pattern_id in all_patterns:
            score1 = path1_scores.get(pattern_id, 0.0) * PATH1_WEIGHT
            score2 = path2_scores.get(pattern_id, 0.0) * PATH2_WEIGHT
            score3 = path3_scores.get(pattern_id, 0.0) * PATH3_WEIGHT
            final_scores[pattern_id] = score1 + score2 + score3

        ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        top_k = ranked[:FINAL_TOP_K]

        # 构建召回结果
        recalled_patterns = [
            (pattern_id, pattern_map.get(pattern_id, {}), score)
            for pattern_id, score in top_k
        ]

        # 恢复 argv
        sys.argv = original_argv

        print("-" * 80)
        print(f"✅ 召回完成: Top-{len(recalled_patterns)} Patterns\n")

        # 运行 Pipeline
        pipeline = Idea2StoryPipeline(user_idea, recalled_patterns, papers)
        result = pipeline.run()

        # 保存结果
        output_file = OUTPUT_DIR / "final_story.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result['final_story'], f, ensure_ascii=False, indent=2)

        print(f"\n💾 最终 Story 已保存到: {output_file}")

        # 保存完整结果
        full_result_file = OUTPUT_DIR / "pipeline_result.json"
        with open(full_result_file, 'w', encoding='utf-8') as f:
            json.dump({
                'user_idea': user_idea,
                'success': result['success'],
                'iterations': result['iterations'],
                'selected_patterns': result['selected_patterns'],
                'final_story': result['final_story'],
                'review_history': result['review_history'],
                'review_summary': {
                    'total_reviews': len(result['review_history']),
                    'final_score': result['review_history'][-1]['avg_score'] if result['review_history'] else 0
                },
                'refinement_summary': {
                    'total_refinements': len(result['refinement_history']),
                    'issues_addressed': [r['issue'] for r in result['refinement_history']]
                },
                'verification_summary': {
                    'collision_detected': result['verification_result']['collision_detected'],
                    'max_similarity': result['verification_result']['max_similarity']
                }
            }, f, ensure_ascii=False, indent=2)

        print(f"💾 完整结果已保存到: {full_result_file}")

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()


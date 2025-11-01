from typing import List, Dict, Any, Tuple
from difflib import SequenceMatcher
import re


class ProofreadComparator:
    """
    比较原始语音稿与校对稿，输出相同与不同之处的结构化结果，并支持 Markdown 渲染。

    granularity:
      - "char": 字符级对齐（中文推荐，精度高）
      - "sentence": 句子级对齐（可读性好）
      - "word": 词元级（对中文不友好，谨慎使用）
    """

    def __init__(self, granularity: str = "char"):
        self.granularity = granularity

    def _tokenize(self, text: str) -> List[str]:
        if self.granularity == "char":
            return list(text)
        elif self.granularity == "sentence":
            # 简易句子切分，保留标点
            # 示例：["今天早上八点半我去医院看病。", "之后我就回家了。"]
            pattern = r'[^。！？!?；;]*[。！？!?；;]?'
            return [s for s in re.findall(pattern, text) if s]
        elif self.granularity == "word":
            # 非中文场景可用；中文不建议
            return re.findall(r'\w+|[^\w\s]', text, flags=re.UNICODE)
        else:
            return list(text)


    def render_html_singleview_highlight(self, original: str, corrected: str) -> str:
        """
        单视图渲染（HTML）：以校对稿为主
        - equal: 直接输出
        - insert: 以浅绿背景高亮新增片段
        - delete: 以浅红背景插入原稿被删除的片段
        - replace: 先浅红显示原片段，再浅绿显示新片段
        """
        from difflib import SequenceMatcher

        a = self._tokenize(original)
        b = self._tokenize(corrected)
        sm = SequenceMatcher(a=a, b=b)

        parts: list[str] = []
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            a_seg = "".join(a[i1:i2])
            b_seg = "".join(b[j1:j2])

            if tag == "equal":
                parts.append(b_seg)
            elif tag == "insert":
                parts.append(f'<span style="background-color:#e6ffe6">{b_seg}</span>')
            elif tag == "delete":
                parts.append(f'<span style="background-color:#ffe6e6">{a_seg}</span>')
            elif tag == "replace":
                parts.append(f'<span style="background-color:#ffe6e6">{a_seg}</span>'
                             f'<span style="background-color:#e6ffe6">{b_seg}</span>')

        html = "".join(parts)
        return f'<div style="font-size: 1.25em; line-height: 1.8;">{html}</div>'


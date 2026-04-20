from __future__ import annotations

import re
import shutil
from pathlib import Path

import pythoncom
import win32com.client as win32
from win32com.client import constants


ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT / "paper"
TEMPLATE_PATH = PAPER_DIR / "本科毕业论文模板.doc"
OUTPUT_PATH = PAPER_DIR / "本科毕业论文.doc"
MARKDOWN_PATH = PAPER_DIR / "毕业论文定稿.md"


META = {
    "school": "深圳大学",
    "thesis_kind": "本科毕业论文（设计）",
    "title": "基于地理分组泛化与异构专家融合的多源农业环境数据 PM2.5 浓度预测研究",
    "author": "曹博宇",
    "major": "计算机科学与技术",
    "college": "计算机与软件学院",
    "student_id": "2022080182",
    "advisor": "张昊迪",
    "advisor_title": "待补充",
    "cover_date": "2026 年 4 月",
    "statement_date": "2026 年 4 月 20 日",
}


ACKNOWLEDGEMENTS = [
    "本课题从选题确定、实验设计、代码实现到论文撰写与修改的全过程，得到了指导教师张昊迪老师的持续指导与帮助。张老师在研究方向把握、实验组织方式、论文结构梳理以及细节表达方面都给予了我耐心而具体的建议，使我能够逐步完成从问题识别、方法设计到结果验证的完整研究过程。在此谨向张老师致以诚挚的感谢。",
    "感谢计算机与软件学院各位老师在本科阶段的教学与培养。课程学习与实验训练为本文的顺利开展奠定了扎实基础，也使我在面对真实的多源数据、复杂的实验流程与持续迭代的论文写作时，能够保持较好的方法意识与工程习惯。",
    "同时，感谢家人和同学在毕业设计期间给予的支持、理解与鼓励。正是因为他们在学习与生活上的陪伴，我才能较为从容地完成本次毕业论文的研究、整理与定稿工作。",
]


def strip_md_inline(text: str) -> str:
    text = text.strip()
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = text.replace("**", "")
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = text.replace("\\(", "(").replace("\\)", ")")
    text = text.replace("\\[", "[").replace("\\]", "]")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def strip_heading_number(text: str) -> str:
    text = strip_md_inline(text)
    text = re.sub(r"^\d+(?:\.\d+)*\s+", "", text)
    return text.strip()


def get_section(lines: list[str], heading: str) -> tuple[int, int]:
    start = None
    for i, line in enumerate(lines):
        if line.strip() == heading:
            start = i + 1
            break
    if start is None:
        raise ValueError(f"Missing heading: {heading}")
    end = len(lines)
    for j in range(start, len(lines)):
        if j > start and lines[j].startswith("## "):
            end = j
            break
    return start, end


def find_heading_index(lines: list[str], heading: str) -> int:
    for i, line in enumerate(lines):
        if line.strip() == heading:
            return i
    raise ValueError(f"Missing heading: {heading}")


def parse_markdown() -> dict:
    lines = MARKDOWN_PATH.read_text(encoding="utf-8").splitlines()
    title = strip_md_inline(lines[0].removeprefix("# ").strip())

    zh_start, zh_end = get_section(lines, "## 摘要")
    en_start, en_end = get_section(lines, "## Abstract")
    body_start = find_heading_index(lines, "## 1 绪论")
    ref_heading_index = find_heading_index(lines, "## 参考文献")
    ref_start, ref_end = get_section(lines, "## 参考文献")
    app_a_start = find_heading_index(lines, "## 附录 A 关键实验资产")

    zh_lines = lines[zh_start:zh_end]
    en_lines = lines[en_start:en_end]

    zh_kw_line = next(line for line in zh_lines if line.startswith("**关键词**"))
    zh_kw = strip_md_inline(zh_kw_line.split("：", 1)[1])
    zh_abstract = " ".join(
        strip_md_inline(line)
        for line in zh_lines
        if line.strip() and not line.startswith("**关键词**")
    )

    en_kw_line = next(line for line in en_lines if line.startswith("**Key Words**"))
    en_kw = strip_md_inline(en_kw_line.split(":", 1)[1])
    en_abstract = " ".join(
        strip_md_inline(line)
        for line in en_lines
        if line.strip() and not line.startswith("**Key Words**")
    )

    body_lines = lines[body_start:ref_heading_index]
    ref_lines = []
    for line in lines[ref_start:ref_end]:
        stripped = line.strip()
        match = re.match(r"^\d+\.\s+(.*)$", stripped)
        if match:
            ref_lines.append(strip_md_inline(match.group(1)))
    appendix_lines = lines[app_a_start:]

    return {
        "title": title,
        "zh_abstract": zh_abstract,
        "zh_keywords": zh_kw,
        "en_abstract": en_abstract,
        "en_keywords": en_kw,
        "body_blocks": parse_blocks(body_lines),
        "references": ref_lines,
        "appendix_blocks": parse_blocks(appendix_lines),
    }


def parse_table(lines: list[str], start: int) -> tuple[dict, int]:
    rows: list[list[str]] = []
    i = start
    while i < len(lines) and lines[i].strip().startswith("|"):
        line = lines[i].strip()
        parts = [strip_md_inline(cell) for cell in line.strip("|").split("|")]
        if not all(re.fullmatch(r"[-:\s]+", cell) for cell in parts):
            rows.append(parts)
        i += 1
    return {"type": "table", "rows": rows}, i


def parse_blocks(lines: list[str]) -> list[dict]:
    blocks: list[dict] = []
    i = 0
    while i < len(lines):
        raw = lines[i]
        line = raw.strip()
        if not line or line == "---":
            i += 1
            continue
        if line.startswith("## "):
            blocks.append({"type": "h1", "text": strip_md_inline(line[3:])})
            i += 1
            continue
        if line.startswith("### "):
            blocks.append({"type": "h2", "text": strip_md_inline(line[4:])})
            i += 1
            continue
        if line.startswith("#### "):
            blocks.append({"type": "h3", "text": strip_md_inline(line[5:])})
            i += 1
            continue
        if line.startswith("```"):
            i += 1
            code_lines: list[str] = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                if lines[i].strip():
                    code_lines.append(lines[i].rstrip())
                i += 1
            if i < len(lines) and lines[i].strip().startswith("```"):
                i += 1
            blocks.append({"type": "code", "lines": code_lines})
            continue
        if re.match(r"^!\[(.*?)\]\((.*?)\)$", line):
            m = re.match(r"^!\[(.*?)\]\((.*?)\)$", line)
            assert m is not None
            blocks.append(
                {
                    "type": "image",
                    "caption": strip_md_inline(m.group(1)),
                    "path": m.group(2).strip(),
                }
            )
            i += 1
            continue
        if line.startswith("|"):
            table_block, new_i = parse_table(lines, i)
            blocks.append(table_block)
            i = new_i
            continue
        if re.match(r"^(表|图)\s*\d+", line):
            blocks.append({"type": "caption", "text": strip_md_inline(line)})
            i += 1
            continue
        if re.match(r"^\d+\.\s+", line) or line.startswith("- "):
            list_type = "ordered" if re.match(r"^\d+\.\s+", line) else "bullet"
            items: list[str] = []
            while i < len(lines):
                candidate = lines[i].strip()
                if list_type == "ordered" and re.match(r"^\d+\.\s+", candidate):
                    items.append(strip_md_inline(candidate))
                    i += 1
                    continue
                if list_type == "bullet" and candidate.startswith("- "):
                    items.append(strip_md_inline(candidate))
                    i += 1
                    continue
                break
            blocks.append({"type": "list", "items": items})
            continue

        para_lines = [line]
        i += 1
        while i < len(lines):
            candidate = lines[i].strip()
            if not candidate or candidate == "---":
                break
            if (
                candidate.startswith("## ")
                or candidate.startswith("### ")
                or candidate.startswith("#### ")
                or candidate.startswith("```")
                or candidate.startswith("|")
                or candidate.startswith("![" )
                or re.match(r"^(表|图)\s*\d+", candidate)
                or re.match(r"^\d+\.\s+", candidate)
                or candidate.startswith("- ")
            ):
                break
            para_lines.append(candidate)
            i += 1
        blocks.append({"type": "paragraph", "text": strip_md_inline(" ".join(para_lines))})
    return blocks


def set_paragraph_text(doc, index: int, text: str) -> None:
    doc.Paragraphs(index).Range.Text = text + "\r"


def apply_cover_and_frontmatter(doc, content: dict) -> None:
    set_paragraph_text(doc, 8, content["title"])
    set_paragraph_text(doc, 9, f"姓名：{META['author']}")
    set_paragraph_text(doc, 10, f"专业：{META['major']}")
    set_paragraph_text(doc, 11, f"学院（部）：{META['college']}")
    set_paragraph_text(doc, 12, f"学号：{META['student_id']}")
    set_paragraph_text(doc, 13, f"指导教师：{META['advisor']}")
    set_paragraph_text(doc, 14, f"职称：{META['advisor_title']}")
    set_paragraph_text(doc, 19, META["cover_date"])

    statement = (
        f"本人郑重声明：所呈交的毕业论文（设计），题目《{content['title']}》是在指导教师的指导下，"
        "由本人独立开展研究工作所取得的成果。除文中已经明确注明引用的内容外，本文不包含任何其"
        "他个人或集体已经发表或撰写过的研究成果。对本文研究做出重要贡献的个人和集体，均已在文"
        "中以明确方式注明。本人愿意对本声明的真实性承担相应责任。"
    )
    set_paragraph_text(doc, 25, statement)
    set_paragraph_text(doc, 31, f"毕业论文（设计）作者签名：{META['author']}")
    set_paragraph_text(doc, 32, f"日期：{META['statement_date']}")

    set_paragraph_text(doc, 85, content["title"])
    set_paragraph_text(doc, 86, f"{META['college']}{META['major']}专业 {META['author']}")
    set_paragraph_text(doc, 87, f"学号：{META['student_id']}")
    set_paragraph_text(doc, 88, f"【摘要】{content['zh_abstract']}")
    set_paragraph_text(doc, 89, f"【关键词】{content['zh_keywords']}")


def rebuild_toc(doc) -> None:
    toc_start = doc.Paragraphs(39).Range.Start
    toc_end = doc.Paragraphs(84).Range.End
    doc.Range(toc_start, toc_end).Delete()

    selection = doc.Application.Selection
    selection.SetRange(toc_start, toc_start)
    type_paragraph(selection, "摘要(关键词)\t4", "TOC 1")

    toc_range = selection.Range
    doc.TablesOfContents.Add(
        toc_range,
        False,
        1,
        3,
        False,
        "",
        True,
        True,
        "毕设1,1,毕设2,2,毕设3,3",
        True,
        False,
        False,
    )


def set_style_and_alignment(selection, style_name: str, alignment: int | None = None) -> None:
    selection.Style = style_name
    if alignment is not None:
        selection.ParagraphFormat.Alignment = alignment


def type_paragraph(selection, text: str, style_name: str = "正文", alignment: int | None = None) -> None:
    set_style_and_alignment(selection, style_name, alignment)
    selection.TypeText(text)
    selection.TypeParagraph()


def insert_table(doc, selection, rows: list[list[str]]) -> None:
    if not rows:
        return
    col_count = max(len(row) for row in rows)
    table = doc.Tables.Add(selection.Range, len(rows), col_count)
    table.Borders.Enable = True
    table.Range.Font.NameFarEast = "宋体"
    table.Range.Font.Name = "Times New Roman"
    table.Range.Font.Size = 9.5
    table.Rows.Alignment = constants.wdAlignRowCenter
    table.AllowAutoFit = True

    for r, row in enumerate(rows, start=1):
        for c in range(1, col_count + 1):
            cell_text = row[c - 1] if c - 1 < len(row) else ""
            table.Cell(r, c).Range.Text = cell_text
            table.Cell(r, c).VerticalAlignment = constants.wdCellAlignVerticalCenter
    for c in range(1, col_count + 1):
        table.Cell(1, c).Range.Bold = True

    selection.SetRange(table.Range.End, table.Range.End)
    selection.TypeParagraph()


def insert_image(selection, image_path: Path, caption: str) -> None:
    set_style_and_alignment(selection, "正文", constants.wdAlignParagraphCenter)
    shape = selection.InlineShapes.AddPicture(str(image_path), False, True)
    if shape.Width > 400:
        shape.Width = 400
    selection.TypeParagraph()
    type_paragraph(selection, caption, "图标", constants.wdAlignParagraphCenter)
    selection.ParagraphFormat.Alignment = constants.wdAlignParagraphJustify


def insert_blocks(doc, selection, blocks: list[dict]) -> None:
    for block in blocks:
        block_type = block["type"]
        if block_type == "h1":
            type_paragraph(selection, strip_heading_number(block["text"]), "毕设1", constants.wdAlignParagraphJustify)
        elif block_type == "h2":
            type_paragraph(selection, strip_heading_number(block["text"]), "毕设2", constants.wdAlignParagraphJustify)
        elif block_type == "h3":
            type_paragraph(selection, strip_heading_number(block["text"]), "毕设3", constants.wdAlignParagraphJustify)
        elif block_type == "paragraph":
            type_paragraph(selection, block["text"], "正文", constants.wdAlignParagraphJustify)
        elif block_type == "list":
            for item in block["items"]:
                type_paragraph(selection, item, "正文", constants.wdAlignParagraphJustify)
        elif block_type == "code":
            for line in block["lines"]:
                type_paragraph(selection, line, "正文", constants.wdAlignParagraphJustify)
        elif block_type == "caption":
            style = "图标" if block["text"].startswith("图 ") else "题注"
            type_paragraph(selection, block["text"], style, constants.wdAlignParagraphCenter)
        elif block_type == "table":
            insert_table(doc, selection, block["rows"])
        elif block_type == "image":
            image_path = (MARKDOWN_PATH.parent / block["path"]).resolve()
            if image_path.exists():
                insert_image(selection, image_path, block["caption"])
            else:
                type_paragraph(selection, f"[缺失图片] {block['caption']}：{image_path}", "正文")


def insert_tail_sections(doc, selection, content: dict) -> None:
    selection.InsertBreak(constants.wdPageBreak)
    type_paragraph(selection, "参考文献", "毕设1", constants.wdAlignParagraphJustify)
    for idx, ref in enumerate(content["references"], start=1):
        type_paragraph(selection, f"{idx}. {ref}", "正文", constants.wdAlignParagraphJustify)

    selection.InsertBreak(constants.wdPageBreak)
    type_paragraph(selection, "致谢", "毕设1", constants.wdAlignParagraphJustify)
    for para in ACKNOWLEDGEMENTS:
        type_paragraph(selection, para, "正文", constants.wdAlignParagraphJustify)

    selection.InsertBreak(constants.wdPageBreak)
    type_paragraph(selection, "Abstract(Key words)", "毕设1", constants.wdAlignParagraphJustify)
    type_paragraph(selection, META["title"], "正文", constants.wdAlignParagraphCenter)
    type_paragraph(selection, f"【Abstract】 {content['en_abstract']}", "正文", constants.wdAlignParagraphJustify)
    type_paragraph(selection, f"【Key words】 {content['en_keywords']}", "正文", constants.wdAlignParagraphJustify)
    type_paragraph(selection, f"Advisor: {META['advisor']}", "正文", constants.wdAlignParagraphJustify)

    selection.InsertBreak(constants.wdPageBreak)
    type_paragraph(selection, "附录", "毕设1", constants.wdAlignParagraphJustify)
    insert_blocks(doc, selection, content["appendix_blocks"])


def find_paragraph_end_by_prefix(doc, prefix: str) -> int:
    for i in range(1, doc.Paragraphs.Count + 1):
        text = re.sub(r"\s+", " ", str(doc.Paragraphs(i).Range.Text)).strip()
        if text.startswith(prefix):
            return doc.Paragraphs(i).Range.End
    raise ValueError(f"Paragraph prefix not found: {prefix}")


def rebuild_body(doc, word, content: dict) -> None:
    start = find_paragraph_end_by_prefix(doc, "【关键词】")
    end = doc.Content.End - 1
    doc.Range(start, end).Delete()

    selection = word.Selection
    selection.SetRange(start, start)
    selection.InsertBreak(constants.wdPageBreak)
    insert_blocks(doc, selection, content["body_blocks"])
    insert_tail_sections(doc, selection, content)


def update_fields(doc) -> None:
    for toc in doc.TablesOfContents:
        toc.Update()
    doc.Fields.Update()
    doc.Repaginate()


def main() -> None:
    content = parse_markdown()
    pythoncom.CoInitialize()
    word = None
    doc = None
    try:
        shutil.copy2(TEMPLATE_PATH, OUTPUT_PATH)

        word = win32.gencache.EnsureDispatch("Word.Application")
        word.Visible = False
        word.DisplayAlerts = 0

        doc = word.Documents.Open(str(OUTPUT_PATH))
        apply_cover_and_frontmatter(doc, content)
        rebuild_toc(doc)
        rebuild_body(doc, word, content)
        update_fields(doc)
        doc.Save()
    finally:
        if doc is not None:
            doc.Close(False)
        if word is not None:
            word.Quit()
        pythoncom.CoUninitialize()


if __name__ == "__main__":
    main()

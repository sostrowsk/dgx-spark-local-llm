#!/usr/bin/env python3
"""
Wandelt die Handouts dieses Repos von HTML nach GitHub-Markdown.

Nur stdlib. Der Konverter ist bewusst auf genau die Bausteine zugeschnitten,
die in den Handouts vorkommen (masthead, toc, tldr, note, details, scroller,
spec, glossary, chart) -- kein allgemeiner HTML-nach-Markdown-Uebersetzer.

SVG-Diagramme werden nicht automatisch uebersetzt: sie tragen im HTML bereits
ausgerechnete Pixelkoordinaten, aus denen sich die Ausgangsdaten nicht sicher
rekonstruieren lassen. Stattdessen setzt der Konverter eine Marke
<!--CHART:n--> und mermaid/*.mmd liefert den handgeschriebenen Ersatz.

Aufruf:  ./html2md.py <datei.html> [...]
"""

import html
import html.parser
import os
import re
import sys
import unicodedata


def github_anchor(text):
    """Bildet die Sprungmarke, die GitHub aus einer Ueberschrift erzeugt."""
    t = unicodedata.normalize("NFC", text).strip().lower()
    t = re.sub(r"[^\w\s-]", "", t, flags=re.UNICODE)
    return re.sub(r"[\s]+", "-", t)


class HandoutParser(html.parser.HTMLParser):
    SKIP = {"style", "script", "svg", "head"}
    INLINE_WRAP = {"strong": "**", "b": "**", "em": "*", "i": "*"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out = []           # fertige Bloecke
        self.buf = []           # laufender Inline-Text
        self.skip_depth = 0
        self.chart_index = 0

        self.list_stack = []    # ('ul'|'ol', counter)
        self.in_toc = False
        self.toc_items = []
        self.headings = []      # (level, text)

        # Tabellenzustand
        self.table = None       # {'caption':str,'rows':[[str]],'head':int}
        self.row = None
        self.cell = None

        self.in_pre = False
        self.pre_buf = []

        self.details_open = 0
        self.note_kind = None
        self.note_head = None
        self.in_summary = False
        self.in_caption = False
        self.in_figcaption = False

        self.dl_mode = None     # 'spec' | 'glossary'
        self.dt_buf = None
        self.in_dt = False
        self.in_dd = False

        self.in_fig = False
        self.span_stack = []

        self.masthead = {}
        self.field = None       # 'eyebrow'|'h1'|'standfirst'|'meta'
        self.in_footer = False
        self.footer = []

    # ---------------------------------------------------------------- Hilfen
    def text(self):
        s = "".join(self.buf)
        self.buf = []
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def emit(self, block):
        if block:
            self.out.append(block)

    def push(self, s):
        if self.skip_depth == 0:
            self.buf.append(s)

    # ------------------------------------------------------------ Starttags
    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        cls = a.get("class", "")

        if tag in self.SKIP:
            if tag == "svg":
                self.chart_index += 1
                self.emit(f"<!--CHART:{self.chart_index}-->")
            self.skip_depth += 1
            return
        if self.skip_depth:
            return

        if tag == "aside" and "toc" in cls:
            self.in_toc = True
        elif tag == "footer":
            self.in_footer = True
        elif tag == "p" and "eyebrow" in cls:
            self.field = "eyebrow"
        elif tag == "p" and "standfirst" in cls:
            self.field = "standfirst"
        elif tag == "div" and "meta" in cls and "meta" not in self.masthead:
            self.field = "meta"
        elif tag in ("h1", "h2", "h3", "h4"):
            self.flush_para()
            self.field = tag
        elif tag == "p":
            self.flush_para()
        elif tag in ("ul", "ol"):
            self.flush_para()
            self.list_stack.append([tag, 0])
        elif tag == "li":
            self.flush_para()
            if self.list_stack:
                self.list_stack[-1][1] += 1
        elif tag == "table":
            self.flush_para()
            self.table = {"caption": "", "rows": [], "head": 0}
        elif tag == "caption":
            self.in_caption = True
        elif tag == "tr" and self.table is not None:
            self.row = []
        elif tag in ("td", "th") and self.table is not None:
            self.cell = {"tag": tag, "buf": []}
            self.buf = []
        elif tag == "pre":
            self.flush_para()
            self.in_pre = True
            self.pre_buf = []
        elif tag == "details":
            self.flush_para()
            self.details_open += 1
        elif tag == "summary":
            self.in_summary = True
        elif tag == "div" and "note" in cls.split():
            self.flush_para()
            self.note_kind = ("error" if "error" in cls
                              else "caution" if "caution" in cls else "note")
        elif tag == "dl":
            self.flush_para()
            self.dl_mode = "spec" if "spec" in cls else "glossary"
        elif tag == "dt":
            self.in_dt = True
            self.buf = []
        elif tag == "dd":
            self.in_dd = True
            self.buf = []
        elif tag == "figcaption":
            self.in_figcaption = True
            self.buf = []
        elif tag == "div" and "fig" == cls.strip():
            # Kennzahl-Kachel: <span class="n">Wert</span><span class="l">Label</span>
            self.flush_para()
            self.in_fig = True
        elif tag == "span":
            c = cls.split()
            self.span_stack.append(c)
            if self.in_fig and "n" in c:
                self.push("**")
            elif self.in_fig and "l" in c:
                self.push(" — ")
        elif tag == "code" and not self.in_pre:
            self.push("`")
        elif tag in self.INLINE_WRAP and not self.in_pre:
            self.push(self.INLINE_WRAP[tag])
        elif tag == "span" and "hl" in cls:
            pass
        elif tag == "br":
            self.push("  \n")
        elif tag == "a" and self.in_toc:
            self._href = a.get("href", "")

    # -------------------------------------------------------------- Endtags
    def handle_endtag(self, tag):
        if tag in self.SKIP:
            self.skip_depth = max(0, self.skip_depth - 1)
            return
        if self.skip_depth:
            return

        if tag == "aside" and self.in_toc:
            self.in_toc = False
        elif tag == "footer":
            self.in_footer = False
            self.footer.append(self.text())
        elif tag in ("h1", "h2", "h3", "h4"):
            t = self.text()
            if self.in_toc:
                self.field = None
                return
            if tag == "h1":
                self.masthead["h1"] = t
            elif self.note_kind:
                # Ueberschrift innerhalb eines Callouts: als fette Zeile in den
                # Blockquote ziehen, sonst steht sie optisch ausserhalb.
                self.note_head = t
            else:
                lvl = int(tag[1])
                self.headings.append((lvl, t))
                self.emit("#" * lvl + " " + t)
            self.field = None
        elif tag == "p":
            f = self.field
            t = self.text()
            if f in ("eyebrow", "standfirst"):
                self.masthead[f] = t
            elif t:
                self.emit(self.decorate(t))
            self.field = None
        elif tag == "div" and self.field == "meta":
            self.masthead["meta"] = self.text()
            self.field = None
        elif tag == "li":
            t = self.text()
            if not t:
                return
            kind, n = self.list_stack[-1] if self.list_stack else ("ul", 1)
            depth = max(0, len(self.list_stack) - 1)
            bullet = f"{n}." if kind == "ol" else "-"
            if self.in_toc:
                self.toc_items.append((t, getattr(self, "_href", "")))
            else:
                self.emit("   " * depth + f"{bullet} {t}")
        elif tag in ("ul", "ol"):
            if self.list_stack:
                self.list_stack.pop()
        elif tag == "caption":
            self.in_caption = False
            if self.table is not None:
                self.table["caption"] = self.text()
        elif tag in ("td", "th") and self.cell is not None:
            val = self.text()
            if self.cell["tag"] == "th":
                self.table["head"] = 1
            self.row.append(val)
            self.cell = None
        elif tag == "tr" and self.row is not None:
            self.table["rows"].append(self.row)
            self.row = None
        elif tag == "thead" and self.table is not None:
            self.table["headrows"] = len(self.table["rows"])
        elif tag == "table" and self.table is not None:
            self.emit(self.render_table(self.table))
            self.table = None
        elif tag == "pre":
            self.in_pre = False
            code = "".join(self.pre_buf).strip("\n")
            self.emit("```bash\n" + code + "\n```")
            self.buf = []
        elif tag == "summary":
            self.in_summary = False
            self.emit("<details>\n<summary>" + self.text() + "</summary>\n")
        elif tag == "details":
            self.details_open = max(0, self.details_open - 1)
            self.emit("</details>")
        elif tag == "span":
            c = self.span_stack.pop() if self.span_stack else []
            if self.in_fig and "n" in c:
                self.push("**")
        elif tag == "div" and self.in_fig:
            self.in_fig = False
            t = self.text()
            if t:
                self.emit("- " + t)
        elif tag == "div" and self.note_kind:
            self.note_kind = None
            self.note_head = None
        elif tag == "dt":
            self.in_dt = False
            self.dt_buf = self.text()
        elif tag == "dd":
            self.in_dd = False
            dd = self.text()
            if self.dl_mode == "spec":
                self.emit(f"**{self.dt_buf}** — {dd}")
            else:
                self.emit(f"**{self.dt_buf}**  \n{dd}")
            self.dt_buf = None
        elif tag == "dl":
            self.dl_mode = None
        elif tag == "figcaption":
            self.in_figcaption = False
            t = self.text()
            if t:
                self.emit("*" + t + "*")
        elif tag == "code" and not self.in_pre:
            self.push("`")
        elif tag in self.INLINE_WRAP and not self.in_pre:
            self.push(self.INLINE_WRAP[tag])

    def handle_data(self, d):
        if self.skip_depth:
            return
        if self.in_pre:
            self.pre_buf.append(d)
        else:
            self.buf.append(d)

    # ------------------------------------------------------------- Ausgabe
    def decorate(self, t):
        """Notes werden zu Blockquotes mit GitHub-Callout-Praefix."""
        if not self.note_kind:
            return t
        marker = {"error": "> [!WARNING]", "caution": "> [!IMPORTANT]",
                  "note": "> [!NOTE]"}[self.note_kind]
        lead = ""
        if self.note_head:
            lead = "> **" + self.note_head + "**\n>\n"
            self.note_head = None
        body = "\n".join("> " + line for line in t.split("\n"))
        return marker + "\n" + lead + body

    def flush_para(self):
        t = self.text()
        if t:
            self.emit(t)

    @staticmethod
    def render_table(tb):
        rows = [r for r in tb["rows"] if r]
        if not rows:
            return ""
        # Ein Pipe im Zellinhalt (z. B. in `docker pull … | tail -5`) wuerde die
        # Tabellensyntax sprengen -- GitHub erwartet dort \| , auch in Backticks.
        rows = [[c.replace("|", r"\|") for c in r] for r in rows]
        width = max(len(r) for r in rows)
        rows = [r + [""] * (width - len(r)) for r in rows]
        nhead = tb.get("headrows", 1 if tb["head"] else 0)
        out = []
        if tb["caption"]:
            out.append("*" + tb["caption"] + "*")
            out.append("")
        if nhead:
            head, body = rows[:nhead], rows[nhead:]
            out.append("| " + " | ".join(head[0]) + " |")
        else:
            body = rows
            out.append("| " + " | ".join([" "] * width) + " |")
        out.append("|" + "---|" * width)
        for r in body:
            out.append("| " + " | ".join(c or " " for c in r) + " |")
        return "\n".join(out)


def convert(path, mermaid_dir):
    src = open(path, encoding="utf8").read()
    p = HandoutParser()
    p.feed(src)

    title = p.masthead.get("h1", os.path.basename(path))
    doc = ["# " + title, ""]
    if p.masthead.get("eyebrow"):
        doc += ["*" + p.masthead["eyebrow"] + "*", ""]
    if p.masthead.get("standfirst"):
        doc += ["> " + p.masthead["standfirst"], ""]
    if p.masthead.get("meta"):
        doc += [p.masthead["meta"], ""]

    # Inhaltsverzeichnis auf GitHub-Anker umschreiben
    if p.toc_items:
        id2anchor = {}
        htexts = [t for lvl, t in p.headings if lvl == 2]
        for (label, href), htext in zip(p.toc_items, htexts):
            id2anchor[href] = "#" + github_anchor(htext)
        doc.append("## Inhalt" if "de." in path else "## Contents")
        doc.append("")
        for i, (label, href) in enumerate(p.toc_items, 1):
            doc.append(f"{i}. [{label}]({id2anchor.get(href, href)})")
        doc.append("")

    body = "\n\n".join(p.out)

    # Diagramm-Marken durch Mermaid ersetzen
    base = os.path.basename(path).replace(".html", "")
    stem = base.replace(".de", "")
    for n in range(1, p.chart_index + 1):
        # sprachspezifische Fassung zuerst, dann die sprachneutrale
        for cand in (f"{base}-chart{n}.mmd", f"{stem}-chart{n}.mmd"):
            mmd = os.path.join(mermaid_dir, cand)
            if os.path.exists(mmd):
                block = "```mermaid\n" + open(mmd, encoding="utf8").read().strip() + "\n```"
                break
        else:
            block = f"> *(Diagramm {n} liegt nur in der HTML-Fassung vor.)*"
        body = body.replace(f"<!--CHART:{n}-->", block)

    doc.append(body)
    if p.footer:
        doc += ["", "---", "", "*" + " ".join(p.footer).strip() + "*"]

    text = "\n".join(doc)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    return text.rstrip() + "\n"


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    mdir = os.path.join(here, "mermaid")
    for f in sys.argv[1:]:
        out = f.replace(".html", ".md")
        open(out, "w", encoding="utf8").write(convert(f, mdir))
        print(f"{f}  ->  {out}")

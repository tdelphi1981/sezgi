"""mkdocs build hook: strip mkdocs-jupyter's own mermaid-support CDN
script from rendered notebook pages.

Why this exists: mkdocs-jupyter (via nbconvert's own
`lab/mermaidjs.html.j2` template macro, inlined by mkdocs-jupyter's
`mkdocs_html/notebook.html.j2`) unconditionally injects a
`<script type="module">` block into *every* rendered notebook page. At
view time that script does `await import("https://cdnjs.cloudflare.com/
ajax/libs/mermaid/...")` (and a second cdnjs URL for the ELK layout
plugin) -- but only after a guard, `if (!diagrams.length) return;`, that
checks for a `.jp-Mermaid > pre.mermaid` element on the page. None of
this site's four notebooks contain a mermaid diagram today, so the
import never actually fires -- but it is still a latent violation of the
site's no-CDN-at-view-time guarantee, and it is NOT caught by the
`privacy`/`offline` plugins already configured in mkdocs.yml: those only
rewrite static `href=`/`src=` HTML attributes (which is how they DO
already vendor the notebook pages' MathJax `<script src=...>` tag), and
this is a bare string literal inside a dynamic `import()` call inside a
`<script>` body, which their attribute-based scanners never see.

mkdocs-jupyter 0.26.3 exposes no plugin config option to disable this
(its `config_options` list has no "mermaid"/"highlight extras" toggle),
and the underlying nbconvert `HTMLExporter.mermaid_js_url` /
`mermaid_layout_elk_js_url` traits are not even `.tag(config=True)`, so
they cannot be overridden through nbconvert's own configuration
mechanism either (unlike `mathjax_url`, which IS `.tag(config=True)` and
which mkdocs-jupyter does override). Post-processing the rendered page
content is the only remaining lever.

If a future notebook adds a real mermaid diagram, this hook would also
strip the (now needed) rendering script -- `py-sezgi/tests/
test_docs_notebooks.py::test_no_notebook_has_mermaid_cells` fails loudly
the day that happens, as a reminder to revisit this.
"""

import re

# Matches the whole macro output emitted by nbconvert's
# lab/mermaidjs.html.j2: the <script type="module"> that imports
# mermaid(+layout-elk) from cdnjs, its accompanying <style> block, and
# the macro's own trailing HTML comment.
_MERMAID_CDN_BLOCK_RE = re.compile(
    r'<script type="module">.*?cdnjs\.cloudflare\.com/ajax/libs/mermaid'
    r".*?</script>"
    r"(?:\s*<style>.*?</style>)?"
    r"(?:\s*<!--\s*End of mermaid configuration\s*-->)?",
    re.DOTALL,
)


def on_page_content(html, page, config, files):
    """Strip the CDN-importing mermaid-support script from notebook
    pages (identified by their `.ipynb` source file)."""
    if not str(page.file.src_uri).endswith(".ipynb"):
        return html
    return _MERMAID_CDN_BLOCK_RE.sub("", html)

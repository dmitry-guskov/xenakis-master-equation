"""Inline the figures as data URIs and emit the final artifact HTML."""
import base64, os, sys

SRC = "page.src.html"
OUT = "/home/claude/xenakis/xenakis_master_equation.html"
# fig 1 is now the live sound-graph explorer in the page itself; figures.py
# still renders it into out/ for reference.
IMG = {"{{FIG2}}": ("out/fig2_chord_space.png", "image/png"),
       "{{FIG3}}": ("out/fig3_quantum.png", "image/png"),
       "{{ANIM}}": ("out/chord_lattice.gif", "image/gif")}

EXTERNAL = "--external" in sys.argv     # for GitHub Pages: assets as separate files

html = open(SRC).read()
if EXTERNAL:
    import shutil
    os.makedirs("docs/assets", exist_ok=True)
    for token, (path, mime) in IMG.items():
        name = os.path.basename(path)
        shutil.copyfile(path, os.path.join("docs/assets", name))
        html = html.replace(token, "assets/" + name)
        print(f"  {token:9s} <- assets/{name}")
    open("docs/index.html", "w").write(html)
    print(f"wrote docs/index.html  ({os.path.getsize('docs/index.html')/1e3:.0f} kB "
          f"+ assets)")
else:
    for token, (path, mime) in IMG.items():
        if not os.path.exists(path):
            print(f"  MISSING {path} -- leaving {token} unresolved")
            continue
        b64 = base64.b64encode(open(path, "rb").read()).decode()
        html = html.replace(token, f"data:{mime};base64,{b64}")
        print(f"  {token:9s} <- {path}  ({len(b64)/1e6:.2f} MB base64)")
    open(OUT, "w").write(html)
    print(f"wrote {OUT}  ({os.path.getsize(OUT)/1e6:.2f} MB)")

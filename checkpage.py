"""Load the artifact in headless Chromium, report console errors, screenshot it."""
import sys, asyncio
from playwright.async_api import async_playwright

PATH = __import__("os").environ.get("PAGE","/home/claude/xenakis/xenakis_master_equation.html")


async def main():
    theme = sys.argv[1] if len(sys.argv) > 1 else "light"
    async with async_playwright() as pw:
        b = await pw.chromium.launch(executable_path="/opt/pw-browsers/chromium",
                                    proxy={"server": "http://127.0.0.1:46519"},
                                    args=["--ignore-certificate-errors"])
        pg = await b.new_page(viewport={"width": 1280, "height": 1000},
                              color_scheme=theme)
        msgs, errs = [], []
        pg.on("console", lambda m: msgs.append((m.type, m.text)))
        pg.on("pageerror", lambda e: errs.append(str(e)))
        await pg.goto("file://" + PATH)
        await pg.wait_for_timeout(6000)
        print("page errors:", errs or "none")
        bad = [m for m in msgs if m[0] in ("error", "warning")]
        print("console:", bad[:8] or "clean")
        # does the simulation advance?
        t1 = await pg.inner_text("#clock")
        await pg.wait_for_timeout(2500)
        t2 = await pg.inner_text("#clock")
        print(f"clock {t1!r} -> {t2!r}   (must advance)")
        print("readout C:", (await pg.inner_text("#roC"))[:90])
        print("readout Q:", (await pg.inner_text("#roQ"))[:90])
        mj = await pg.evaluate("document.querySelectorAll('mjx-container').length")
        print("MathJax containers rendered:", mj)
        h = await pg.evaluate("document.documentElement.scrollWidth - window.innerWidth")
        print("horizontal overflow (px, want <=0):", h)
        await pg.screenshot(path=f"/tmp/page_{theme}_top.png")
        await pg.evaluate("document.getElementById('sim').scrollIntoView()")
        await pg.wait_for_timeout(3000)
        await pg.screenshot(path=f"/tmp/page_{theme}_sim.png")
        await pg.evaluate("document.getElementById('quantum').scrollIntoView()")
        await pg.wait_for_timeout(900)
        await pg.screenshot(path=f"/tmp/page_{theme}_mid.png")
        await b.close()

asyncio.run(main())

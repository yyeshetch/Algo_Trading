"""Quick probe of screener.in HTML structure used during development."""

from __future__ import annotations

import sys
from bs4 import BeautifulSoup

from intraday_engine.research.fundamentals_screener import _http_get


def main(symbol: str = "HCLTECH") -> None:
    url = f"https://www.screener.in/company/{symbol}/consolidated/"
    html = _http_get(url)
    if not html:
        url = f"https://www.screener.in/company/{symbol}/"
        html = _http_get(url)
    print(f"URL: {url}\nlen: {len(html) if html else 0}")
    if not html:
        return
    soup = BeautifulSoup(html, "lxml")

    print("\n== Top ratios ==")
    top = soup.select_one("#top-ratios")
    for li in top.find_all("li") if top else []:
        n = li.select_one(".name")
        v = li.select_one(".value")
        if n and v:
            print(f"  {n.get_text(strip=True):30s} = {v.get_text(' ', strip=True)}")

    print("\n== ranges-table sections (growth/profit) ==")
    for tbl in soup.find_all("table", class_="ranges-table"):
        th = tbl.find("th")
        if th:
            print(f"  heading: '{th.get_text(strip=True)}'")
            for tr in tbl.find_all("tr"):
                cells = [c.get_text(' ', strip=True) for c in tr.find_all(['td', 'th'])]
                if len(cells) >= 2 and cells[0] != th.get_text(strip=True):
                    print(f"     {cells}")

    print("\n== D/E search ==")
    # D/E in #ratios section, or in balance-sheet
    for sect_id in ("ratios", "balance-sheet"):
        sect = soup.select_one(f"#{sect_id}")
        if not sect:
            continue
        for tr in sect.find_all("tr"):
            cells = [c.get_text(' ', strip=True) for c in tr.find_all(['td', 'th'])]
            if cells and any('debt' in c.lower() for c in cells):
                print(f"  ({sect_id}) {cells[:8]}")


if __name__ == "__main__":
    sym = sys.argv[1] if len(sys.argv) > 1 else "HCLTECH"
    main(sym)

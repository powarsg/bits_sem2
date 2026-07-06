"""Insert the assignment metrics table into the design report PDF."""

import fitz

SRC = "ACI-Design-Report.pdf"
DST = "designPS04_G116.pdf"

HEADERS = ["Metric", "Description", "GBFS-h1", "GBFS-h2", "A*-h1"]
ROWS = [
    ["Nodes Expanded", "Total explored nodes", "14", "14", "29"],
    ["Runtime", "Execution time (ms)", "0.1390", "0.1910", "0.2279"],
    ["Memory Usage", "OPEN + CLOSED size", "23", "28", "40"],
    ["Total Path Cost", "Sum of transition costs", "17", "17", "14"],
    ["Path Length", "Total moves", "13", "13", "13"],
    [
        "Heuristic",
        "Heuristic value (at start node)",
        "9.2195",
        "29.4821",
        "9.2195",
    ],
    ["Weather Zones Crossed", "Weather hazard cells entered", "1", "1", "0"],
    ["Weather Penalty", "Total weather transition cost", "4", "4", "0"],
]

COL_WIDTHS = [105, 155, 75, 75, 75]
ROW_HEIGHT = 28
HEADER_HEIGHT = 32
FONT = "helv"
FONT_BOLD = "helv"
TITLE_SIZE = 14
HEADER_SIZE = 9
BODY_SIZE = 8.5


def draw_table(page, x0, y0):
    """Draw the metrics table and return the y position below it."""
    total_width = sum(COL_WIDTHS)
    y = y0

    page.insert_text(
        fitz.Point(x0, y),
        "Metrics Comparison Table (outputPS04.txt)",
        fontname=FONT_BOLD,
        fontsize=TITLE_SIZE,
    )
    y += 24
    page.insert_text(
        fitz.Point(x0, y),
        "Start: (0,0)  |  Goal: (6,7)  |  Group: G116",
        fontname=FONT,
        fontsize=9,
        color=(0.3, 0.3, 0.3),
    )
    y += 18

    def draw_row(cells, yy, height, bold=False, fill=None):
        xx = x0
        for i, (cell, width) in enumerate(zip(cells, COL_WIDTHS)):
            rect = fitz.Rect(xx, yy, xx + width, yy + height)
            if fill:
                page.draw_rect(rect, color=(0.75, 0.75, 0.75), fill=fill)
            else:
                page.draw_rect(rect, color=(0, 0, 0), width=0.6)
            fontsize = HEADER_SIZE if bold else BODY_SIZE
            text = str(cell)
            if i == 0 or bold:
                page.insert_textbox(
                    rect + (4, 4, -4, -4),
                    text,
                    fontname=FONT_BOLD,
                    fontsize=fontsize,
                    align=fitz.TEXT_ALIGN_LEFT,
                )
            else:
                page.insert_textbox(
                    rect + (4, 4, -4, -4),
                    text,
                    fontname=FONT,
                    fontsize=fontsize,
                    align=fitz.TEXT_ALIGN_LEFT if i == 1 else fitz.TEXT_ALIGN_CENTER,
                )
            xx += width
        return yy + height

    y = draw_row(HEADERS, y, HEADER_HEIGHT, bold=True, fill=(0.88, 0.88, 0.88))
    for idx, row in enumerate(ROWS):
        fill = (0.96, 0.96, 0.96) if idx % 2 == 0 else None
        height = 34 if row[0] == "Heuristic" else ROW_HEIGHT
        y = draw_row(row, y, height, fill=fill)

    page.draw_rect(
        fitz.Rect(x0, y0 + 42, x0 + total_width, y),
        color=(0, 0, 0),
        width=1.2,
    )
    return y + 16


def main():
    doc = fitz.open(SRC)
    insert_at = 3  # after section 3, before Results (page 4 in 1-based)
    page = doc.new_page(pno=insert_at, width=595, height=842)

    page.insert_text(
        fitz.Point(40, 40),
        "Group: G116",
        fontname=FONT,
        fontsize=10,
        color=(0.4, 0.4, 0.4),
    )
    page.insert_text(
        fitz.Point(40, 58),
        "4. Results, Analysis and Comparison",
        fontname=FONT_BOLD,
        fontsize=16,
    )

    intro = (
        "The following table records performance metrics extracted from outputPS04.txt "
        "for GBFS with heuristic h1 (Euclidean Distance), GBFS with heuristic h2 "
        "(Bounding-Box Risk-Weighted), and A* with heuristic h1."
    )
    rect = fitz.Rect(40, 78, 555, 118)
    page.insert_textbox(rect, intro, fontname=FONT, fontsize=10, align=fitz.TEXT_ALIGN_LEFT)

    table_bottom = draw_table(page, 40, 130)

    note = (
        "Notes: GBFS-h1 and GBFS-h2 both found a path of 13 moves with total cost 17 "
        "(one weather zone crossed). A* found the optimal path with cost 14 and no "
        "weather penalties, at the expense of expanding more nodes (29 vs 14). "
        "Heuristic values are h(n) at the start node (0,0): Euclidean h1 = 9.2195, "
        "Bounding-Box h2 = 29.4821."
    )
    page.insert_textbox(
        fitz.Rect(40, table_bottom, 555, table_bottom + 70),
        note,
        fontname=FONT,
        fontsize=9.5,
        align=fitz.TEXT_ALIGN_LEFT,
    )

    page_count = doc.page_count
    doc.save(DST)
    doc.close()
    print(f"Created {DST} ({page_count} pages)")


if __name__ == "__main__":
    main()

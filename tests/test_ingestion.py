"""Unit tests for extraction and chunking. No PDF libraries needed."""

import tempfile
from pathlib import Path

import pytest

from manual_rag.ingestion import (
    Chunk,
    chunk_pages,
    clean_line,
    clean_section_title,
    discover_documents,
    extract_pages,
    find_boilerplate,
    is_header,
    is_page_number,
    is_toc_line,
    keep_chunk_language,
)

# Real pymupdf4llm output for page 17 of the GE manual (trimmed).
PAGE_17 = """**<u>Care and Cleaning</u>**

###### **Cooktop Light/Nite Light**

**Replace with a 120 volt, 50 watt (max.) halogen bulb. See back cover for ordering online at GEAppliances.com.**

1. To replace the cooktop light/nite light, first disconnect the power at the main fuse or circuit breaker panel, or pull the plug.

2. Remove the screw from the side of the light compartment cover and lower the cover until it stops.

3. Be sure the bulb is cool before removing. Break the adhesive seal by gently unscrewing the bulb.

4. Screw in the new bulb, then raise the light cover and replace the screw. Connect electrical power to the oven.

###### **Vent Fan**

<!-- Start of picture text -->
Charcoal filter<br><!-- End of picture text -->

The vent fan has two metal reusable vent filters.
"""


def test_real_page_splits_into_correct_sections():
    chunks = chunk_pages([{"page": 17, "text": PAGE_17}], source="734301.pdf")
    assert [c.section for c in chunks] == ["Cooktop Light/Nite Light", "Vent Fan"]


def test_numbered_steps_stay_in_content():
    # Regression: the original notebook turned short steps into section titles,
    # which removed them from the indexed text.
    assert not is_header("1. Press Cook Time.")
    assert not is_header("49-7000115 Rev. 0")
    light = chunk_pages([{"page": 17, "text": PAGE_17}], source="m.pdf")[0]
    for step in ("1. To replace", "2. Remove the screw", "3. Be sure", "4. Screw in"):
        assert step in light.content


def test_bold_sentence_is_content_not_heading():
    light = chunk_pages([{"page": 17, "text": PAGE_17}], source="m.pdf")[0]
    assert "halogen bulb" in light.content
    assert not is_header("**Do not operate the oven empty.**")


def test_markup_removed_but_picture_text_kept():
    assert clean_line("Charcoal filter<br><!-- End of picture text -->") == "Charcoal filter"
    assert clean_line("**<u>Care and Cleaning</u>**") == "**Care and Cleaning**"
    fan = chunk_pages([{"page": 17, "text": PAGE_17}], source="m.pdf")[1]
    assert "Charcoal filter" in fan.content
    assert "<!--" not in fan.content


def test_heading_detection():
    assert is_header("## Care and Cleaning")
    assert is_header("**Care and Cleaning**")
    assert is_header("3.1 Cleaning the Filters")


def test_line_numbers_point_to_content():
    light = chunk_pages([{"page": 17, "text": PAGE_17}], source="m.pdf")[0]
    lines = PAGE_17.split("\n")
    assert "halogen bulb" in lines[light.start_line - 1]
    assert lines[light.end_line - 1].startswith("4. Screw")


def test_max_lines_splits_long_sections():
    text = "# Big Section\n" + "\n".join(f"line number {i} of text" for i in range(65))
    chunks = chunk_pages([{"page": 1, "text": text}], source="m.pdf", max_lines_per_chunk=30)
    assert [len(c.content.split("\n")) for c in chunks] == [30, 30, 5]


def test_section_carries_across_pages():
    pages = [
        {"page": 1, "text": "# Troubleshooting\nOven will not start because the fuse blew."},
        {"page": 2, "text": "Door not securely closed, open and close it again."},
    ]
    assert chunk_pages(pages, source="m.pdf")[1].section == "Troubleshooting"


def test_short_fragments_dropped():
    assert chunk_pages([{"page": 1, "text": "Rev. 0\n"}], source="m.pdf") == []


def test_language_filter():
    en = "Press the Start button to begin cooking the food and wait for the signal."
    es = "Presione el botón para iniciar la cocción de los alimentos en el horno y espere la señal."
    assert keep_chunk_language(en, ["en"])
    assert not keep_chunk_language(es, ["en"])
    assert keep_chunk_language("Popcorn Beverage Reheat Potato", ["en"])  # too short, kept
    assert keep_chunk_language(es, None)  # no filter configured


def test_chunk_id_is_deterministic_and_unique():
    a = Chunk(1, "S", 1, 5, "x", "a.pdf")
    b = Chunk(1, "S", 1, 5, "y", "a.pdf")
    c = Chunk(1, "S", 1, 5, "x", "b.pdf")
    assert a.chunk_id == b.chunk_id
    assert a.chunk_id != c.chunk_id
    assert "content" not in a.to_metadata()


def test_clean_section_title():
    assert clean_section_title("###### **Cooktop Light/Nite Light**") == "Cooktop Light/Nite Light"
    assert clean_section_title("#") == "General"


def test_broken_glyphs_and_mark_tags_removed():
    raw = "- \ufffd\ufffd <mark>a</mark> **\ufffd** Read \ufffd\ufffd\ufffd all instructions."
    assert clean_line(raw) == "- a Read all instructions."
    assert clean_line("- \ufffd\ufffd\ufffd") == ""


def test_repeated_footer_and_page_numbers_removed():
    pages = [
        {"page": n, "text": f"# Topic {n}\nUseful instruction number {n} for this page.\n{n}\n49-7000115 Rev. 0"}
        for n in range(1, 7)
    ]
    assert find_boilerplate(pages) == {"49-7000115 rev. 0"}
    chunks = chunk_pages(pages, source="m.pdf")
    assert len(chunks) == 6
    for c in chunks:
        assert "Rev. 0" not in c.content
        assert c.content.startswith("Useful instruction")
        assert "\n" not in c.content  # page number line was dropped too


def test_page_number_detection():
    for line in ("14", "Page 14", "14 of 22", "3/10"):
        assert is_page_number(line)
    for line in ("1. Press Start.", "120 volt bulb", "49-7000115 Rev. 0"):
        assert not is_page_number(line)


def test_boilerplate_needs_enough_pages():
    pages = [{"page": 1, "text": "Footer"}, {"page": 2, "text": "Footer"}]
    assert find_boilerplate(pages) == set()


# --- Phase 8: more formats and folders ------------------------------------- #
def test_text_manual_splits_pages_on_form_feed():
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "router.txt"
        f.write_text("# Setup\nPlug in the router and wait.\f# Reset\nHold the reset button for 10 seconds.")
        pages = extract_pages(f)
        assert [p["page"] for p in pages] == [1, 2]
        chunks = chunk_pages(pages, source=f.name)
        assert (chunks[1].section, chunks[1].page) == ("Reset", 2)


def test_docx_headings_become_sections():
    docx = pytest.importorskip("docx")  # skipped if python-docx isn't installed
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "kettle.docx"
        doc = docx.Document()
        doc.add_heading("Descaling", level=1)
        doc.add_paragraph("Fill the kettle with equal parts vinegar and water.")
        doc.add_heading("Warranty", level=2)
        doc.add_paragraph("This kettle is covered for two years from purchase.")
        doc.save(f)
        chunks = chunk_pages(extract_pages(f), source=f.name)
        assert [c.section for c in chunks] == ["Descaling", "Warranty"]
        assert "vinegar" in chunks[0].content


def test_unsupported_file_type_rejected():
    with tempfile.TemporaryDirectory() as d:
        bad = Path(d) / "manual.xyz"
        bad.write_text("hello")
        with pytest.raises(ValueError):
            extract_pages(bad)


def test_discover_documents_finds_supported_files_recursively():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "kitchen").mkdir()
        for name in ["a.pdf", "kitchen/b.docx", "c.txt", "notes.xyz", ".hidden.pdf", ".gitkeep"]:
            (root / name).write_text("x")
        names = sorted(p.name for p in discover_documents(root))
        assert names == ["a.pdf", "b.docx", "c.txt"]


# --- Table of contents removal --------------------------------------------- #
def test_toc_line_detection():
    # Real lines from the Mr. Coffee and GE manuals
    for line in (
        "Warranty information . . . . . . . . . . . . . . . . . . 9 .",
        "Cleaning Instructions . . . . . . . . . . . . . . . . . . , 8",
        "lntroduction............................................... 4",
        "**SAFETY INFORMATION** . . . . . . . . . .3",
    ):
        assert is_toc_line(line), line
    for line in (
        "1. Press Cook Time.",
        "Let solution stand in decanter for about 20 minutes. Discard..",
        "Wait... then press Start 2",
        "Soft Water Every 80 Brew Cycles",
    ):
        assert not is_toc_line(line), line


def test_toc_lines_dropped_but_intro_kept():
    page = {"page": 4, "text": (
        "# INTRODUCTION\n"
        "This Coffeemaker is specially designed to give you the ultimate coffee brewing experience.\n"
        "# TABLE OF CONTENTS\n"
        "If Service Is Required . . . . . . . . . . . . . 3\n"
        "Warranty information . . . . . . . . . . . . . . 9\n"
    )}
    chunks = chunk_pages([page], source="coffee.pdf")
    text = " ".join(c.content for c in chunks)
    assert "ultimate coffee brewing" in text
    assert "Warranty information" not in text
    assert chunk_pages([page], source="c.pdf", remove_toc=False)[-1].content.startswith("If Service")

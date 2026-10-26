"""Generate the fictional sample product sheet used for the Q&A demo.

Writes a minimal single-page PDF by hand so no extra dependency is needed.
Run: python scripts/make_sample_pdf.py
"""
from pathlib import Path
import textwrap

OUT = Path(__file__).resolve().parent.parent / "sample_docs" / "hydra_drench_serum_info.pdf"

CONTENT = """\
# Hydra Drench Serum - Product Information Sheet
Brand: Dewleaf (fictional demo brand). This document is sample data for the SkinSense AI demo.

# Product overview
Hydra Drench Serum is a lightweight, water-based serum. It is designed to help skin feel hydrated and plump. Net contents: 30 ml in a glass dropper bottle.

# Full ingredient list (INCI)
Aqua (Water), Glycerin, Sodium Hyaluronate, Panthenol, Trehalose, Pentylene Glycol, Hydroxyethylcellulose, Sodium Benzoate, Citric Acid.

# Fragrance
The formula contains no added fragrance and no essential oils.

# How to use
Apply two to three drops to clean, dry skin in the morning and evening. Follow with a moisturiser. In the morning, finish with sunscreen.

# Storage
Store below 25 degrees Celsius, away from direct sunlight. Use within 12 months after opening.

# Precautions
For external use only. Avoid contact with eyes. Do a patch test before first use and stop using the product if discomfort occurs.

# Disclaimer
This is a cosmetic product. It is not intended to diagnose, treat, cure or prevent any disease or skin condition.
"""


def _esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def build_stream() -> bytes:
    ops = ["BT", "50 800 Td"]
    for raw in CONTENT.splitlines():
        if raw.startswith("# "):
            ops += ["0 -10 Td", "/F2 13 Tf", f"({_esc(raw[2:])}) Tj", "0 -18 Td"]
            continue
        for line in textwrap.wrap(raw, 92) or [""]:
            ops += ["/F1 10.5 Tf", f"({_esc(line)}) Tj", "0 -14 Td"]
    ops.append("ET")
    return "\n".join(ops).encode("latin-1")


def build_pdf() -> bytes:
    stream = build_stream()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
        b"/Resources << /Font << /F1 4 0 R /F2 5 0 R >> >> /Contents 6 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, obj in enumerate(objects, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + obj + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % o for o in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref)
    return bytes(out)


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(build_pdf())
    print(f"Wrote {OUT}")

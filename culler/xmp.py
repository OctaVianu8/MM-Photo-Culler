"""XMP sidecars: xmp:Rating + xmp:Label.

Why a colour label for pick/reject: Lightroom Classic keeps its pick/reject
*flags* only in the catalog - they are never read from (or written to) XMP.
Rating and Label are. So picks get Label "Green" and rejects "Red" (both
configurable); in Lightroom, filter by label and press P / X once.

An existing sidecar (Lightroom develop settings, keywords...) is merged, never
replaced: only xmp:Rating and xmp:Label are touched and everything else is
round-tripped.
"""
from __future__ import annotations

import io
import xml.etree.ElementTree as ET

NS_X = "adobe:ns:meta/"
NS_RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
NS_XMP = "http://ns.adobe.com/xap/1.0/"
RATING = f"{{{NS_XMP}}}Rating"
LABEL = f"{{{NS_XMP}}}Label"
DESC = f"{{{NS_RDF}}}Description"

HEADER = '<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
FOOTER = '\n<?xpacket end="w"?>\n'


class XmpError(Exception):
    pass


def _parse(text: str) -> ET.Element:
    try:
        for _, (prefix, uri) in ET.iterparse(io.StringIO(text), events=["start-ns"]):
            try:
                ET.register_namespace(prefix, uri)     # keep the file's own prefixes
            except ValueError:
                pass
        root = ET.fromstring(text)
    except ET.ParseError as e:
        raise XmpError(f"existing sidecar is not valid XML: {e}") from e
    if root.find(f".//{{{NS_RDF}}}RDF") is None:
        raise XmpError("existing sidecar has no rdf:RDF element")
    return root


def read(text: str) -> dict:
    """Current rating/label in a sidecar (attribute or element form)."""
    root = _parse(text)
    out = {"rating": None, "label": None}
    for d in root.iter(DESC):
        for key, tag in (("rating", RATING), ("label", LABEL)):
            v = d.get(tag)
            if v is None and d.find(tag) is not None:
                v = d.find(tag).text
            if v is not None and out[key] is None:
                out[key] = v.strip()
    if out["rating"] is not None:
        try:
            out["rating"] = int(float(out["rating"]))
        except ValueError:
            out["rating"] = None
    return out


def build(existing: str | None, rating: int | None, label: str | None,
          owned_labels: tuple[str, ...] = ()) -> str:
    """Return sidecar text with rating/label applied.

    rating None -> an existing rating is kept (except a -1 we wrote earlier).
    label None  -> an existing label is kept unless it is one of ours
                   (owned_labels), i.e. you un-picked the photo since.
    """
    for prefix, uri in (("x", NS_X), ("rdf", NS_RDF), ("xmp", NS_XMP)):
        ET.register_namespace(prefix, uri)
    if existing and existing.strip():
        root = _parse(existing)
        cur = read(existing)
    else:
        root = ET.Element(f"{{{NS_X}}}xmpmeta", {f"{{{NS_X}}}xmptk": "MM Photo Culler"})
        rdf = ET.SubElement(root, f"{{{NS_RDF}}}RDF")
        ET.SubElement(rdf, DESC, {f"{{{NS_RDF}}}about": ""})
        cur = {"rating": None, "label": None}

    if rating is None and cur["rating"] is not None and cur["rating"] >= 0:
        rating = cur["rating"]
    if label is None and cur["label"] and cur["label"] not in owned_labels:
        label = cur["label"]

    rdf = root.find(f".//{{{NS_RDF}}}RDF")
    descs = list(rdf.iter(DESC))
    if not descs:
        descs = [ET.SubElement(rdf, DESC, {f"{{{NS_RDF}}}about": ""})]
    for d in descs:
        for tag in (RATING, LABEL):
            d.attrib.pop(tag, None)
            for child in d.findall(tag):
                d.remove(child)
    if rating is not None:
        descs[0].set(RATING, str(int(rating)))
    if label:
        descs[0].set(LABEL, label)
    ET.indent(root, space=" ")
    return HEADER + ET.tostring(root, encoding="unicode") + FOOTER

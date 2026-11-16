import xml.etree.ElementTree as ET

import pytest

from culler import xmp

LIGHTROOM = """<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="Adobe XMP Core 7.0">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:xmp="http://ns.adobe.com/xap/1.0/"
    xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/"
    xmlns:dc="http://purl.org/dc/elements/1.1/"
   xmp:Rating="2" xmp:Label="Blue" crs:Exposure2012="+0.35" crs:Temperature="3200">
   <dc:subject><rdf:Bag><rdf:li>concert</rdf:li><rdf:li>live</rdf:li></rdf:Bag></dc:subject>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>"""


def test_new_sidecar():
    text = xmp.build(None, 4, "Green")
    assert text.startswith("<?xpacket begin=") and text.rstrip().endswith('<?xpacket end="w"?>')
    assert 'xmp:Rating="4"' in text and 'xmp:Label="Green"' in text
    assert xmp.read(text) == {"rating": 4, "label": "Green"}


def test_merge_keeps_develop_settings_and_keywords():
    text = xmp.build(LIGHTROOM, 5, "Green", ("Green", "Red"))
    assert xmp.read(text) == {"rating": 5, "label": "Green"}
    assert 'crs:Exposure2012="+0.35"' in text and 'crs:Temperature="3200"' in text
    assert "<rdf:li>concert</rdf:li>" in text
    ET.fromstring(text[text.index("<x:xmpmeta"):text.index("<?xpacket end")])   # still well-formed


def test_no_opinion_leaves_foreign_values_alone():
    assert xmp.read(xmp.build(LIGHTROOM, None, None, ("Green", "Red"))) == {"rating": 2, "label": "Blue"}


def test_unpicking_removes_only_our_label():
    ours = xmp.build(LIGHTROOM, None, "Green", ("Green", "Red"))
    assert xmp.read(xmp.build(ours, None, None, ("Green", "Red"))) == {"rating": 2, "label": None}


def test_element_form_is_replaced():
    src = LIGHTROOM.replace(' xmp:Rating="2"', "").replace(
        "<dc:subject>", "<xmp:Rating>1</xmp:Rating><dc:subject>")
    assert xmp.read(src)["rating"] == 1
    out = xmp.build(src, 3, None)
    assert xmp.read(out)["rating"] == 3 and "<xmp:Rating>" not in out


def test_garbage_sidecar_is_refused_not_overwritten():
    with pytest.raises(xmp.XmpError):
        xmp.build("this is not xml <<<", 3, "Green")

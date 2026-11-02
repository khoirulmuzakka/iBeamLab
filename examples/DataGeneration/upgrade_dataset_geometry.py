"""Upgrade existing dataset manifests using their exact original SIMNRA files.

Spectrum shards are never opened or modified. Original manifests are retained as
``manifest.ibd.before-geometry``. The normal dataset reader remains strict.
"""
import argparse
from pathlib import Path
import re
import struct
import sys
import tempfile
import xml.etree.ElementTree as ET


def checksum(data):
    """Return the dataset envelope/SIMNRA provenance FNV-1a checksum."""
    value = 14695981039346656037
    for byte in data:
        value = ((value ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return value


class Cursor:
    """Read only the sample/setup prefix, retaining all other metadata bytes."""
    def __init__(self, data):
        self.data, self.offset = data, 0

    def take(self, size):
        end = self.offset + size
        if size < 0 or end > len(self.data):
            raise ValueError("truncated manifest")
        value = self.data[self.offset:end]
        self.offset = end
        return value

    def integer(self):
        return struct.unpack("<Q", self.take(8))[0]

    def text(self):
        return self.take(self.integer()).decode("utf-8")


def text(value):
    value = value.encode("utf-8")
    return struct.pack("<Q", len(value)) + value


def upgrade(raw, references):
    """Return an upgraded envelope, or None for an already current manifest."""
    if raw[:8] != b"IBDATA01" or len(raw) < 24:
        raise ValueError("not a dataset manifest")
    size, expected = struct.unpack("<QQ", raw[8:24])
    payload = raw[24:]
    if len(payload) != size or checksum(payload) != expected:
        raise ValueError("manifest checksum/length mismatch")
    cursor = Cursor(payload)
    for _ in range(cursor.integer()):
        cursor.take(32)  # thickness, roughness, porosity, pore diameter
        for _ in range(cursor.integer()):
            cursor.text()
            cursor.take(8)  # concentration
            cursor.take(24 * cursor.integer())  # isotope entries
    start = cursor.offset
    if payload[start:start + 8] == struct.pack("<Q", len("ibeamlab.setup.geometry")):
        if cursor.text() == "ibeamlab.setup.geometry" and cursor.integer() == 1:
            return None
        raise ValueError("unknown setup contract")
    count = cursor.integer()
    if not 0 < count <= 1024:
        raise ValueError("invalid measurement count")
    setup = bytearray(text("ibeamlab.setup.geometry") + struct.pack("<QQ", 1, count))
    fingerprints = []
    for _ in range(count):
        measurement_start = cursor.offset
        label = cursor.text()
        cursor.text()  # particle
        cursor.take(48)  # beam and detector scalar fields
        cursor.text()  # detector info
        if label not in references:
            raise ValueError(f"missing original SIMNRA reference for {label}")
        fingerprint, angles = references[label]
        fingerprints.append((label, fingerprint))
        setup.extend(payload[measurement_start:cursor.offset])
        setup.extend(struct.pack("<ddd", *angles))
        setup.extend(cursor.take(24))  # acquisition
    end = cursor.offset
    # Decode parameter descriptors only to locate the provenance string exactly.
    for _ in range(cursor.integer()):
        cursor.text()
        kind = cursor.integer()
        if kind == 0:
            cursor.integer()
        elif kind == 1:
            cursor.integer()
            cursor.text()
        elif 2 <= kind <= 8:
            cursor.text()
        else:
            raise ValueError("unknown original parameter target")
        cursor.text()
    provenance = cursor.text()
    for label, fingerprint in fingerprints:
        if not re.search(r"(?:^|;)" + re.escape(label) + ":" + str(fingerprint) + r"(?=;|\n|$)", provenance):
            raise ValueError(f"{label}: reference checksum does not match generation provenance")
    updated = payload[:start] + setup + payload[end:]
    return b"IBDATA01" + struct.pack("<QQ", len(updated), checksum(updated)) + updated


def main():
    """Validate all candidates with the native reader before replacing manifests."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Dataset root containing layers_*/manifest.ibd")
    parser.add_argument("--reference", action="append", required=True, metavar="LABEL=FILE")
    parser.add_argument("--extension-dir", type=Path)
    args = parser.parse_args()
    if args.extension_dir:
        sys.path.insert(0, str(args.extension_dir.resolve()))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from ibeamlab._engine import engine
    references = {}
    for item in args.reference:
        label, filename = item.split("=", 1)
        data = Path(filename).read_bytes()
        root = ET.fromstring(data)
        angles = []
        for name in ("incidenceangle", "exitangle", "scatteringangle"):
            matches = [e for e in root.iter() if e.tag.rsplit("}", 1)[-1] == name]
            if len(matches) != 1:
                raise ValueError(f"{filename}: expected exactly one {name}")
            angles.append(float(matches[0].text))
        if label in references:
            raise ValueError(f"duplicate reference: {label}")
        references[label] = checksum(data), angles
        print(f"{label}: incidence/exit/scattering = {angles} degrees")
    paths = sorted(args.dataset.glob("layers_*/manifest.ibd"))
    if not paths:
        raise ValueError("no completed layer datasets found")
    pending = []
    for path in paths:
        raw = path.read_bytes()
        updated = upgrade(raw, references)
        if updated is None:
            engine().DatasetReader(path.parent)
            print(f"Already current: {path.parent.name}")
            continue
        backup = path.with_name("manifest.ibd.before-geometry")
        if backup.exists() and backup.read_bytes() != raw:
            raise ValueError(f"conflicting backup: {backup}")
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "manifest.ibd").write_bytes(updated)
            engine().DatasetReader(Path(directory))
        pending.append((path, backup, raw, updated))
    for path, backup, raw, updated in pending:
        if path.read_bytes() != raw:
            raise ValueError(f"manifest changed during upgrade: {path}")
        if not backup.exists():
            with backup.open("xb") as stream:
                stream.write(raw)
        temporary = path.with_name("manifest.geometry.tmp")
        with temporary.open("xb") as stream:
            stream.write(updated)
        temporary.replace(path)
        metadata = engine().DatasetReader(path.parent).metadata
        print(f"Updated {path.parent.name}: {metadata.accepted:,} accepted records; backup retained")


if __name__ == "__main__":
    main()

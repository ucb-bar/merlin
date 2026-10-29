"""Authenticate new archive DATA, then bounded ZIP inspection/extraction only."""
import hashlib
import os
from pathlib import PurePosixPath
import stat
import struct
import zipfile
import zlib

import fdio
from policy import require


def authenticate(fd, policy):
    before = os.fstat(fd)
    require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1
            and before.st_size == policy["exact_bytes"], "parent archive FD/size")
    os.lseek(fd, 0, os.SEEK_SET)
    h, count = hashlib.sha256(), 0
    while block := os.read(fd, min(1024**2, policy["exact_bytes"] + 1 - count)):
        h.update(block)
        count += len(block)
        require(count <= policy["exact_bytes"], "parent archive grew")
    require(count == policy["exact_bytes"] and h.hexdigest() == policy["expected_sha256"],
            "parent archive authentication failed before ZIP processing")
    require(fdio.facts(before) == fdio.facts(os.fstat(fd)), "parent archive changed during authentication")
    return fdio.facts(before)


def central_bound(fd, limits):
    """Bound central-directory allocation before stdlib ZIP parsing.

    Whole-archive ZIP64/multidisk are explicitly unsupported in this candidate.
    No relaxation is implied if the uninspected official archive uses them.
    """
    size = os.fstat(fd).st_size
    tail = os.pread(fd, min(size, 65557), max(0, size - 65557))
    index = tail.rfind(b"PK\x05\x06")
    require(index >= 0 and len(tail) - index >= 22, "missing ZIP end record")
    _, disk, cd_disk, disk_count, count, cd_size, cd_offset, comment = struct.unpack_from("<4s4H2LH", tail, index)
    require(index + 22 + comment == len(tail), "trailing ZIP bytes/end record")
    require(disk == cd_disk == 0 and disk_count == count, "multidisk ZIP refused")
    require(count != 65535 and cd_size != 0xffffffff and cd_offset != 0xffffffff, "ZIP64 whole archive unsupported")
    require(count <= limits["member_count_max"] and cd_size <= 64 * 1024**2, "central directory cap")
    require(cd_offset + cd_size == size - len(tail) + index, "ZIP central directory layout")
    return count


def roster(infos, limits):
    require(0 < len(infos) <= limits["member_count_max"], "member count bound")
    rows, names, files, directories, total = [], set(), set(), set(), 0
    for info in infos:
        name = info.orig_filename
        require(type(name) is str and not any(ord(c) < 32 or ord(c) == 127 for c in name)
                and "\\" not in name and ":" not in name, "member path/control")
        clean = name[:-1] if name.endswith("/") else name
        p = PurePosixPath(clean)
        require(not p.is_absolute() and str(p) == clean and ".." not in p.parts
                and p.parts and p.parts[0] == limits["expected_top_directory"], "member escapes top root")
        require(clean not in names, "duplicate ZIP path")
        names.add(clean)
        require(info.flag_bits & (1 | 64) == 0, "encrypted member")
        require(info.compress_type in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED), "unsupported compression")
        mode = (info.external_attr >> 16) & 0xffff
        kind = stat.S_IFMT(mode)
        is_dir = name.endswith("/")
        require(info.create_system == 3 and kind == (stat.S_IFDIR if is_dir else stat.S_IFREG),
                "nonregular/link/unknown member type")
        # Only timestamp and UID/GID extras are admitted. Unix link-bearing and
        # ZIP64/other unknown extras fail closed instead of assuming authenticity.
        offset, seen = 0, set()
        while offset < len(info.extra):
            require(offset + 4 <= len(info.extra), "truncated ZIP extra")
            key, length = struct.unpack_from("<HH", info.extra, offset)
            require(key in (0x5455, 0x7875) and key not in seen and offset + 4 + length <= len(info.extra),
                    "unsupported/link-bearing/duplicate ZIP extra")
            seen.add(key)
            offset += 4 + length
        require(type(info.file_size) is int and 0 <= info.file_size <= limits["per_member_bytes_max"], "member size cap")
        if is_dir:
            require(info.file_size == 0 and info.CRC == 0, "directory with payload/CRC")
            directories.add(clean)
        else:
            files.add(clean)
            total += info.file_size
        for parent in p.parents:
            if str(parent) != ".":
                directories.add(str(parent))
        rows.append({"path": clean, "kind": "directory" if is_dir else "file", "declared_bytes": info.file_size,
                     "crc32": info.CRC, "source_mode": mode, "compression": info.compress_type})
    require(not files & directories, "file/directory prefix collision")
    require(total <= limits["total_uncompressed_bytes_max"], "total expanded size cap")
    return rows, directories


def extract(archive_fd, output_root, request, state):
    state.update(phase="parent-authentication", pending=b"")
    archive_stat = authenticate(archive_fd, request["fetch"])
    state["parent_archive_stat"] = archive_stat
    count = central_bound(archive_fd, request["extract"])
    os.lseek(archive_fd, 0, os.SEEK_SET)
    # dup shares the authenticated open file description; no path reopen.
    with os.fdopen(os.dup(archive_fd), "rb") as stream, zipfile.ZipFile(stream) as archive:
        infos = archive.infolist()
        require(len(infos) == count, "central directory count mismatch")
        rows, directories = roster(infos, request["extract"])
        fdio.save(output_root, "ZIP-ROSTER.json", {"members": rows, "parent_sha256": request["fetch"]["expected_sha256"]})
        os.mkdir("data", 0o700, dir_fd=output_root)
        tree = os.open("data", fdio.DIR, dir_fd=output_root)
        manifests, actual_total = [], 0
        try:
            for directory in sorted(directories, key=lambda p: (len(PurePosixPath(p).parts), p)):
                opened = fdio.child_dirs(tree, PurePosixPath(directory).parts)
                os.close(opened)
            for info, row in zip(infos, rows, strict=True):
                state.update(phase="member", member=row, pending=b"", pending_offset=0, member_received=0)
                if row["kind"] == "directory":
                    with archive.open(info) as member:
                        require(member.read(1) == b"", "directory payload")
                    manifests.append({**row, "measured_bytes": 0, "sha256": None})
                    continue
                p = PurePosixPath(row["path"])
                parent = fdio.child_dirs(tree, p.parts[:-1])
                target = None
                try:
                    target = fdio.new_file(parent, p.name)
                    h, size, crc = hashlib.sha256(), 0, 0
                    with archive.open(info) as member:
                        while True:
                            chunk = member.read(min(1024**2, row["declared_bytes"] + 1 - size))
                            state.update(pending=chunk, pending_offset=size, member_received=size + len(chunk))
                            if not chunk:
                                break
                            size += len(chunk)
                            actual_total += len(chunk)
                            require(size <= row["declared_bytes"] and actual_total <= request["extract"]["total_uncompressed_bytes_max"],
                                    "expanded byte bound")
                            h.update(chunk)
                            crc = zlib.crc32(chunk, crc)
                            fdio.write_all(target, chunk)
                            state["pending"] = b""
                    require(size == row["declared_bytes"], "member size mismatch")
                    require(crc == row["crc32"], "member CRC mismatch")
                    os.fsync(target)
                    st = fdio.stable_file(target, parent, p.name, size)
                    os.fsync(parent)
                    manifests.append({**row, "measured_bytes": size, "sha256": h.hexdigest(), "output_stat": st})
                except BaseException:
                    if target is not None:
                        state["partial_output_stat"] = fdio.facts(os.fstat(target))
                    raise
                finally:
                    if target is not None:
                        os.close(target)
                    os.close(parent)
            require(archive_stat == fdio.facts(os.fstat(archive_fd)), "parent changed during extraction")
            check_tree(tree, manifests, directories)
        finally:
            os.close(tree)
    parent_ref = {"url": request["fetch"]["url"], "sha256": request["fetch"]["expected_sha256"],
                  "bytes": request["fetch"]["exact_bytes"], "identity": request["environment_identity"]}
    return {"schema": "merlin.fresh-sdk-authenticated-member-manifest.v1", "parent_archive": parent_ref,
            "members": manifests, "expanded_bytes": actual_total, "new_environment_pins": True,
            "old_expectations_replaced": False, "authority": dict(request["authority"])}


def check_tree(tree, members, directories):
    """Exact metadata roster; any extra/missing/link reports concrete child sets."""
    files = {m["path"]: m for m in members if m["kind"] == "file"}
    children = {}
    for p in set(files) | directories:
        parent = str(PurePosixPath(p).parent)
        children.setdefault(parent, set()).add(PurePosixPath(p).name)

    def visit(fd, prefix):
        # A fresh open description avoids relying on an enumeration view created
        # before descendants existed. Bind it to the same directory identity.
        listing = os.open(".", fdio.DIR, dir_fd=fd)
        try:
            original, fresh = os.fstat(fd), os.fstat(listing)
            require((original.st_dev, original.st_ino, original.st_mode) ==
                    (fresh.st_dev, fresh.st_ino, fresh.st_mode), "enumeration directory identity")
            names = os.listdir(listing)
        finally:
            os.close(listing)
        expected = children.get(prefix or ".", set())
        require(set(names) == expected,
                f"output roster differs at {prefix!r}: actual={sorted(names)!r} expected={sorted(expected)!r}")
        for name in names:
            p = str(PurePosixPath(prefix) / name)
            if p in directories:
                child = os.open(name, fdio.DIR, dir_fd=fd)
                try:
                    visit(child, p)
                finally:
                    os.close(child)
            else:
                st = os.stat(name, dir_fd=fd, follow_symlinks=False)
                require(stat.S_ISREG(st.st_mode) and st.st_nlink == 1
                        and fdio.facts(st) == files[p]["output_stat"], "output member association changed")
    visit(tree, "")

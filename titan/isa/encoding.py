"""Binary encoding of TITAN ISA v1.0 (little-endian 64-bit words).

Header word
    [7:0] opcode  [15:8] func  [31:16] flags  [47:32] core
    [51:48] n_operands  [55:52] n_imm  [63:56] reserved (0)
Tensor operand (2 words)
    w0: [1:0] space  [5:2] dtype  [7:6] reserved  [47:8] address  [63:48] reserved
    w1: [19:0] rows  [39:20] cols  [63:40] stride (elements)
Immediate: 1 word (float immediates use the low 32 bits as IEEE binary32)

Program container: b"TITN" u16 major u16 minor u32 n_kernels u32 meta_len
    meta (JSON, zero-padded to 8 bytes) | per kernel: u32 id u32 n_words words | u32 n_words words
"""
from __future__ import annotations

import json
import struct

import numpy as np

from ..dtypes import DType
from .spec import ISA_VERSION, ErrorCode, Instruction, ISAError, Opcode, Program, Space, TensorRef

MAGIC = b"TITN"
_M = lambda bits: (1 << bits) - 1  # noqa: E731


def _field(v: int, bits: int, name: str) -> int:
    if not 0 <= v <= _M(bits):
        raise ISAError(ErrorCode.BAD_ENCODING, f"{name}={v} does not fit in {bits} bits")
    return v


def encode_instruction(ins: Instruction) -> list[int]:
    words = [_field(int(ins.opcode), 8, "opcode") | _field(ins.func, 8, "func") << 8 |
             _field(ins.flags, 16, "flags") << 16 | _field(ins.core, 16, "core") << 32 |
             _field(len(ins.operands), 4, "n_operands") << 48 | _field(len(ins.imm), 4, "n_imm") << 52]
    for r in ins.operands:
        words.append(int(r.space) | int(r.dtype) << 2 | _field(r.addr, 40, "addr") << 8)
        words.append(_field(r.rows, 20, "rows") | _field(r.cols, 20, "cols") << 20 |
                     _field(r.stride, 24, "stride") << 40)
    for i in ins.imm:
        words.append(_field(i, 64, "imm"))
    return words


def decode_instruction(words, pos: int = 0) -> tuple[Instruction, int]:
    if pos >= len(words):
        raise ISAError(ErrorCode.BAD_ENCODING, "truncated stream (header)")
    h = int(words[pos])
    if h >> 56:
        raise ISAError(ErrorCode.BAD_ENCODING, f"reserved header bits set at word {pos}")
    try:
        op = Opcode(h & 0xFF)
    except ValueError:
        raise ISAError(ErrorCode.ILLEGAL_OPCODE, f"opcode {h & 0xFF:#x} at word {pos}") from None
    func, flags, core = (h >> 8) & 0xFF, (h >> 16) & 0xFFFF, (h >> 32) & 0xFFFF
    n_ops, n_imm = (h >> 48) & 0xF, (h >> 52) & 0xF
    need = 1 + 2 * n_ops + n_imm
    if pos + need > len(words):
        raise ISAError(ErrorCode.BAD_ENCODING, f"truncated instruction at word {pos}")
    ops = []
    p = pos + 1
    for _ in range(n_ops):
        w0, w1 = int(words[p]), int(words[p + 1])
        p += 2
        if (w0 >> 6) & 0x3 or w0 >> 48:
            raise ISAError(ErrorCode.BAD_ENCODING, f"reserved operand bits set at word {p - 2}")
        try:
            sp, dt = Space(w0 & 0x3), DType((w0 >> 2) & 0xF)
        except ValueError:
            raise ISAError(ErrorCode.BAD_ENCODING, f"bad space/dtype code at word {p - 2}") from None
        ops.append(TensorRef(sp, (w0 >> 8) & _M(40), w1 & _M(20), (w1 >> 20) & _M(20), dt, (w1 >> 40) & _M(24)))
    imm = tuple(int(words[p + i]) for i in range(n_imm))
    return Instruction(op, core, tuple(ops), imm, func, flags), pos + need


def encode_stream(instrs) -> list[int]:
    out: list[int] = []
    for ins in instrs:
        out.extend(encode_instruction(ins))
    return out


def decode_stream(words) -> list[Instruction]:
    out, pos = [], 0
    while pos < len(words):
        ins, pos = decode_instruction(words, pos)
        out.append(ins)
    return out


def _words_bytes(words) -> bytes:
    return np.asarray(words, dtype=np.uint64).astype("<u8").tobytes()


def encode_program(prog: Program) -> bytes:
    meta = json.dumps(prog.meta, sort_keys=True, default=str).encode()
    meta += b"\0" * (-len(meta) % 8)
    out = [MAGIC, struct.pack("<HHII", *ISA_VERSION, len(prog.kernels), len(meta)), meta]
    for kid in sorted(prog.kernels):
        w = encode_stream(prog.kernels[kid])
        out += [struct.pack("<II", kid, len(w)), _words_bytes(w)]
    w = encode_stream(prog.instructions)
    out += [struct.pack("<I", len(w)), _words_bytes(w)]
    return b"".join(out)


def decode_program(data: bytes) -> Program:
    if data[:4] != MAGIC:
        raise ISAError(ErrorCode.BAD_ENCODING, "bad magic")
    try:
        major, minor, nk, ml = struct.unpack_from("<HHII", data, 4)
        if major != ISA_VERSION[0]:
            raise ISAError(ErrorCode.BAD_ENCODING, f"unsupported ISA major version {major}")
        pos = 16
        meta = json.loads(data[pos:pos + ml].rstrip(b"\0") or b"{}")
        pos += ml
        kernels = {}
        for _ in range(nk):
            kid, nw = struct.unpack_from("<II", data, pos)
            pos += 8
            kernels[kid] = decode_stream(np.frombuffer(data, "<u8", nw, pos).tolist())
            pos += 8 * nw
        (nw,) = struct.unpack_from("<I", data, pos)
        pos += 4
        main = decode_stream(np.frombuffer(data, "<u8", nw, pos).tolist())
        pos += 8 * nw
    except (struct.error, ValueError) as e:
        raise ISAError(ErrorCode.BAD_ENCODING, f"truncated or corrupt program: {e}") from None
    if pos != len(data):
        raise ISAError(ErrorCode.BAD_ENCODING, f"{len(data) - pos} trailing bytes")
    return Program(main, kernels, meta)

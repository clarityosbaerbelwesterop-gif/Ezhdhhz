import re
from pathlib import Path

import numpy as np
import pytest

from titan.dtypes import DType
from titan.isa import (SPEC, Act, ErrorCode, Flag, Instruction, ISAError, Norm, Opcode, Program, ReduceOp, Space,
                       TensorRef, assemble, decode_instruction, decode_program, disassemble, disassemble_program,
                       encode_instruction, encode_program, validate)
from titan.isa import build as B
from titan.isa.encoding import encode_stream

L = lambda a, r, c, d=DType.FP32, s=0: TensorRef(Space.LOCAL, a, r, c, d, s)  # noqa: E731
G = lambda a, r, c, d=DType.FP32, s=0: TensorRef(Space.GLOBAL, a, r, c, d, s)  # noqa: E731


def one_of_each() -> list[Instruction]:
    f, i8, i32 = DType.FP32, DType.INT8, DType.INT32
    x = L(0, 4, 8)
    y = L(512, 4, 8)
    return [
        B.nop(0),
        B.load(0, L(0, 4, 8), G(0x100, 4, 8, f, 16)),
        B.store(1, G(0x400, 4, 8), L(0, 4, 8), notify=True),
        B.copy(0, L(1024, 8, 4), L(0, 4, 8), transpose=True),
        B.matmul(0, L(0, 4, 4, i32), L(256, 4, 8, i8), L(512, 4, 8, i8), acc=True, tb=True),
        B.gemm(0, G(0, 16, 16), G(4096, 16, 8), G(8192, 8, 16), 0.5, 1.0),
        B.vec(Opcode.VECTOR_ADD, 0, x, x, L(2048, 1, 8), bcast_b="row"),
        B.vec(Opcode.VECTOR_MULTIPLY, 0, x, L(2048, 4, 1), y, bcast_a="col"),
        B.vec(Opcode.VECTOR_SUB, 0, x, x, L(2048, 1, 1), bcast_b="scalar"),
        B.vec(Opcode.VECTOR_MAX, 0, x, x, y),
        B.vec(Opcode.VECTOR_DIV, 0, x, x, y),
        B.scale_add(0, x, x, 2.0, -1.0, y),
        B.reduce(0, L(2048, 4, 1), x, ReduceOp.SUM, 1),
        B.accumulate(0, x, y),
        B.activation(0, y, x, Act.GELU),
        B.normalization(0, y, x, Norm.LAYERNORM, 1e-5),
        B.convert(0, L(4096, 4, 8, i8), x, 16.0),
        B.signal(0, 3, 2),
        B.wait(1, 3, 2),
        B.barrier(0, 0, 2),
        B.execute_kernel(0, 7, 4096),
    ]


def test_every_opcode_covered_and_roundtrips():
    instrs = one_of_each()
    assert {i.opcode for i in instrs} == set(Opcode)
    for ins in instrs:
        words = encode_instruction(ins)
        back, pos = decode_instruction(words)
        assert back == ins and pos == len(words)
        assert assemble(disassemble(ins)).instructions[0] == ins


def test_program_binary_roundtrip(tiny):
    prog = Program(one_of_each(), {7: [B.nop(0), B.load(0, L(0, 4, 8), G(0, 4, 8))]}, {"name": "t", "n": 3})
    blob = encode_program(prog)
    back = decode_program(blob)
    assert back.instructions == prog.instructions and back.kernels == prog.kernels and back.meta == prog.meta
    assert assemble(disassemble_program(prog)).instructions == prog.instructions
    with pytest.raises(ISAError):
        decode_program(blob + b"\0" * 8)
    with pytest.raises(ISAError):
        decode_program(b"XXXX" + blob[4:])
    with pytest.raises(ISAError):
        decode_program(blob[:-8])


def test_decoder_rejects_bad_words():
    with pytest.raises(ISAError) as e:
        decode_instruction([0xEE])
    assert e.value.code == ErrorCode.ILLEGAL_OPCODE
    with pytest.raises(ISAError) as e:
        decode_instruction([1 << 60])
    assert e.value.code == ErrorCode.BAD_ENCODING
    w = encode_stream([B.load(0, L(0, 4, 8), G(0, 4, 8))])
    with pytest.raises(ISAError):
        decode_instruction(w[:3])
    w[1] |= 1 << 50
    with pytest.raises(ISAError):
        decode_instruction(w)


BAD = [
    (B.matmul(0, L(0, 4, 4, DType.FP32), L(256, 4, 8), L(512, 7, 4)), ErrorCode.SHAPE_MISMATCH),
    (B.matmul(0, L(0, 4, 4, DType.FP32), L(256, 4, 8, DType.INT8), L(512, 8, 4, DType.INT8)), ErrorCode.DTYPE_MISMATCH),
    (B.matmul(0, L(0, 4, 4), G(256, 4, 8), L(512, 8, 4)), ErrorCode.ILLEGAL_SPACE),
    (B.load(0, L(2, 4, 4), G(0, 4, 4)), ErrorCode.MISALIGNED),
    (B.load(0, L(0, 4, 4), G(0, 4, 4, DType.FP16)), ErrorCode.DTYPE_MISMATCH),
    (B.load(0, L(64 * 1024 - 16, 4, 4), G(0, 4, 4)), ErrorCode.OUT_OF_BOUNDS),
    (B.store(0, L(0, 4, 4), G(0, 4, 4)), ErrorCode.ILLEGAL_SPACE),
    (Instruction(Opcode.ACTIVATION, 0, (L(0, 2, 2), L(64, 2, 2)), func=99), ErrorCode.BAD_FUNC),
    (Instruction(Opcode.LOAD, 0, (L(0, 2, 2), G(0, 2, 2)), flags=int(Flag.ACC)), ErrorCode.BAD_FLAGS),
    (B.signal(0, 99), ErrorCode.BAD_SEMAPHORE),
    (B.barrier(0, 1, 1), ErrorCode.BAD_CORE),
    (B.load(5, L(0, 2, 2), G(0, 2, 2)), ErrorCode.BAD_CORE),
    (B.gemm(0, G(0, 4, 4, DType.INT32), G(64, 4, 4, DType.INT8), G(128, 4, 4, DType.INT8), 2.0, 0.0),
     ErrorCode.BAD_IMMEDIATE),
    (B.vec(Opcode.VECTOR_ADD, 0, L(0, 4, 8), L(256, 4, 8), L(512, 1, 8)), ErrorCode.SHAPE_MISMATCH),
    (B.vec(Opcode.VECTOR_DIV, 0, L(0, 4, 8, DType.INT32), L(256, 4, 8, DType.INT32), L(512, 4, 8, DType.INT32)),
     ErrorCode.DTYPE_MISMATCH),
    (B.reduce(0, L(0, 1, 8), L(256, 4, 8), ReduceOp.SUM, 1), ErrorCode.SHAPE_MISMATCH),
    (B.execute_kernel(0, 42), ErrorCode.KERNEL_NOT_FOUND),
    (B.load(0, L(0, 3, 3, DType.INT4), G(0, 3, 3, DType.INT4)), ErrorCode.MISALIGNED),
]


@pytest.mark.parametrize("ins,code", BAD)
def test_validation_rules(ins, code, tiny):
    with pytest.raises(ISAError) as e:
        validate(ins, tiny, kernel_ids=set())
    assert e.value.code == code


def test_valid_instructions_pass(tiny):
    for ins in one_of_each():
        if ins.opcode != Opcode.EXECUTE_KERNEL:
            validate(ins, tiny)


def test_barrier_not_allowed_in_kernel(tiny):
    with pytest.raises(ISAError):
        Program([], {1: [B.barrier(0)]}).validate(tiny)


def test_isa_doc_in_sync():
    doc = (Path(__file__).resolve().parent.parent / "docs" / "ISA.md").read_text()
    for op in Opcode:
        assert re.search(rf"\b{op.name}\b", doc), f"{op.name} missing from docs/ISA.md"
        assert f"0x{op.value:02X}" in doc, f"opcode value of {op.name} missing from docs/ISA.md"
    for f in Flag:
        if f.name != "NONE":
            assert f.name in doc
    for code in ErrorCode:
        assert code.name in doc
    for op, sp in SPEC.items():
        if sp.func is not None:
            for v in sp.func:
                assert v.name in doc


def test_tensorref_sub_view():
    r = G(0, 8, 8, DType.FP32)
    s = r.sub(2, 4, 3, 2)
    assert s.addr == (2 * 8 + 4) * 4 and s.stride == 8 and s.shape == (3, 2)
    with pytest.raises(ValueError):
        r.sub(7, 0, 2, 2)
    with pytest.raises(ValueError):
        G(0, 2, 8, DType.INT4).sub(0, 1, 1, 2)
    assert np.isclose(s.nbytes, 24)

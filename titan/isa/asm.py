"""TITAN assembly text format (assembler + disassembler).

    ; comment
    .kernel <id>            ... .end      kernel definition (no BARRIER inside)
    <mnemonic>[.<func>] @<core> [+flag ...] [operand, ...] [# imm, ...]

operand := <G|S|L>:<addr>:<rows>x<cols>[/<stride>]:<dtype>     e.g. L:0x400:16x16/32:bf16
"""
from __future__ import annotations

from ..dtypes import DType
from .spec import SPEC, ErrorCode, Flag, Instruction, ISAError, Opcode, Program, Space, TensorRef, f2imm, imm2f

_SPACE = {"G": Space.GLOBAL, "S": Space.SHARED, "L": Space.LOCAL}
_FLAGS = {f.name.lower(): f for f in Flag if f.name not in ("NONE",)}


def parse_ref(tok: str) -> TensorRef:
    try:
        sp, addr, shape, dt = tok.strip().split(":")
        stride = 0
        if "/" in shape:
            shape, st = shape.split("/")
            stride = int(st, 0)
        r, c = shape.lower().split("x")
        return TensorRef(_SPACE[sp.upper()], int(addr, 0), int(r, 0), int(c, 0), DType.parse(dt), stride)
    except (ValueError, KeyError) as e:
        raise ISAError(ErrorCode.BAD_ENCODING, f"bad operand {tok!r}: {e}") from None


def disassemble(ins: Instruction) -> str:
    sp = SPEC[ins.opcode]
    s = sp.name.lower()
    if sp.func is not None:
        s += "." + sp.func(ins.func).name.lower()
    s += f" @{ins.core}"
    for name, f in _FLAGS.items():
        if ins.flags & f:
            s += f" +{name}"
    if ins.operands:
        s += " " + ", ".join(str(r) for r in ins.operands)
    if ins.imm:
        kinds = sp.imm_kind
        vals = [repr(imm2f(v)) if i < len(kinds) and kinds[i] == "f" else str(v) for i, v in enumerate(ins.imm)]
        s += " # " + ", ".join(vals)
    return s


def parse_line(line: str) -> Instruction:
    body, _, imm_txt = line.partition("#")
    toks = body.split(None, 1)
    if not toks:
        raise ISAError(ErrorCode.BAD_ENCODING, "empty line")
    mnem, rest = toks[0], (toks[1] if len(toks) > 1 else "")
    name, _, fname = mnem.partition(".")
    try:
        op = Opcode[name.upper()]
    except KeyError:
        raise ISAError(ErrorCode.ILLEGAL_OPCODE, f"unknown mnemonic {name!r}") from None
    sp = SPEC[op]
    func = 0
    if fname:
        if sp.func is None:
            raise ISAError(ErrorCode.BAD_FUNC, f"{name} takes no function suffix")
        try:
            func = int(sp.func[fname.upper()])
        except KeyError:
            raise ISAError(ErrorCode.BAD_FUNC, f"unknown function {fname!r} for {name}") from None
    core, flags, op_toks = 0, 0, []
    words = rest.replace(",", " , ").split()
    i = 0
    while i < len(words) and (words[i].startswith("@") or words[i].startswith("+")):
        w = words[i]
        if w.startswith("@"):
            core = int(w[1:], 0)
        else:
            try:
                flags |= int(_FLAGS[w[1:].lower()])
            except KeyError:
                raise ISAError(ErrorCode.BAD_FLAGS, f"unknown flag {w!r}") from None
        i += 1
    op_txt = " ".join(words[i:])
    if op_txt.strip():
        op_toks = [t for t in (x.strip() for x in op_txt.split(",")) if t]
    operands = tuple(parse_ref(t.replace(" ", "")) for t in op_toks)
    imm = []
    if imm_txt.strip():
        for j, t in enumerate(x.strip() for x in imm_txt.split(",")):
            kind = sp.imm_kind[j] if j < len(sp.imm_kind) else "i"
            imm.append(f2imm(float(t)) if kind == "f" else int(t, 0))
    return Instruction(op, core, operands, tuple(imm), func, flags)


def assemble(text: str) -> Program:
    prog = Program()
    cur: list | None = None
    for ln, raw in enumerate(text.splitlines(), 1):
        line = raw.split(";", 1)[0].strip()
        if not line:
            continue
        try:
            if line.startswith(".kernel"):
                if cur is not None:
                    raise ISAError(ErrorCode.BAD_ENCODING, "nested .kernel")
                kid = int(line.split()[1], 0)
                cur = prog.kernels.setdefault(kid, [])
            elif line == ".end":
                if cur is None:
                    raise ISAError(ErrorCode.BAD_ENCODING, ".end without .kernel")
                cur = None
            else:
                (cur if cur is not None else prog.instructions).append(parse_line(line))
        except ISAError as e:
            raise ISAError(e.code, f"line {ln}: {e}") from None
    if cur is not None:
        raise ISAError(ErrorCode.BAD_ENCODING, "unterminated .kernel")
    return prog


def disassemble_program(prog: Program) -> str:
    out = []
    for kid in sorted(prog.kernels):
        out.append(f".kernel {kid}")
        out += ["    " + disassemble(i) for i in prog.kernels[kid]]
        out.append(".end")
    out += [disassemble(i) for i in prog.instructions]
    return "\n".join(out) + "\n"

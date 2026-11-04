import numpy as np
import pytest

from titan.dtypes import (DType, bf16_bits_to_f32, convert, f32_to_bf16_bits, pack_int4, round_bf16, unpack_int4,
                          wrap_int32)


def test_bf16_round_to_nearest_even():
    x = np.array([1.0, 1 + 2**-8, 1 + 3 * 2**-8, -2.5, 65504.0, np.inf], dtype=np.float32)
    got = round_bf16(x)
    assert got[0] == 1.0
    assert got[1] == 1.0                      # exact tie -> even mantissa
    assert got[2] == 1 + 2**-6                # tie -> even (mantissa 2)
    assert got[3] == -2.5
    assert np.isinf(got[5])
    assert f32_to_bf16_bits(np.array([1.0], np.float32))[0] == 0x3F80
    assert f32_to_bf16_bits(np.array([np.nan], np.float32))[0] == 0x7FC0
    assert bf16_bits_to_f32(np.array([0x3F80], np.uint16))[0] == 1.0


def test_bf16_rounding_error_bound(rng):
    x = rng.standard_normal(10000).astype(np.float32)
    rel = np.abs(round_bf16(x) - x) / np.abs(x)
    assert rel.max() <= 2**-8 + 1e-9          # half ULP of an 8-bit significand


def test_int4_pack_roundtrip():
    v = np.array([np.arange(-8, 8)], dtype=np.int8)
    assert pack_int4(v).shape == (1, 8)
    np.testing.assert_array_equal(unpack_int4(pack_int4(v)), v)


def test_wrap_int32_two_complement():
    np.testing.assert_array_equal(wrap_int32(np.array([2**31, -2**31 - 1, 5])), [-2**31, 2**31 - 1, 5])


@pytest.mark.parametrize("x,dst,scale,want", [
    (300.7, DType.INT8, 1.0, 127), (-1000.0, DType.INT8, 1.0, -128), (2.5, DType.INT8, 1.0, 2),
    (3.5, DType.INT8, 1.0, 4), (np.nan, DType.INT8, 1.0, 0), (9.0, DType.INT4, 1.0, 7), (0.26, DType.INT8, 10.0, 3),
])
def test_convert_saturating_round_half_even(x, dst, scale, want):
    assert convert(np.array([[x]], np.float32), DType.FP32, dst, scale)[0, 0] == want


def test_convert_float_targets():
    x = np.array([[1 + 2**-9, 70000.0]], np.float32)
    assert convert(x, DType.FP32, DType.BF16)[0, 0] == 1.0
    assert np.isinf(convert(x, DType.FP32, DType.FP16)[0, 1])
    q = np.array([[-7, 100]], np.int8)
    np.testing.assert_allclose(convert(q, DType.INT8, DType.FP32, 0.5), [[-3.5, 50.0]])

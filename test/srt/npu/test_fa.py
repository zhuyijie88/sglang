import itertools
import unittest

import torch
import torch.nn.functional as F
import torch_npu
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

from sglang.test.test_utils import CustomTestCase

torch.manual_seed(1234)

precision = {
    torch.bfloat16: 1e-2,
    torch.float16: 1e-3,
    torch.float32: 1e-5,
}


class TestNpuFA(CustomTestCase):
    kv_lora_rank = 512
    q_nope_dim = 128
    q_rope_dim = 64
    k_nope_dim = 128
    k_rope_dim = 64
    v_head_dim = 128
    tp_q_head_num = [8, 16]
    bs_qlen = [4, 8, 16, 32]
    dtype = [torch.bfloat16]
    PAGE_SIZE = 128
    MAX_SEQLEN = 1024

    def _ifa_test(self, bs_qlen, n, dtype):
        scaling = 0.13523
        seq_lens_list = [9] * bs_qlen
        max_seqlen_pad = (self.MAX_SEQLEN + self.PAGE_SIZE - 1) // self.PAGE_SIZE
        block_kv_indices = torch.full(
            (bs_qlen, max_seqlen_pad),
            0,
            dtype=torch.int32,
        )
        for i in range(bs_qlen):
            block_kv_indices[i, 0] = max_seqlen_pad * i
        block_table = block_kv_indices.npu()
        q_nope = torch.randn([bs_qlen, 1, n, self.kv_lora_rank], dtype=dtype).npu()
        q_rope = torch.randn([bs_qlen, 1, n, self.q_rope_dim], dtype=dtype).npu()
        k_nope = torch.randn(
            [512, self.PAGE_SIZE, self.kv_lora_rank], dtype=dtype
        ).npu()
        k_rope = torch.randn([512, self.PAGE_SIZE, self.k_rope_dim], dtype=dtype).npu()

        out, _ = torch.ops.npu.npu_fused_infer_attention_score(
            q_nope,
            k_nope,
            k_nope,
            query_rope=q_rope,
            key_rope=k_rope,
            num_heads=n,
            num_key_value_heads=1,
            input_layout="BSND",
            atten_mask=None,
            sparse_mode=0,
            scale=scaling,
            antiquant_mode=0,
            antiquant_scale=None,
            block_table=block_table,
            block_size=self.PAGE_SIZE,
            actual_seq_lengths_kv=seq_lens_list,
        )
        out = out.cpu()

        q = torch.cat([q_nope, q_rope], dim=-1).transpose(1, 2)  # (B, N, S_q, D)
        k_nope_list = []
        k_rope_list = []
        for i in range(bs_qlen):
            k_nope_i = k_nope[block_kv_indices[i, 0]][: seq_lens_list[i]]
            k_nope_list.append(k_nope_i)
            k_rope_i = k_rope[block_kv_indices[i, 0]][: seq_lens_list[i]]
            k_rope_list.append(k_rope_i)
        k_nope_tensor = torch.stack(k_nope_list, dim=0).unsqueeze(1)
        k_rope_tensor = torch.stack(k_rope_list, dim=0).unsqueeze(1)
        k = torch.cat([k_nope_tensor, k_rope_tensor], dim=-1)  # (B, 1, S_kv, D)
        attn_weight = (
            torch.matmul(q, k.transpose(-1, -2)) * scaling
        )  # (B, N, S_q, S_kv)
        attn_weight = F.softmax(attn_weight, dim=-1, dtype=torch.float32).to(q.dtype)
        ref_out = torch.matmul(attn_weight, k_nope_tensor).transpose(1, 2).cpu()

        atol = rtol = precision[ref_out.dtype]
        diff = torch.isclose(ref_out, out, atol=atol, rtol=rtol)
        error_ratio = torch.sum(~diff) / diff.numel()
        assert error_ratio < 0.02

    def _pfa_test(self, bs_qlen, n, dtype):
        pa_atten_mask = ~torch.tril(
            torch.ones((2048, 2048), dtype=torch.bool, device="npu")
        )
        scaling = 0.13523
        seq_lens_cumsum_list = [2, bs_qlen]
        q_nope = torch.randn([bs_qlen, n, self.q_nope_dim], dtype=dtype).npu()
        q_rope = torch.randn([bs_qlen, n, self.q_rope_dim], dtype=dtype).npu()
        k_nope = torch.randn([bs_qlen, n, self.k_nope_dim], dtype=dtype).npu()
        k_rope = (
            torch.randn([bs_qlen, 1, self.k_rope_dim], dtype=dtype)
            .npu()
            .repeat(1, n, 1)
        )
        v = torch.randn([bs_qlen, n, self.v_head_dim], dtype=dtype).npu()

        out, _ = torch.ops.npu.npu_fused_infer_attention_score(
            q_nope,
            k_nope,
            v,
            query_rope=q_rope,
            key_rope=k_rope,
            num_heads=n,
            num_key_value_heads=n,
            input_layout="TND",
            atten_mask=pa_atten_mask,
            sparse_mode=3,
            scale=scaling,
            antiquant_mode=0,
            antiquant_scale=None,
            next_tokens=0,
            actual_seq_lengths=seq_lens_cumsum_list,
            actual_seq_lengths_kv=seq_lens_cumsum_list,
        )
        out = out.cpu()

        q = torch.cat([q_nope, q_rope], dim=-1)  # (B * S_q, N, D)
        k = torch.cat([k_nope, k_rope], dim=-1)  # (B * S_kv, N, D)
        offset = 0
        ref_out = []
        for i in range(len(seq_lens_cumsum_list)):
            q_i = q[offset : seq_lens_cumsum_list[i]].transpose(0, 1)
            k_i = k[offset : seq_lens_cumsum_list[i]].transpose(0, 1)
            v_i = v[offset : seq_lens_cumsum_list[i]].transpose(0, 1)  # (N, S, 128)
            atten_mask = (
                _prepare_4d_causal_attention_mask(
                    None,
                    (1, seq_lens_cumsum_list[i] - offset),
                    torch.arange(seq_lens_cumsum_list[i] - offset).float(),
                    0,
                    self.MAX_SEQLEN,
                )
                .npu()
                .squeeze(0)
            )

            attn_weight = (
                torch.matmul(q_i, k_i.transpose(-1, -2)) * scaling
            )  # (N, S_q, S_kv)
            attn_weight = attn_weight + atten_mask.squeeze(0)
            attn_weight = F.softmax(attn_weight, dim=-1, dtype=torch.float32).to(
                q.dtype
            )
            ref_out_i = torch.matmul(attn_weight, v_i).transpose(0, 1)
            ref_out.append(ref_out_i)
            offset = seq_lens_cumsum_list[i]
        ref_out = torch.cat(ref_out, dim=0).cpu()

        atol = rtol = precision[ref_out.dtype]
        diff = torch.isclose(ref_out, out, atol=atol, rtol=rtol)
        error_ratio = torch.sum(~diff) / diff.numel()
        assert error_ratio < 0.01

    def test_ifa_paged_attention(self):
        for params in itertools.product(self.bs_qlen, self.tp_q_head_num, self.dtype):
            with self.subTest(
                bs_qlen=params[0], tp_q_head_num=params[1], dtype=params[2]
            ):
                self._ifa_test(*params)

    def test_pfa(self):
        for params in itertools.product(self.bs_qlen, self.tp_q_head_num, self.dtype):
            with self.subTest(
                bs_qlen=params[0], tp_q_head_num=params[1], dtype=params[2]
            ):
                self._pfa_test(*params)


if __name__ == "__main__":
    unittest.main()

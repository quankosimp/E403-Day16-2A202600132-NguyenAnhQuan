# Báo Cáo Triển Khai AI Inference Service trên AWS

## 1. Thông tin chung

- **Học viên:** Nguyễn Anh Quân
- **Mã HV:** 2A202600132
- **Ngày thực hiện:** 2026-04-23
- **Region:** ap-southeast-2 (Sydney)
- **Instance Type:** g4dn.xlarge (NVIDIA T4 GPU, 16GB VRAM)
- **Model:** google/gemma-4-E2B-it (chạy qua vLLM container)
- **Endpoint ALB:** http://ai-inference-alb-ad5387ec-53138683.ap-southeast-2.elb.amazonaws.com

---

## 2. Cold Start Time (Thời gian triển khai)

**Mục tiêu:** < 15 phút cho instance T4

| Giai đoạn | Thời gian | Ghi chú |
|---|---|---|
| `terraform init` | ~15s | Tải provider AWS |
| `terraform plan` | ~10s | Lập kế hoạch resources |
| `terraform apply` (tạo VPC, Subnet, IGW, NAT, SG, IAM) | ~2 phút | Network stack |
| Tạo EC2 g4dn.xlarge + EBS gp3 | ~1 phút 30s | Provisioning instance |
| Tạo ALB + Target Group + Listener | ~2 phút 30s | Mất nhiều thời gian nhất ở Load Balancer |
| User-data: cập nhật OS, cài Docker, NVIDIA Container Toolkit | ~3 phút | apt-get update + install |
| Pull Docker image `vllm/vllm-openai:latest` | ~3 phút | Image dung lượng lớn (~10GB) |
| Tải model weights (gemma) từ Hugging Face | ~2 phút | Phụ thuộc dung lượng model |
| vLLM EngineCore khởi động + warm-up | ~30s | Load model lên GPU |
| Health check ALB → Healthy | ~1 phút | Đợi target qua healthy threshold |
| **TỔNG CỘNG (Cold Start)** | **~13 phút** | ✅ Đạt mục tiêu < 15 phút |

> Lưu ý: Lần triển khai đầu tiên thường lâu hơn do phải pull image. Các lần restart sau (warm) chỉ mất ~1-2 phút.

---

## 3. Lỗi gặp phải và cách xử lý

### 3.1. Lỗi từ phía Client (HTTP 500 InternalServerError)

```bash
curl -X POST http://ai-inference-alb-.../v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{ "model": "google/gemma-4-E2B-it", ... }'
```

**Response:**
```json
{
  "error": {
    "message": "EngineCore encountered an issue. See stack trace (above) for the root cause.",
    "type": "InternalServerError",
    "code": 500
  }
}
```

→ ALB vẫn route được request tới container, nhưng vLLM EngineCore crash khi xử lý → trả về 500.

### 3.2. Lỗi gốc từ vLLM EngineCore (Server-side)

Kiểm tra log container bằng `docker logs vllm --tail 30`, lỗi root cause là:

```
triton.runtime.errors.OutOfResources: out of resource: shared memory,
Required: 98304, Hardware limit: 65536.
Reducing block sizes or `num_stages` may help.
```

**Stack trace tóm tắt:**
- `vllm/v1/attention/backends/triton_attn.py` → `unified_attention()`
- `triton_unified_attention.py` → `kernel_unified_attention_2d`
- Triton kernel yêu cầu **98304 bytes (96KB)** shared memory
- GPU **NVIDIA T4** chỉ hỗ trợ **65536 bytes (64KB)** shared memory per SM
- → Kernel không launch được, EngineCore crash, APIServer shutdown

### 3.3. Phân tích nguyên nhân

| Yếu tố | Chi tiết |
|---|---|
| **Phần cứng T4** | Compute Capability 7.5 (Turing), shared memory tối đa 64KB/SM |
| **vLLM Triton Attention Kernel** | Mặc định block size lớn, được tối ưu cho A100/H100 (Ampere/Hopper) – các GPU này có 96KB+ shared memory |
| **Model gemma E2B** | Cần kernel attention với block size lớn → vượt quá giới hạn của T4 |

### 3.4. Giải pháp đề xuất

**Cách 1 — Force vLLM dùng backend khác (không phải Triton):**
```bash
docker run ... \
  -e VLLM_ATTENTION_BACKEND=XFORMERS \
  vllm/vllm-openai:latest \
  --model google/gemma-4-E2B-it
```
Hoặc dùng `FLASH_ATTN` (nếu hỗ trợ) thay vì `TRITON_ATTN_VLLM_V1`.

**Cách 2 — Giảm block size / num_stages của Triton:**
Set biến môi trường:
```bash
-e VLLM_TRITON_ATTN_BLOCK_SIZE=64
-e TRITON_NUM_STAGES=2
```

**Cách 3 — Dùng phiên bản vLLM cũ hơn tương thích T4:**
```bash
docker pull vllm/vllm-openai:v0.6.3   # thay vì latest
```

**Cách 4 — Đổi sang instance GPU mới hơn:**
- `g5.xlarge` (NVIDIA A10G, 24GB, shared memory 100KB+) → tương thích native với vLLM mới.
- `g6.xlarge` (NVIDIA L4) → cũng OK.

**Cách 5 (khuyến nghị nhanh nhất):** Thay model bằng phiên bản nhỏ hơn / quantized phù hợp T4:
```bash
--model TheBloke/gemma-2b-it-AWQ
```

---

## 4. Kết luận

- ✅ **Hạ tầng AWS triển khai thành công** trong **~13 phút**, đạt mục tiêu cold start < 15 phút.
- ✅ **ALB + EC2 + Security Group + IAM** hoạt động đúng (request đến được container).
- ❌ **vLLM container không serve được** trên GPU T4 do **Triton attention kernel vượt giới hạn shared memory (96KB > 64KB)**.
- 🔧 **Hướng khắc phục:** đổi attention backend sang `XFORMERS`, hoặc dùng vLLM phiên bản cũ tương thích T4, hoặc nâng cấp instance lên `g5.xlarge` (A10G).

---

## 5. Tài liệu tham khảo

- [vLLM Supported Hardware](https://docs.vllm.ai/en/latest/getting_started/installation.html)
- [NVIDIA T4 Compute Capability 7.5 specs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities)
- [vLLM Issue: Triton OOM on T4](https://github.com/vllm-project/vllm/issues)

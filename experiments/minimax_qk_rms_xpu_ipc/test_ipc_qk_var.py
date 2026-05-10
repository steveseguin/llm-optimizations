import os

import torch
import torch.distributed as dist

import minimax_qk_rms_xpu_ipc


def signed_ptr(ptr: int) -> int:
    return ptr - (1 << 64) if ptr >= (1 << 63) else ptr


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])

    torch.xpu.set_device(local_rank)
    dist.init_process_group("xccl")

    max_tokens = 8
    slots = 3
    mailbox = torch.full((slots, max_tokens, 2), -0.0, device="xpu", dtype=torch.float32)
    handle = minimax_qk_rms_xpu_ipc.get_ipc_handle(mailbox)
    handles = [None for _ in range(world)]
    dist.all_gather_object(handles, handle)

    peer_ptrs_host = []
    opened = []
    for peer_rank, peer_handle in enumerate(handles):
        if peer_rank == rank:
            ptr = mailbox.data_ptr()
        else:
            ptr = minimax_qk_rms_xpu_ipc.open_ipc_handle(peer_handle, local_rank)
            opened.append(ptr)
        peer_ptrs_host.append(ptr)

    ordered_ptrs = peer_ptrs_host
    peer_ptrs = torch.tensor(
        [signed_ptr(ptr) for ptr in ordered_ptrs], device="xpu", dtype=torch.int64
    )

    tokens = 4
    qk_var = torch.empty((tokens, 2), device="xpu", dtype=torch.float32)
    qk_var[:, 0] = float(rank + 1)
    qk_var[:, 1] = float((rank + 1) * 10)

    mailbox[0, :tokens, :].copy_(qk_var)
    torch.xpu.synchronize()
    dist.barrier()
    minimax_qk_rms_xpu_ipc.reduce_qk_var_from_mailboxes(
        qk_var, peer_ptrs, 0, max_tokens, world
    )
    torch.xpu.synchronize()

    expected_q = sum(float(r + 1) for r in range(world)) / world
    expected_k = sum(float((r + 1) * 10) for r in range(world)) / world
    expected = torch.empty_like(qk_var)
    expected[:, 0] = expected_q
    expected[:, 1] = expected_k

    ok = torch.allclose(qk_var.cpu(), expected.cpu(), atol=0, rtol=0)
    print(f"rank={rank} qk_var={qk_var.cpu().tolist()} ok={ok}", flush=True)
    assert ok

    for ptr in opened:
        minimax_qk_rms_xpu_ipc.close_ipc_handle(ptr, local_rank)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

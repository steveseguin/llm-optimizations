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

    tokens = int(os.environ.get("MINIMAX_QK_IPC_TOKENS", "4"))
    iters = int(os.environ.get("MINIMAX_QK_IPC_ITERS", "1"))
    timeout_iters = int(os.environ.get("MINIMAX_QK_IPC_TIMEOUT_ITERS", "500000000"))
    max_tokens = max(8, tokens)
    slots = int(os.environ.get("MINIMAX_QK_IPC_SLOTS", str(max(3, iters))))
    mailbox = torch.full((slots, max_tokens, 2), -0.0, device="xpu", dtype=torch.float32)
    seq_mailbox = torch.zeros((slots, max_tokens, 2), device="xpu", dtype=torch.int32)
    seq_counter = torch.zeros((slots,), device="xpu", dtype=torch.int32)
    handle = minimax_qk_rms_xpu_ipc.get_ipc_handle(mailbox)
    seq_handle = minimax_qk_rms_xpu_ipc.get_ipc_handle(seq_mailbox)
    handles = [None for _ in range(world)]
    seq_handles = [None for _ in range(world)]
    dist.all_gather_object(handles, handle)
    dist.all_gather_object(seq_handles, seq_handle)

    peer_ptrs_host = []
    seq_ptrs_host = []
    opened = []
    opened_seq = []
    for peer_rank, peer_handle in enumerate(handles):
        if peer_rank == rank:
            ptr = mailbox.data_ptr()
        else:
            ptr = minimax_qk_rms_xpu_ipc.open_ipc_handle(peer_handle, local_rank)
            opened.append(ptr)
        peer_ptrs_host.append(ptr)
    for peer_rank, peer_handle in enumerate(seq_handles):
        if peer_rank == rank:
            ptr = seq_mailbox.data_ptr()
        else:
            ptr = minimax_qk_rms_xpu_ipc.open_ipc_handle(peer_handle, local_rank)
            opened_seq.append(ptr)
        seq_ptrs_host.append(ptr)

    single_kernel = os.environ.get("MINIMAX_QK_IPC_SINGLE_KERNEL", "0") == "1"
    sequence_kernel = os.environ.get("MINIMAX_QK_IPC_SEQ", "1") == "1"
    counter_kernel = os.environ.get("MINIMAX_QK_IPC_COUNTER", "0") == "1"
    if single_kernel:
        # The single-kernel prototype writes to peer_ptrs[0] and polls all
        # entries, so put the local mailbox first. The sum is order-invariant.
        ordered_ptrs = [mailbox.data_ptr()] + [
            peer_ptrs_host[i] for i in range(world) if i != rank
        ]
        ordered_seq_ptrs = [seq_mailbox.data_ptr()] + [
            seq_ptrs_host[i] for i in range(world) if i != rank
        ]
    else:
        ordered_ptrs = peer_ptrs_host
        ordered_seq_ptrs = seq_ptrs_host
    peer_ptrs = torch.tensor(
        [signed_ptr(ptr) for ptr in ordered_ptrs], device="xpu", dtype=torch.int64
    )
    seq_ptrs = torch.tensor(
        [signed_ptr(ptr) for ptr in ordered_seq_ptrs], device="xpu", dtype=torch.int64
    )

    qk_var = torch.empty((tokens, 2), device="xpu", dtype=torch.float32)
    for iteration in range(iters):
        slot = iteration % slots
        q_base = float(rank + 1 + iteration)
        k_base = float((rank + 1 + iteration) * 10)
        qk_var[:, 0] = q_base
        qk_var[:, 1] = k_base

        if single_kernel and counter_kernel:
            dist.barrier()
            minimax_qk_rms_xpu_ipc.allreduce_qk_var_seq_counter(
                qk_var,
                peer_ptrs,
                seq_ptrs,
                seq_counter,
                slot,
                max_tokens,
                world,
                timeout_iters,
            )
        elif single_kernel and sequence_kernel:
            dist.barrier()
            minimax_qk_rms_xpu_ipc.allreduce_qk_var_seq(
                qk_var,
                peer_ptrs,
                seq_ptrs,
                slot,
                iteration + 1,
                max_tokens,
                world,
                timeout_iters,
            )
        elif single_kernel:
            mailbox[slot, :, :].fill_(-0.0)
            torch.xpu.synchronize()
            dist.barrier()
            minimax_qk_rms_xpu_ipc.allreduce_qk_var(
                qk_var, peer_ptrs, slot, max_tokens, world, timeout_iters
            )
        else:
            mailbox[slot, :tokens, :].copy_(qk_var)
            torch.xpu.synchronize()
            dist.barrier()
            minimax_qk_rms_xpu_ipc.reduce_qk_var_from_mailboxes(
                qk_var, peer_ptrs, slot, max_tokens, world
            )
        torch.xpu.synchronize()

        expected_q = sum(float(r + 1 + iteration) for r in range(world)) / world
        expected_k = (
            sum(float((r + 1 + iteration) * 10) for r in range(world)) / world
        )
        expected = torch.empty_like(qk_var)
        expected[:, 0] = expected_q
        expected[:, 1] = expected_k

        ok = torch.allclose(qk_var.cpu(), expected.cpu(), atol=0, rtol=0)
        if not ok or iteration == iters - 1:
            print(
                f"rank={rank} iter={iteration} qk_var={qk_var.cpu().tolist()} ok={ok}",
                flush=True,
            )
        assert ok

    for ptr in opened:
        minimax_qk_rms_xpu_ipc.close_ipc_handle(ptr, local_rank)
    for ptr in opened_seq:
        minimax_qk_rms_xpu_ipc.close_ipc_handle(ptr, local_rank)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

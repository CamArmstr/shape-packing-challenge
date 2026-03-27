#!/usr/bin/env python3
"""
GPU-parallel penalty SA for semicircle packing.
Runs N_CHAINS independent SA chains simultaneously on GPU.
Each chain: one random move per step, penalty = fast_MEC + lambda * overlap.
Periodically extracts promising chains and exact-scores them on CPU.
"""

import torch
import math
import json
import time
import sys
import numpy as np

sys.path.insert(0, '/home/camcore/.openclaw/workspace/shape-packing-challenge')
from src.semicircle_packing.scoring import validate_and_score
from src.semicircle_packing.geometry import Semicircle

device = torch.device('cuda')
N = 15          # semicircles
ARC = 16        # arc sample points per semicircle for fast MEC/overlap
BEST_FILE = '/home/camcore/.openclaw/workspace/shape-packing-challenge/best_solution.json'

# ─────────────────────────────────────────────
# GPU kernels (vectorized over chains)
# ─────────────────────────────────────────────

def build_arc_offsets():
    """Precompute arc point offsets (relative to center) for each semicircle orientation slot."""
    # Will be applied per-chain: actual pts = center + offsets rotated by theta
    # We sample ARC points from -pi/2 to pi/2 relative to theta, plus 2 flat endpoints
    t_offsets = torch.linspace(-math.pi/2, math.pi/2, ARC, device=device)
    cos_t = torch.cos(t_offsets)  # [ARC]
    sin_t = torch.sin(t_offsets)  # [ARC]
    return cos_t, sin_t  # relative offsets, apply as: x + cos(theta+t_offset)*R


def fast_mec_batch(xs, ys, ts, _cos_t=None, _sin_t=None):
    """
    Approximate MEC radius for a batch of packings.
    xs, ys, ts: [B, N] — batch of solutions
    Returns: [B] — approximate MEC radius for each
    """
    B = xs.shape[0]
    t_off = torch.linspace(-math.pi/2, math.pi/2, ARC, device=device)
    angles = ts.unsqueeze(-1) + t_off.view(1, 1, -1)  # [B, N, ARC]
    
    arc_x = xs.unsqueeze(-1) + torch.cos(angles)  # [B, N, ARC]
    arc_y = ys.unsqueeze(-1) + torch.sin(angles)  # [B, N, ARC]
    
    # Flat endpoints
    ep1_x = xs + torch.cos(ts + math.pi/2)  # [B, N]
    ep1_y = ys + torch.sin(ts + math.pi/2)
    ep2_x = xs + torch.cos(ts - math.pi/2)
    ep2_y = ys + torch.sin(ts - math.pi/2)
    
    # All boundary x,y: [B, N*ARC + N + N] = [B, N*(ARC+2)]
    all_x = torch.cat([arc_x.reshape(B, -1), ep1_x, ep2_x], dim=1)  # [B, N*(ARC+2)]
    all_y = torch.cat([arc_y.reshape(B, -1), ep1_y, ep2_y], dim=1)
    
    # Iterative minimax MEC
    cx = all_x.mean(dim=1, keepdim=True)  # [B, 1]
    cy = all_y.mean(dim=1, keepdim=True)
    
    for _ in range(25):
        dx = all_x - cx
        dy = all_y - cy
        dists = dx*dx + dy*dy
        far_idx = dists.argmax(dim=1)  # [B]
        far_x = all_x[torch.arange(B, device=device), far_idx].unsqueeze(1)
        far_y = all_y[torch.arange(B, device=device), far_idx].unsqueeze(1)
        cx = 0.7 * cx + 0.3 * far_x
        cy = 0.7 * cy + 0.3 * far_y
    
    dx = all_x - cx
    dy = all_y - cy
    mec_r = torch.sqrt((dx*dx + dy*dy).max(dim=1).values)
    return mec_r  # [B]


def overlap_penalty_batch(xs, ys, ts):
    """
    Approximate pairwise overlap penalty for a batch.
    Uses distance between arc midpoints as proxy: penalize pairs where centers < 2.
    xs, ys, ts: [B, N]
    Returns: [B] overlap penalty
    """
    B = xs.shape[0]
    # All pairwise center distances
    xi = xs.unsqueeze(2)  # [B, N, 1]
    xj = xs.unsqueeze(1)  # [B, 1, N]
    yi = ys.unsqueeze(2)
    yj = ys.unsqueeze(1)
    
    dx = xi - xj  # [B, N, N]
    dy = yi - yj
    dist2 = dx*dx + dy*dy  # [B, N, N]
    
    # Soft penalty: max(0, 2 - dist)^2 for pairs where centers < 2
    # (two unit semicircles CAN'T overlap if centers > 2)
    dist = torch.sqrt(dist2.clamp(min=1e-8))
    pen = torch.clamp(2.0 - dist, min=0.0) ** 2
    
    # Zero out diagonal
    mask = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0)
    pen = pen.masked_fill(mask, 0.0)
    
    # Sum upper triangle only
    penalty = pen.sum(dim=[1, 2]) / 2.0  # [B]
    return penalty


def objective_batch(xs, ys, ts, lam):
    """Total objective: fast_mec + lam * overlap_penalty."""
    mec = fast_mec_batch(xs, ys, ts, None, None)
    pen = overlap_penalty_batch(xs, ys, ts)
    return mec + lam * pen


def run_gpu_sa(init_raw, n_chains=256, n_steps=500000, 
               T_start=1.5, T_end=0.002,
               lam_start=10.0, lam_end=3000.0,
               label="gpu"):
    """
    Run n_chains independent SA chains on GPU simultaneously.
    Returns list of (score, raw_solution) for all chains that reached feasibility.
    """
    # Initialize all chains from init_raw + small noise
    xs0 = torch.tensor([d['x'] for d in init_raw], device=device, dtype=torch.float32)
    ys0 = torch.tensor([d['y'] for d in init_raw], device=device, dtype=torch.float32)
    ts0 = torch.tensor([d['theta'] for d in init_raw], device=device, dtype=torch.float32)
    
    # [n_chains, N]
    noise_scale = 0.3
    xs = xs0.unsqueeze(0).expand(n_chains, -1).clone() + torch.randn(n_chains, N, device=device) * noise_scale
    ys = ys0.unsqueeze(0).expand(n_chains, -1).clone() + torch.randn(n_chains, N, device=device) * noise_scale
    ts = ts0.unsqueeze(0).expand(n_chains, -1).clone() + torch.randn(n_chains, N, device=device) * 0.5
    
    lam = lam_start
    T = T_start
    alpha_T = (T_end / T_start) ** (1.0 / n_steps)
    alpha_lam = (lam_end / lam_start) ** (1.0 / n_steps)
    
    step_xy = 0.15
    step_t = 0.3
    
    current_obj = objective_batch(xs, ys, ts, lam)
    best_obj = current_obj.clone()
    best_xs = xs.clone()
    best_ys = ys.clone()
    best_ts = ts.clone()
    
    t0 = time.time()
    log_every = 50000
    
    for step in range(n_steps):
        T *= alpha_T
        lam *= alpha_lam
        
        # Pick random semicircle per chain
        idx = torch.randint(0, N, (n_chains,), device=device)
        
        # Propose moves
        dx = torch.randn(n_chains, device=device) * step_xy
        dy = torch.randn(n_chains, device=device) * step_xy
        dt = torch.randn(n_chains, device=device) * step_t
        
        move_type = torch.rand(n_chains, device=device)
        # 0.35: xy only, 0.35: t only, 0.30: both
        apply_xy = (move_type > 0.35).float()
        apply_t = (move_type < 0.70).float()
        
        # Apply moves to selected index
        new_xs = xs.clone()
        new_ys = ys.clone()
        new_ts = ts.clone()
        
        chain_idx = torch.arange(n_chains, device=device)
        new_xs[chain_idx, idx] += dx * apply_xy
        new_ys[chain_idx, idx] += dy * apply_xy
        new_ts[chain_idx, idx] += dt * apply_t
        
        # Compute new objective
        new_obj = objective_batch(new_xs, new_ys, new_ts, lam)
        
        # SA acceptance
        delta = new_obj - current_obj
        accept_prob = torch.exp(-delta / T).clamp(max=1.0)
        accept = (torch.rand(n_chains, device=device) < accept_prob)
        
        # Update accepted chains
        xs = torch.where(accept.unsqueeze(1), new_xs, xs)
        ys = torch.where(accept.unsqueeze(1), new_ys, ys)
        ts = torch.where(accept.unsqueeze(1), new_ts, ts)
        current_obj = torch.where(accept, new_obj, current_obj)
        
        # Track per-chain bests
        improved = current_obj < best_obj
        best_xs = torch.where(improved.unsqueeze(1), xs, best_xs)
        best_ys = torch.where(improved.unsqueeze(1), ys, best_ys)
        best_ts = torch.where(improved.unsqueeze(1), ts, best_ts)
        best_obj = torch.minimum(best_obj, current_obj)
        
        if step % log_every == 0 and step > 0:
            elapsed = time.time() - t0
            best_so_far = best_obj.min().item()
            median_obj = current_obj.median().item()
            print(f"  [{label}] step={step}/{n_steps} T={T:.4f} lam={lam:.0f} "
                  f"best={best_so_far:.4f} med={median_obj:.4f} ({step/elapsed:.0f}/s)", flush=True)
    
    elapsed = time.time() - t0
    print(f"  [{label}] Done {elapsed:.1f}s | {n_steps/elapsed:.0f} steps/sec", flush=True)
    
    # Return all chains' best solutions
    results = []
    bx_np = best_xs.cpu().numpy()
    by_np = best_ys.cpu().numpy()
    bt_np = best_ts.cpu().numpy()
    obj_np = best_obj.cpu().numpy()
    
    # Sort by objective, return top chains
    order = np.argsort(obj_np)
    for i in order[:min(20, n_chains)]:
        raw = [{'x': float(bx_np[i,j]), 'y': float(by_np[i,j]), 'theta': float(bt_np[i,j])} for j in range(N)]
        results.append((float(obj_np[i]), raw))
    
    return results


def center_solution(raw):
    sol = [Semicircle(d['x'], d['y'], d['theta']) for d in raw]
    result = validate_and_score(sol)
    if not result.valid:
        return raw, None
    cx, cy, r = result.mec
    centered = [{'x': round(d['x']-cx, 6), 'y': round(d['y']-cy, 6), 'theta': round(d['theta'], 6)} for d in raw]
    return centered, r


def load_best():
    with open(BEST_FILE) as f:
        return json.load(f)


def save_best(raw):
    with open(BEST_FILE, 'w') as f:
        json.dump(raw, f, indent=2)


if __name__ == '__main__':
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    
    best_raw = load_best()
    best_centered, r_init = center_solution(best_raw)
    overall_best = r_init
    save_best(best_centered)
    print(f"Starting best: {overall_best:.6f}", flush=True)
    
    run_num = 0
    t_total = time.time()
    max_runtime = 3600
    
    while time.time() - t_total < max_runtime:
        run_num += 1
        print(f"\n{'='*60}\nRun {run_num}\n{'='*60}", flush=True)
        
        init_raw = load_best()
        
        try:
            candidates = run_gpu_sa(
                init_raw,
                n_chains=256,
                n_steps=300000,
                T_start=1.5, T_end=0.002,
                lam_start=8.0, lam_end=2000.0,
                label=f"run{run_num}"
            )
            
            print(f"  Exact-scoring top {len(candidates)} candidates...", flush=True)
            improved_count = 0
            for approx_score, raw in candidates:
                sol = [Semicircle(d['x'], d['y'], d['theta']) for d in raw]
                result = validate_and_score(sol)
                if result.valid and result.score < overall_best:
                    centered, _ = center_solution(raw)
                    overall_best = result.score
                    save_best(centered)
                    improved_count += 1
                    print(f"  *** NEW BEST: {overall_best:.6f} (approx was {approx_score:.4f}) ***", flush=True)
                    # Save viz
                    sol_v = [Semicircle(d['x'], d['y'], d['theta']) for d in centered]
                    r_v = validate_and_score(sol_v)
                    from src.semicircle_packing.visualization import plot_packing
                    plot_packing(sol_v, r_v.mec,
                                 save_path='/home/camcore/.openclaw/workspace/shape-packing-challenge/best_solution.png')
            
            if improved_count == 0:
                # Show best valid score from this run
                best_valid = [(s, r) for s, r in candidates 
                              if validate_and_score([Semicircle(d['x'], d['y'], d['theta']) for d in r]).valid]
                if best_valid:
                    print(f"  No improvement. Best valid this run: checking...", flush=True)
                    s, r = best_valid[0]
                    res = validate_and_score([Semicircle(d['x'], d['y'], d['theta']) for d in r])
                    print(f"  Best valid: {res.score:.6f} (global: {overall_best:.6f})", flush=True)
                else:
                    print(f"  No valid solutions found this run", flush=True)
        
        except Exception as e:
            import traceback
            traceback.print_exc()
    
    print(f"\nFinal best: {overall_best:.6f}", flush=True)

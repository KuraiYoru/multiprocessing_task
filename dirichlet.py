import multiprocessing as mp
import sys
import time
from array import array
from multiprocessing import shared_memory


def make_problem(N):
    h = 1.0 / (N + 1)
    t = [k * h for k in range(N + 2)]
    u = [[0.0] * (N + 2) for _ in range(N + 2)]
    for k in range(N + 2):
        u[0][k] = 100 - 200 * t[k]          # x = 0
        u[N + 1][k] = -100 + 200 * t[k]     # x = 1
        u[k][0] = 100 - 200 * t[k]          # y = 0
        u[k][N + 1] = -100 + 200 * t[k]     # y = 1
    f = [[0.0] * (N + 2) for _ in range(N + 2)]
    return u, f, h


def exact_solution(N):
    h = 1.0 / (N + 1)
    return [[100 * (1 - 2 * i * h) * (1 - 2 * j * h) for j in range(N + 2)]
            for i in range(N + 2)]


def max_diff(a, b):
    return max(abs(x - y) for ra, rb in zip(a, b) for x, y in zip(ra, rb))


def solve_poisson_sequential(u, f, N, h, eps, max_iter=100_000):
    h2 = h * h
    it = 0
    while True:
        dmax = 0.0
        for i in range(1, N + 1):
            up, cur, down, frow = u[i - 1], u[i], u[i + 1], f[i]
            for j in range(1, N + 1):
                temp = cur[j]
                cur[j] = 0.25 * (up[j] + down[j] + cur[j - 1] + cur[j + 1] - h2 * frow[j])
                dm = abs(temp - cur[j])
                if dmax < dm:
                    dmax = dm
        it += 1
        if not (dmax > eps and it < max_iter):
            return it


def _attach(shm_name):
    shm = shared_memory.SharedMemory(name=shm_name)
    return shm, shm.buf.cast("d")


def _static_range(lo, hi, rank, nprocs):
    n = hi - lo + 1
    base, extra = divmod(n, nprocs)
    start = lo + rank * base + min(rank, extra)
    return range(start, start + base + (1 if rank < extra else 0))


def _merge_dmax(local_dmax, dmax, dmax_lock):
    with dmax_lock:
        if dmax.value < local_dmax:
            dmax.value = local_dmax


def _decide(rank, steps, eps, max_iter, sync):
    dmax, _, barrier, stop, iters, _ = sync
    if rank == 0:
        iters.value += steps
        if dmax.value > eps and iters.value < max_iter:
            dmax.value = 0.0
        else:
            stop.value = 1
    barrier.wait()
    return stop.value == 1


def _relax_block(u, W, f, h2, N, B, bi, bj, track):
    i0, i1 = 1 + bi * B, min(N, (bi + 1) * B)
    j0, j1 = 1 + bj * B, min(N, (bj + 1) * B)
    a, b, n = j0 - 1, j1 + 2, j1 - j0 + 1

    up = u[(i0 - 1) * W + a:(i0 - 1) * W + b].tolist()
    cur = u[i0 * W + a:i0 * W + b].tolist()
    dm = 0.0
    for i in range(i0, i1 + 1):
        down = u[(i + 1) * W + a:(i + 1) * W + b].tolist()
        frow = f[i]
        for k in range(1, n + 1):
            temp = cur[k]
            new = 0.25 * (up[k] + down[k] + cur[k - 1] + cur[k + 1] - h2 * frow[a + k])
            cur[k] = new
            if track:
                d = abs(temp - new)
                if dm < d:
                    dm = d
        u[i * W + j0:i * W + j1 + 1] = array("d", cur[1:n + 1])
        up, cur = cur, down
    return dm


def _worker_wavefront(rank, nprocs, shm_name, N, f, h, eps, max_iter, sync, row_max):
    dmax, _, barrier, _, _, elapsed = sync
    shm, u = _attach(shm_name)
    W = N + 2
    h2 = h * h

    barrier.wait()
    t0 = time.perf_counter()
    while True:
        for nx in range(1, 2 * N):
            i_start = max(1, nx - N + 1)
            i_end = min(nx, N)
            for i in _static_range(i_start, i_end, rank, nprocs):
                j = nx + 1 - i
                k = i * W + j
                temp = u[k]
                new = 0.25 * (u[k - W] + u[k + W] + u[k - 1] + u[k + 1] - h2 * f[i][j])
                u[k] = new
                d = abs(temp - new)
                if row_max[i] < d:
                    row_max[i] = d
            barrier.wait()

        if rank == 0:
            m = 0.0
            for i in range(1, N + 1):
                if m < row_max[i]:
                    m = row_max[i]
                row_max[i] = 0.0
            dmax.value = m
        if _decide(rank, 1, eps, max_iter, sync):
            break

    if rank == 0:
        elapsed.value = time.perf_counter() - t0
    del u
    shm.close()


def _worker_blocks(rank, nprocs, shm_name, N, f, h, eps, max_iter, sync, B):
    dmax, dmax_lock, barrier, _, _, elapsed = sync
    shm, u = _attach(shm_name)
    W = N + 2
    h2 = h * h
    nb = (N + B - 1) // B
    last = 2 * nb - 2

    barrier.wait()
    t0 = time.perf_counter()
    while True:
        local_dmax = 0.0
        for nx in range(last + 1):
            bi_start = max(0, nx - nb + 1)
            bi_end = min(nx, nb - 1)
            for bi in range(bi_start + rank, bi_end + 1, nprocs):
                d = _relax_block(u, W, f, h2, N, B, bi, nx - bi, True)
                if local_dmax < d:
                    local_dmax = d
            if nx < last:
                barrier.wait()

        _merge_dmax(local_dmax, dmax, dmax_lock)
        barrier.wait()
        if _decide(rank, 1, eps, max_iter, sync):
            break

    if rank == 0:
        elapsed.value = time.perf_counter() - t0
    del u
    shm.close()


def _cache_wave_tasks(w, steps, nb):
    tasks = []
    for t in range(steps):
        for bi in range(nb):
            bj = w - 2 * t - bi
            if 0 <= bj < nb:
                tasks.append((t, bi, bj))
    return tasks


def _worker_cache(rank, nprocs, shm_name, N, f, h, eps, max_iter, sync, B, T):
    dmax, dmax_lock, barrier, _, iters, elapsed = sync
    shm, u = _attach(shm_name)
    W = N + 2
    h2 = h * h
    nb = (N + B - 1) // B

    barrier.wait()
    t0 = time.perf_counter()
    while True:
        steps = min(T, max_iter - iters.value)
        max_wave = 2 * (nb - 1) + 2 * (steps - 1)
        local_dmax = 0.0
        for w in range(max_wave + 1):
            tasks = _cache_wave_tasks(w, steps, nb)
            for idx in range(rank, len(tasks), nprocs):
                t, bi, bj = tasks[idx]
                d = _relax_block(u, W, f, h2, N, B, bi, bj, t == steps - 1)
                if local_dmax < d:
                    local_dmax = d
            if w < max_wave:
                barrier.wait()

        _merge_dmax(local_dmax, dmax, dmax_lock)
        barrier.wait()
        if _decide(rank, steps, eps, max_iter, sync):
            break

    if rank == 0:
        elapsed.value = time.perf_counter() - t0
    del u
    shm.close()


def _run_parallel(target, u, f, N, h, eps, max_iter, num_procs, extra=()):
    W = N + 2
    shm = shared_memory.SharedMemory(create=True, size=8 * W * W)
    try:
        arr = shm.buf.cast("d")
        for i in range(W):
            for j in range(W):
                arr[i * W + j] = u[i][j]

        sync = (mp.Value("d", 0.0, lock=False),   # dmax
                mp.Lock(),                         # замок для dmax
                mp.Barrier(num_procs),             # барьер
                mp.Value("i", 0, lock=False),      # stop
                mp.Value("i", 0, lock=False),      # iters
                mp.Value("d", 0.0, lock=False))    # elapsed

        procs = [mp.Process(target=target,
                            args=(r, num_procs, shm.name, N, f, h, eps, max_iter, sync) + tuple(extra))
                 for r in range(num_procs)]
        for p in procs:
            p.start()
        for p in procs:
            p.join()

        for i in range(W):
            for j in range(W):
                u[i][j] = arr[i * W + j]
        del arr
    finally:
        shm.close()
        shm.unlink()
    return sync[4].value, sync[5].value


def solve_poisson_wavefront(u, f, N, h, eps, max_iter=100_000, num_procs=4):
    row_max = mp.RawArray("d", N + 1)
    return _run_parallel(_worker_wavefront, u, f, N, h, eps, max_iter, num_procs,
                         (row_max,))


def solve_poisson_blocks(u, f, N, h, eps, max_iter=100_000, num_procs=4, block_size=16):
    return _run_parallel(_worker_blocks, u, f, N, h, eps, max_iter, num_procs,
                         (block_size,))


def solve_poisson_cache(u, f, N, h, eps, max_iter=100_000, num_procs=4,
                        block_size=32, time_steps=4):
    return _run_parallel(_worker_cache, u, f, N, h, eps, max_iter, num_procs,
                         (block_size, time_steps))


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    p = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    eps = 0.1

    print(f"N = {N}, eps = {eps}, процессов = {p}, ядер = {mp.cpu_count()}")
    exact = exact_solution(N)

    u_seq, f, h = make_problem(N)
    t0 = time.perf_counter()
    k_seq = solve_poisson_sequential(u_seq, f, N, h, eps)
    t_seq = time.perf_counter() - t0

    print(f"{'алгоритм':<12}{'итераций':>10}{'время, c':>12}{'ускорение':>12}"
          f"{'ошибка':>14}{'отличие от посл.':>18}")
    print(f"{'sequential':<12}{k_seq:>10}{t_seq:>12.3f}{1.0:>12.2f}"
          f"{max_diff(u_seq, exact):>14.6f}{0.0:>18.2e}")

    solvers = [("wavefront", solve_poisson_wavefront),
               ("blocks", solve_poisson_blocks),
               ("cache", solve_poisson_cache)]
    for name, solver in solvers:
        u, f, h = make_problem(N)
        k, t = solver(u, f, N, h, eps, num_procs=p)

        ref = u_seq
        if k > k_seq:
            ref = [row[:] for row in u_seq]
            solve_poisson_sequential(ref, f, N, h, -1.0, max_iter=k - k_seq)

        print(f"{name:<12}{k:>10}{t:>12.3f}{t_seq / t:>12.2f}"
              f"{max_diff(u, exact):>14.6f}{max_diff(u, ref):>18.2e}")
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

def worker(task_id):
    pos = mp.current_process()._identity[0]  # worker number 1..N
    n_items = 100
    with tqdm(total=n_items, desc=f"task {task_id}", position=pos, leave=False) as bar:
        for _ in range(n_items):
            time.sleep(0.02)
            bar.update(1)
    return task_id

if __name__ == "__main__":
    N = 4
    lock = mp.RLock()
    with ProcessPoolExecutor(
        max_workers=N,
        initializer=tqdm.set_lock,
        initargs=(lock,),
    ) as ex:
        results = list(ex.map(worker, range(8)))
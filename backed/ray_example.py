import ray 
import time 

ray.init()


def normal_sleep(seconds):
    time.sleep(seconds)
    return f"Slept {seconds}s"

@ray.remote
def remote_sleep(seconds):
    time.sleep(seconds)
    return f"Slept {seconds}s"


print("Remote function")
start = time.time()
tasks = []

for i in range(5):
    tasks.append(remote_sleep.remote(seconds = 2))

results = ray.get(tasks)
for i in range(5):
    print(results[i])

print(f"Time1: {time.time() - start:.2f}s")  # ~2秒（并行执行！）


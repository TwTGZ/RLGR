import torch
import torch.multiprocessing as mp

def burn_gpu(gpu_index, size):
    """在指定GPU上运行空转程序"""
    device = torch.device(f"cuda:{gpu_index}")
    print(f"[GPU {gpu_index}] Starting burn process...")
    
    try:
        # 创建大矩阵并开始计算
        mat = torch.randn(size, size, device=device)
        print(f"[GPU {gpu_index}] Matrix created, size: {size}x{size}")
        print(f"[GPU {gpu_index}] Running infinite loop...")
        
        iteration = 0
        while True:
            mat = mat @ mat
            torch.cuda.synchronize()
            iteration += 1
            if iteration % 10 == 0:
                print(f"[GPU {gpu_index}] Iteration {iteration} completed")
    except Exception as e:
        print(f"[GPU {gpu_index}] Error: {e}")

if __name__ == "__main__":
    # ==================== 配置区域 ====================
    # 指定要空转的GPU索引，例如：[0], [1], [0, 1], [1, 2, 3]
    TARGET_GPUS = [0, 1]  # 修改这里来指定要空转的GPU
    
    # 矩阵大小，数值越大占用显存和计算资源越多
    MATRIX_SIZE = 40000  # 可以根据需要调整：20000, 40000, 60000等
    # =================================================
    
    mp.set_start_method("spawn", force=True)
    
    if not torch.cuda.is_available():
        print("❌ No GPU available. Exiting.")
        exit(1)
    
    total_gpus = torch.cuda.device_count()
    print(f"✓ Total GPUs detected: {total_gpus}")
    
    # 验证指定的GPU索引是否有效
    invalid_gpus = [g for g in TARGET_GPUS if g >= total_gpus]
    if invalid_gpus:
        print(f"❌ Invalid GPU indices: {invalid_gpus}")
        print(f"   Available GPU indices: 0-{total_gpus-1}")
        exit(1)
    
    print(f"✓ Target GPUs: {TARGET_GPUS}")
    print(f"✓ Matrix size: {MATRIX_SIZE}x{MATRIX_SIZE}")
    print("-" * 50)
    
    processes = []
    for gpu_id in TARGET_GPUS:
        p = mp.Process(target=burn_gpu, args=(gpu_id, MATRIX_SIZE))
        p.start()
        processes.append(p)
    
    print(f"\n✓ Started {len(processes)} burn process(es)")
    print("Press Ctrl+C to stop all processes...\n")
    
    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("\n\n⚠ Stopping all processes...")
        for p in processes:
            p.terminate()
        for p in processes:
            p.join()
        print("✓ All processes stopped successfully.")
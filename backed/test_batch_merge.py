import torch
from torch.utils.data import DataLoader, TensorDataset

def simple_test():
    """简单测试：只关注最后 147 个样本的处理"""
    
    # 简化的数据集：只有 3 个完整 chunk + 1 个不完整 chunk
    dataset_size = 3 * 512 + 147  # 1683
    chunk_size = 512
    repeat_count = 5
    
    print(f"Dataset size: {dataset_size}")
    print(f"Chunk size: {chunk_size}")
    print(f"Repeat count: {repeat_count}")
    print(f"Last chunk size: {dataset_size % chunk_size}\n")
    
    # 手动模拟 RepeatSampler 的行为
    indexes = list(range(dataset_size))
    chunks = [indexes[i:i+chunk_size] for i in range(0, len(indexes), chunk_size)]
    
    print(f"Number of chunks: {len(chunks)}")
    print(f"Chunk sizes: {[len(c) for c in chunks]}\n")
    
    # 模拟 RepeatSampler 的 yield
    all_yields = []
    for chunk_idx, chunk in enumerate(chunks):
        print(f"Processing chunk {chunk_idx} (size={len(chunk)}):")
        chunk_yields = []
        for repeat_idx in range(repeat_count):
            for index in chunk:
                all_yields.append(index)
                chunk_yields.append(index)
        print(f"  Total yields from this chunk: {len(chunk_yields)}")
        print(f"  = {len(chunk)} samples * {repeat_count} repeats\n")
    
    print(f"Total yields: {len(all_yields)}")
    print(f"Expected: {dataset_size * repeat_count}\n")
    
    # 模拟 DataLoader 的 batching
    dataloader_batch_size = 512
    batches = []
    
    for i in range(0, len(all_yields), dataloader_batch_size):
        batch = all_yields[i:i+dataloader_batch_size]
        batches.append(batch)
    
    print(f"{'='*60}")
    print(f"DataLoader Batching Result:")
    print(f"{'='*60}")
    print(f"Total batches: {len(batches)}")
    
    for i, batch in enumerate(batches):
        if i < 3 or i >= len(batches) - 3:
            unique_samples = len(set(batch))
            print(f"Batch {i:2d}: size={len(batch):3d}, unique_samples={unique_samples:3d}, "
                  f"range=[{min(batch):4d}, {max(batch):4d}]")
    
    # 重点分析最后几个 batch
    print(f"\n{'='*60}")
    print(f"Focus on Last Chunk (147 samples):")
    print(f"{'='*60}")
    
    last_chunk_start_idx = 3 * 512  # 1536
    last_chunk_samples = 147
    last_chunk_total_yields = last_chunk_samples * repeat_count  # 735
    
    print(f"Last chunk original samples: {last_chunk_samples}")
    print(f"After {repeat_count} repeats: {last_chunk_total_yields} samples")
    print(f"These 735 samples will form:")
    print(f"  {last_chunk_total_yields // dataloader_batch_size} complete batch(es) "
          f"= {last_chunk_total_yields // dataloader_batch_size * dataloader_batch_size} samples")
    print(f"  + 1 partial batch = {last_chunk_total_yields % dataloader_batch_size} samples")
    print(f"  Total: {last_chunk_total_yields // dataloader_batch_size + 1} batches")
    
    # 验证
    complete_chunks_batches = (3 * 512 * repeat_count) // dataloader_batch_size
    last_chunk_batches = (last_chunk_total_yields + dataloader_batch_size - 1) // dataloader_batch_size
    
    print(f"\n{'='*60}")
    print(f"Verification:")
    print(f"{'='*60}")
    print(f"Complete chunks (3 * 512 * 5 = 7680 samples):")
    print(f"  Batches: {complete_chunks_batches}")
    print(f"Last chunk (147 * 5 = 735 samples):")
    print(f"  Batches: {last_chunk_batches}")
    print(f"Total batches: {complete_chunks_batches + last_chunk_batches}")
    print(f"Actual batches: {len(batches)}")
    print(f"Match: {'✅' if complete_chunks_batches + last_chunk_batches == len(batches) else '❌'}")

if __name__ == "__main__":
    simple_test()
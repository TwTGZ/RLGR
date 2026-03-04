import ray    
import time    
from typing import List, Dict    
import random  
from datetime import datetime  
  
  
def get_timestamp():  
    """获取精确到毫秒的时间戳"""  
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]  
  
  
# ============ Parameter Server ============    
@ray.remote    
class ParameterServer:    
    """管理模型参数的版本"""    
    def __init__(self, initial_params):    
        self.params = initial_params    
        self.version = 0    
        
    def get_params(self, version=None):    
        """获取指定版本的参数"""    
        if version is None or version == self.version:    
            return self.params, self.version    
        else:    
            return self.params, self.version    
        
    def update_params(self, new_params):    
        """更新参数"""    
        self.params = new_params    
        self.version += 1    
        return self.version    
  
  
# ============ Rollout Worker ============    
@ray.remote(num_gpus=0)  
class RolloutWorker:    
    """负责生成（模拟 vLLM）"""    
    def __init__(self, worker_id: int, param_server):    
        self.worker_id = worker_id    
        self.param_server = param_server    
        self.model_name = "mock-gpt2"  
        self.max_tokens = 50  
        self.temperature = 1.0  
        self.current_param_version = -1    
        
    def _mock_generate(self, prompts: List[str]) -> List[List[int]]:  
        """模拟文本生成，返回 token IDs"""  
        completions = []  
        # 模拟较长的生成时间（0.5-1秒）  
        generation_time = random.uniform(0.5, 1.0)  
        time.sleep(generation_time)  
          
        for prompt in prompts:  
            num_tokens = random.randint(20, self.max_tokens)  
            token_ids = [random.randint(0, 50000) for _ in range(num_tokens)]  
            completions.append(token_ids)  
          
        return completions  
      
    def rollout(self, prompts: List[str], param_version: int):    
        """执行一次 rollout"""  
        start_time = time.time()  
        start_ts = get_timestamp()  
          
        print(f"🔵 [{start_ts}] [Rollout-{self.worker_id}] START - Processing {len(prompts)} prompts")  
          
        # 1. 如果参数版本更新了，加载新参数    
        if param_version != self.current_param_version:    
            params, version = ray.get(self.param_server.get_params.remote(param_version))    
            time.sleep(0.05)  
            self.current_param_version = version    
            print(f"   [{get_timestamp()}] [Rollout-{self.worker_id}] Updated to param version {version}")    
            
        # 2. 模拟生成（这里会花费较长时间）  
        gen_start = time.time()  
        completions = self._mock_generate(prompts)  
        gen_time = time.time() - gen_start  
          
        total_time = time.time() - start_time  
        end_ts = get_timestamp()  
            
        print(f"✅ [{end_ts}] [Rollout-{self.worker_id}] DONE - Generated {len(completions)} completions in {gen_time:.3f}s (total: {total_time:.3f}s)")    
            
        return {    
            'prompts': prompts,    
            'completions': completions,    
            'param_version': param_version,    
            'gen_time': gen_time,  
            'worker_id': self.worker_id,  
            'start_time': start_ts,  
            'end_time': end_ts,  
        }    
  
  
# ============ Training Worker ============    
@ray.remote(num_gpus=0)  
class TrainingWorker:    
    """负责训练"""    
    def __init__(self, worker_id: int, param_server):    
        self.worker_id = worker_id    
        self.param_server = param_server    
        self.model_params = {"layer1": [0.1, 0.2], "layer2": [0.3, 0.4]}  
        self.step = 0    
        
    def train(self, rollout_data: Dict):    
        """训练一个 batch"""    
        start_time = time.time()  
        start_ts = get_timestamp()  
          
        print(f"🟢 [{start_ts}] [Train-{self.worker_id}] START - Training on rollout from worker-{rollout_data['worker_id']}")  
          
        # 模拟训练计算（0.3-0.6秒）  
        num_samples = len(rollout_data['prompts'])  
        train_duration = random.uniform(0.3, 0.6)  
        time.sleep(train_duration)  
          
        # 模拟参数更新  
        for key in self.model_params:  
            self.model_params[key] = [x + random.uniform(-0.01, 0.01) for x in self.model_params[key]]  
          
        mock_loss = random.uniform(0.3, 0.8)  
        train_time = time.time() - start_time    
        self.step += 1  
        end_ts = get_timestamp()  
            
        print(f"✅ [{end_ts}] [Train-{self.worker_id}] DONE - Step {self.step}, loss: {mock_loss:.4f}, time: {train_time:.3f}s")    
            
        return {    
            'loss': mock_loss,  
            'step': self.step,  
            'train_time': train_time,  
            'start_time': start_ts,  
            'end_time': end_ts,  
            'rollout_step': rollout_data.get('rollout_step', -1),  # 记录对应的 rollout step
        }    
        
    def get_params(self):    
        """获取当前模型参数"""    
        return self.model_params.copy()  
  
  
# ============ Hybrid Flow Trainer (完全异步版本) ============    
class HybridFlowTrainer:    
    """完全异步的 Hybrid Flow 实现"""    
    def __init__(    
        self,    
        num_rollout_workers: int = 2,    
        num_training_workers: int = 1,    
        sync_freq: int = 10,
        max_pending_rollouts: int = None,  # 最大并发 rollout 数
    ):    
        initial_params = {"layer1": [0.1, 0.2], "layer2": [0.3, 0.4]}  
        self.param_server = ParameterServer.remote(initial_params)    
            
        self.rollout_workers = [    
            RolloutWorker.remote(i, self.param_server)    
            for i in range(num_rollout_workers)    
        ]    
            
        self.training_workers = [    
            TrainingWorker.remote(i, self.param_server)    
            for i in range(num_training_workers)    
        ]    
            
        self.sync_freq = sync_freq    
        self.current_param_version = 0
        # 默认最大并发数为 worker 数量的 2 倍
        self.max_pending_rollouts = max_pending_rollouts or (num_rollout_workers * 2)
        
    def train(self, dataloader, num_steps: int):    
        """训练主循环 - 完全异步版本"""    
        print("=" * 80)    
        print("Starting Hybrid Flow Training (FULLY ASYNC MODE)")    
        print(f"  - Rollout workers: {len(self.rollout_workers)}")    
        print(f"  - Training workers: {len(self.training_workers)}")    
        print(f"  - Sync frequency: {self.sync_freq}")
        print(f"  - Max pending rollouts: {self.max_pending_rollouts}")
        print("=" * 80)  
        print("\n🔑 KEY: 🔵=Rollout Start, 🟢=Training Start, ✅=Task Complete\n")  
        print("=" * 80)  
            
        # 存储 future 和元数据的字典
        rollout_futures = {}  # {future: (step, launch_ts, worker_idx)}
        training_futures = {}  # {future: (step, launch_ts, rollout_step)}
        
        data_iter = iter(dataloader)    
          
        total_rollout_time = 0  
        total_training_time = 0  
        total_losses = []  
          
        # 记录所有任务的时间线  
        timeline = []
        
        # 统计信息
        completed_rollouts = 0
        completed_trainings = 0
        pending_training_data = []  # 等待训练的 rollout 数据
        
        step = 0
        
        # ========== 阶段 1: 预热 - 提交初始 rollout 任务 ==========
        print(f"\n🔥 [{get_timestamp()}] [Main] Phase 1: Warmup - Submitting initial rollouts")
        warmup_steps = min(self.max_pending_rollouts, num_steps)
        
        for i in range(warmup_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)
            
            worker_idx = i % len(self.rollout_workers)
            rollout_worker = self.rollout_workers[worker_idx]
            
            launch_ts = get_timestamp()
            print(f"⚡ [{launch_ts}] [Main] Warmup {i}: Launching rollout on worker-{worker_idx}")
            
            rollout_future = rollout_worker.rollout.remote(
                prompts=batch['prompts'],
                param_version=self.current_param_version,
            )
            rollout_futures[rollout_future] = (i, launch_ts, worker_idx)
            step += 1
        
        print(f"\n✅ [{get_timestamp()}] [Main] Warmup complete: {len(rollout_futures)} rollouts in flight")
        
        # ========== 阶段 2: 主循环 - 异步处理 ==========
        print(f"\n🚀 [{get_timestamp()}] [Main] Phase 2: Main loop - Async processing")
        
        while step < num_steps or rollout_futures or training_futures:
            # ===== 2.1 检查完成的 rollout 任务 =====
            if rollout_futures:
                ready_rollouts, _ = ray.wait(
                    list(rollout_futures.keys()),
                    num_returns=1,
                    timeout=0  # 非阻塞检查
                )
                
                for ready_future in ready_rollouts:
                    rollout_data = ray.get(ready_future)
                    rollout_step, rollout_launch, worker_idx = rollout_futures.pop(ready_future)
                    
                    completed_rollouts += 1
                    total_rollout_time += rollout_data['gen_time']
                    
                    timeline.append({
                        'type': 'rollout',
                        'step': rollout_step,
                        'start': rollout_data['start_time'],
                        'end': rollout_data['end_time'],
                        'duration': rollout_data['gen_time'],
                        'worker_id': worker_idx,
                    })
                    
                    print(f"📦 [{get_timestamp()}] [Main] Rollout {rollout_step} completed (worker-{worker_idx}), queuing for training")
                    
                    # 添加 rollout_step 信息
                    rollout_data['rollout_step'] = rollout_step
                    pending_training_data.append(rollout_data)
            
            # ===== 2.2 启动训练任务（如果有可用的 training worker）=====
            if pending_training_data and len(training_futures) < len(self.training_workers):
                rollout_data = pending_training_data.pop(0)
                rollout_step = rollout_data['rollout_step']
                
                training_worker_idx = completed_trainings % len(self.training_workers)
                training_worker = self.training_workers[training_worker_idx]
                
                train_launch_ts = get_timestamp()
                print(f"⚡ [{train_launch_ts}] [Main] Launching training for rollout-{rollout_step} on trainer-{training_worker_idx}")
                
                training_future = training_worker.train.remote(rollout_data)
                training_futures[training_future] = (completed_trainings, train_launch_ts, rollout_step)
            
            # ===== 2.3 检查完成的 training 任务 =====
            if training_futures:
                ready_trainings, _ = ray.wait(
                    list(training_futures.keys()),
                    num_returns=1,
                    timeout=0  # 非阻塞检查
                )
                
                for ready_future in ready_trainings:
                    result = ray.get(ready_future)
                    train_step, train_launch, rollout_step = training_futures.pop(ready_future)
                    
                    completed_trainings += 1
                    total_training_time += result['train_time']
                    total_losses.append(result['loss'])
                    
                    timeline.append({
                        'type': 'training',
                        'step': train_step,
                        'start': result['start_time'],
                        'end': result['end_time'],
                        'duration': result['train_time'],
                        'rollout_step': rollout_step,
                    })
                    
                    print(f"📈 [{get_timestamp()}] [Main] Training {train_step} completed (for rollout-{rollout_step}), loss: {result['loss']:.4f}")
            
            # ===== 2.4 提交新的 rollout 任务（如果还有步数且未达到并发上限）=====
            if step < num_steps and len(rollout_futures) < self.max_pending_rollouts:
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    batch = next(data_iter)
                
                worker_idx = step % len(self.rollout_workers)
                rollout_worker = self.rollout_workers[worker_idx]
                
                launch_ts = get_timestamp()
                print(f"⚡ [{launch_ts}] [Main] Step {step}: Launching rollout on worker-{worker_idx}")
                
                rollout_future = rollout_worker.rollout.remote(
                    prompts=batch['prompts'],
                    param_version=self.current_param_version,
                )
                rollout_futures[rollout_future] = (step, launch_ts, worker_idx)
                step += 1
            
            # ===== 2.5 定期同步参数 =====
            if completed_trainings > 0 and completed_trainings % self.sync_freq == 0:
                # 检查是否刚刚达到同步点
                if completed_trainings not in [t[0] for t in training_futures.values()]:
                    sync_ts = get_timestamp()
                    print(f"\n🔄 [{sync_ts}] [Main] Syncing parameters after {completed_trainings} training steps...")
                    
                    # 等待所有进行中的训练完成
                    if training_futures:
                        print(f"   [{get_timestamp()}] Waiting for {len(training_futures)} pending trainings...")
                        remaining_futures = list(training_futures.keys())
                        remaining_results = ray.get(remaining_futures)
                        
                        for future, result in zip(remaining_futures, remaining_results):
                            train_step, train_launch, rollout_step = training_futures.pop(future)
                            total_training_time += result['train_time']
                            total_losses.append(result['loss'])
                            
                            timeline.append({
                                'type': 'training',
                                'step': train_step,
                                'start': result['start_time'],
                                'end': result['end_time'],
                                'duration': result['train_time'],
                                'rollout_step': rollout_step,
                            })
                            completed_trainings += 1
                    
                    # 更新参数
                    new_params = ray.get(self.training_workers[0].get_params.remote())
                    self.current_param_version = ray.get(
                        self.param_server.update_params.remote(new_params)
                    )
                    
                    recent_losses = total_losses[-self.sync_freq:] if len(total_losses) >= self.sync_freq else total_losses
                    avg_loss = sum(recent_losses) / len(recent_losses) if recent_losses else 0
                    
                    print(f"   [{get_timestamp()}] Updated to param version {self.current_param_version}")
                    print(f"   [{get_timestamp()}] Average loss (last {len(recent_losses)} steps): {avg_loss:.4f}")
                    print(f"   [{get_timestamp()}] Progress: {completed_rollouts}/{num_steps} rollouts, {completed_trainings} trainings")
            
            # 短暂休眠避免 CPU 空转
            time.sleep(0.01)
        
        # ========== 阶段 3: 清理 - 等待所有剩余任务 ==========
        print(f"\n⏳ [{get_timestamp()}] [Main] Phase 3: Cleanup - Waiting for remaining tasks...")
        
        # 等待剩余的 rollout
        if rollout_futures:
            print(f"   [{get_timestamp()}] Waiting for {len(rollout_futures)} remaining rollouts...")
            remaining_rollout_futures = list(rollout_futures.keys())
            remaining_rollout_results = ray.get(remaining_rollout_futures)
            
            for future, rollout_data in zip(remaining_rollout_futures, remaining_rollout_results):
                rollout_step, rollout_launch, worker_idx = rollout_futures[future]
                total_rollout_time += rollout_data['gen_time']
                
                timeline.append({
                    'type': 'rollout',
                    'step': rollout_step,
                    'start': rollout_data['start_time'],
                    'end': rollout_data['end_time'],
                    'duration': rollout_data['gen_time'],
                    'worker_id': worker_idx,
                })
                
                rollout_data['rollout_step'] = rollout_step
                pending_training_data.append(rollout_data)
        
        # 启动剩余的训练
        while pending_training_data:
            rollout_data = pending_training_data.pop(0)
            rollout_step = rollout_data['rollout_step']
            
            training_worker = self.training_workers[0]
            training_future = training_worker.train.remote(rollout_data)
            training_futures[training_future] = (completed_trainings, get_timestamp(), rollout_step)
            completed_trainings += 1
        
        # 等待剩余的训练
        if training_futures:
            print(f"   [{get_timestamp()}] Waiting for {len(training_futures)} remaining trainings...")
            remaining_training_futures = list(training_futures.keys())
            remaining_training_results = ray.get(remaining_training_futures)
            
            for future, result in zip(remaining_training_futures, remaining_training_results):
                train_step, train_launch, rollout_step = training_futures[future]
                total_training_time += result['train_time']
                total_losses.append(result['loss'])
                
                timeline.append({
                    'type': 'training',
                    'step': train_step,
                    'start': result['start_time'],
                    'end': result['end_time'],
                    'duration': result['train_time'],
                    'rollout_step': rollout_step,
                })
        
        # ========== 打印分析结果 ==========
        self._print_analysis(timeline, total_rollout_time, total_training_time, total_losses)
    
    def _print_analysis(self, timeline, total_rollout_time, total_training_time, total_losses):
        """打印时间线分析"""
        print("\n" + "=" * 80)  
        print("📊 TIMELINE ANALYSIS (Proof of Asynchronous Execution)")  
        print("=" * 80)  
          
        # 按开始时间排序  
        timeline.sort(key=lambda x: x['start'])  
          
        print("\nTask Execution Timeline:")  
        print(f"{'Type':<10} {'Step':<6} {'Start':<15} {'End':<15} {'Duration':<10} {'Extra':<20}")  
        print("-" * 90)  
        for task in timeline:
            extra_info = f"worker-{task.get('worker_id', 'N/A')}" if task['type'] == 'rollout' else f"rollout-{task.get('rollout_step', 'N/A')}"
            print(f"{task['type']:<10} {task['step']:<6} {task['start']:<15} {task['end']:<15} {task['duration']:.3f}s      {extra_info}")  
          
        # 检查重叠  
        print("\n🔍 Checking for overlapping execution (proof of async):")  
        overlaps = []  
        for i in range(len(timeline)):  
            for j in range(i + 1, len(timeline)):  
                # 检查时间重叠
                if timeline[i]['end'] > timeline[j]['start'] and timeline[i]['start'] < timeline[j]['end']:  
                    overlaps.append((i, j))  
          
        if overlaps:  
            print(f"✅ Found {len(overlaps)} overlapping task pairs - ASYNC CONFIRMED!")  
            for i, j in overlaps[:10]:  # 显示前10个  
                print(f"   - {timeline[i]['type']} (step {timeline[i]['step']}) overlaps with {timeline[j]['type']} (step {timeline[j]['step']})")  
        else:  
            print("⚠️  No overlaps detected - tasks may be running sequentially")  
        
        # 计算并行度
        print("\n📈 Parallelism Analysis:")
        rollout_tasks = [t for t in timeline if t['type'] == 'rollout']
        training_tasks = [t for t in timeline if t['type'] == 'training']
        
        print(f"   - Total rollout tasks: {len(rollout_tasks)}")
        print(f"   - Total training tasks: {len(training_tasks)}")
        print(f"   - Overlapping pairs: {len(overlaps)}")
        print(f"   - Parallelism ratio: {len(overlaps) / max(len(timeline), 1):.2%}")
            
        print("\n" + "=" * 80)    
        print("Hybrid Flow Training Completed")  
        print(f"  - Total rollout time: {total_rollout_time:.2f}s")  
        print(f"  - Total training time: {total_training_time:.2f}s")  
        if total_losses:  
            print(f"  - Average loss: {sum(total_losses) / len(total_losses):.4f}")
            print(f"  - Final loss: {total_losses[-1]:.4f}")
        print("=" * 80)  
  
  
# ============ 使用示例 ============    
def main():    
    ray.init(ignore_reinit_error=True)    
        
    class DummyDataLoader:    
        def __init__(self, num_batches):    
            self.num_batches = num_batches    
            self.current = 0    
            
        def __iter__(self):    
            self.current = 0    
            return self    
            
        def __next__(self):    
            if self.current >= self.num_batches:    
                raise StopIteration    
            self.current += 1    
            return {    
                'prompts': [f"prompt_{self.current}_{i}" for i in range(4)]    
            }    
        
    dataloader = DummyDataLoader(num_batches=20)    
        
    trainer = HybridFlowTrainer(    
        num_rollout_workers=3,
        num_training_workers=2,  # 增加到2个训练worker
        sync_freq=5,
        max_pending_rollouts=6,  # 允许6个并发rollout
    )    
        
    trainer.train(dataloader, num_steps=15)    
        
    ray.shutdown()    
    
    
if __name__ == "__main__":    
    main()
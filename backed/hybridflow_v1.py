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
        }  
      
    def get_params(self):  
        """获取当前模型参数"""  
        return self.model_params.copy()

# ============ Hybrid Flow Trainer ============  
class HybridFlowTrainer:  
    """真正的 Hybrid Flow 实现"""  
    def __init__(  
        self,  
        num_rollout_workers: int = 2,  
        num_training_workers: int = 1,  
        sync_freq: int = 10,  
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
      
    def train(self, dataloader, num_steps: int):  
        """训练主循环"""  
        print("=" * 80)  
        print("Starting Hybrid Flow Training")  
        print(f"  - Rollout workers: {len(self.rollout_workers)}")  
        print(f"  - Training workers: {len(self.training_workers)}")  
        print(f"  - Sync frequency: {self.sync_freq}")  
        print("=" * 80)
        print("\n🔑 KEY: 🔵=Rollout Start, 🟢=Training Start, ✅=Task Complete\n")
        print("=" * 80)
          
        rollout_futures = []  
        training_futures = []  
        data_iter = iter(dataloader)  
        
        total_rollout_time = 0
        total_training_time = 0
        total_losses = []
        
        # 记录所有任务的时间线
        timeline = []
          
        for step in range(num_steps):  
            # ========== 1. 异步启动 Rollout ==========  
            try:  
                batch = next(data_iter)  
            except StopIteration:  
                data_iter = iter(dataloader)  
                batch = next(data_iter)  
              
            worker_idx = step % len(self.rollout_workers)  
            rollout_worker = self.rollout_workers[worker_idx]  
            
            launch_ts = get_timestamp()
            print(f"\n⚡ [{launch_ts}] [Main] Step {step}: Launching rollout on worker-{worker_idx}")
              
            rollout_future = rollout_worker.rollout.remote(  
                prompts=batch['prompts'],  
                param_version=self.current_param_version,  
            )  
            rollout_futures.append((rollout_future, step, launch_ts))
              
            # ========== 2. 如果有完成的 rollout，启动训练 ==========  
            if len(rollout_futures) >= 1:
                # 等待第一个 rollout 完成  
                rollout_data = ray.get(rollout_futures[0][0])
                _, rollout_step, rollout_launch = rollout_futures[0]
                rollout_futures = rollout_futures[1:]  
                
                total_rollout_time += rollout_data['gen_time']
                timeline.append({
                    'type': 'rollout',
                    'step': rollout_step,
                    'start': rollout_data['start_time'],
                    'end': rollout_data['end_time'],
                    'duration': rollout_data['gen_time']
                })
                
                training_worker = self.training_workers[0]
                train_launch_ts = get_timestamp()
                print(f"⚡ [{train_launch_ts}] [Main] Step {step}: Launching training")
                  
                training_future = training_worker.train.remote(rollout_data)  
                training_futures.append((training_future, step, train_launch_ts))
              
            # ========== 3. 定期同步参数 ==========  
            if (step + 1) % self.sync_freq == 0:  
                sync_ts = get_timestamp()
                print(f"\n🔄 [{sync_ts}] [Main] Syncing parameters...")  
                  
                if training_futures:  
                    training_results = []
                    for future, train_step, train_launch in training_futures:
                        result = ray.get(future)
                        training_results.append(result)
                        timeline.append({
                            'type': 'training',
                            'step': train_step,
                            'start': result['start_time'],
                            'end': result['end_time'],
                            'duration': result['train_time']
                        })
                    
                    training_futures = []  
                    
                    for result in training_results:
                        total_training_time += result['train_time']
                        total_losses.append(result['loss'])
                    
                    print(f"   [{get_timestamp()}] Completed {len(training_results)} training steps")  
                    print(f"   Average loss: {sum(total_losses[-len(training_results):]) / len(training_results):.4f}")
                  
                new_params = ray.get(self.training_workers[0].get_params.remote())  
                self.current_param_version = ray.get(  
                    self.param_server.update_params.remote(new_params)  
                )  
                  
                print(f"   [{get_timestamp()}] Updated to param version {self.current_param_version}")  
          
        # 等待所有任务完成  
        print(f"\n⏳ [{get_timestamp()}] [Main] Waiting for remaining tasks...")  
        
        if rollout_futures:  
            for future, step, launch in rollout_futures:
                rollout_data = ray.get(future)
                total_rollout_time += rollout_data['gen_time']
                timeline.append({
                    'type': 'rollout',
                    'step': step,
                    'start': rollout_data['start_time'],
                    'end': rollout_data['end_time'],
                    'duration': rollout_data['gen_time']
                })
                
        if training_futures:  
            for future, step, launch in training_futures:
                result = ray.get(future)
                total_training_time += result['train_time']
                total_losses.append(result['loss'])
                timeline.append({
                    'type': 'training',
                    'step': step,
                    'start': result['start_time'],
                    'end': result['end_time'],
                    'duration': result['train_time']
                })
        
        # 打印时间线分析
        print("\n" + "=" * 80)
        print("📊 TIMELINE ANALYSIS (Proof of Asynchronous Execution)")
        print("=" * 80)
        
        # 按开始时间排序
        timeline.sort(key=lambda x: x['start'])
        
        print("\nTask Execution Timeline:")
        print(f"{'Type':<10} {'Step':<6} {'Start':<15} {'End':<15} {'Duration':<10}")
        print("-" * 80)
        for task in timeline:
            print(f"{task['type']:<10} {task['step']:<6} {task['start']:<15} {task['end']:<15} {task['duration']:.3f}s")
        
        # 检查重叠
        print("\n🔍 Checking for overlapping execution (proof of async):")
        overlaps = []
        for i in range(len(timeline) - 1):
            for j in range(i + 1, len(timeline)):
                # 简单检查：如果两个任务的时间戳有重叠
                if timeline[i]['end'] > timeline[j]['start'] and timeline[i]['start'] < timeline[j]['end']:
                    overlaps.append((i, j))
        
        if overlaps:
            print(f"✅ Found {len(overlaps)} overlapping task pairs - ASYNC CONFIRMED!")
            for i, j in overlaps[:5]:  # 只显示前5个
                print(f"   - {timeline[i]['type']} (step {timeline[i]['step']}) overlaps with {timeline[j]['type']} (step {timeline[j]['step']})")
        else:
            print("⚠️  No overlaps detected - tasks may be running sequentially")
          
        print("\n" + "=" * 80)  
        print("Hybrid Flow Training Completed")
        print(f"  - Total rollout time: {total_rollout_time:.2f}s")
        print(f"  - Total training time: {total_training_time:.2f}s")
        if total_losses:
            print(f"  - Average loss: {sum(total_losses) / len(total_losses):.4f}")
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
        num_rollout_workers=3,  # 增加到3个worker
        num_training_workers=1,  
        sync_freq=5,  
    )  
      
    trainer.train(dataloader, num_steps=15)  
      
    ray.shutdown()  
  
  
if __name__ == "__main__":  
    main()
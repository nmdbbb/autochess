import ray
import numpy as np
from typing import List, Tuple, Dict, Any
import torch
from collections import deque
import random
import time

@ray.remote
class ReplayBuffer:
    def __init__(self, buffer_size: int, batch_size: int):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer = deque(maxlen=buffer_size)
        self.priority_buffer = deque(maxlen=buffer_size)
        self.total_samples = 0
        
    def add(self, state: torch.Tensor, policy: np.ndarray, value: float, 
            priority: float = 1.0) -> None:
        """Add a new experience to the buffer"""
        self.buffer.append((state, policy, value))
        self.priority_buffer.append(priority)
        self.total_samples += 1
        
    def add_batch(self, experiences: List[Tuple[torch.Tensor, np.ndarray, float]],
                 priorities: List[float] = None) -> None:
        """Add a batch of experiences to the buffer"""
        if priorities is None:
            priorities = [1.0] * len(experiences)
            
        for exp, priority in zip(experiences, priorities):
            self.add(*exp, priority)
            
    def sample(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a batch of experiences"""
        if len(self.buffer) < self.batch_size:
            return None
            
        # Use priority sampling
        priorities = np.array(self.priority_buffer)
        probs = priorities / priorities.sum()
        indices = np.random.choice(len(self.buffer), self.batch_size, p=probs)
        
        batch = [self.buffer[i] for i in indices]
        states, policies, values = zip(*batch)
        
        return (
            torch.stack(states),
            torch.tensor(policies, dtype=torch.float32),
            torch.tensor(values, dtype=torch.float32).unsqueeze(1)
        )
        
    def update_priorities(self, indices: List[int], priorities: List[float]) -> None:
        """Update priorities for specific experiences"""
        for idx, priority in zip(indices, priorities):
            if idx < len(self.priority_buffer):
                self.priority_buffer[idx] = priority
                
    def size(self) -> int:
        """Get current size of the buffer"""
        return len(self.buffer)
        
    def total_samples_seen(self) -> int:
        """Get total number of samples seen"""
        return self.total_samples

@ray.remote
class ReplayBufferCoordinator:
    def __init__(self, num_buffers: int, buffer_size: int, batch_size: int):
        self.num_buffers = num_buffers
        self.buffers = [
            ReplayBuffer.remote(buffer_size, batch_size)
            for _ in range(num_buffers)
        ]
        self.batch_size = batch_size
        
    def add_experience(self, buffer_idx: int, state: torch.Tensor, 
                      policy: np.ndarray, value: float, priority: float = 1.0) -> None:
        """Add experience to a specific buffer"""
        ray.get(self.buffers[buffer_idx].add.remote(state, policy, value, priority))
        
    def add_batch(self, buffer_idx: int, experiences: List[Tuple[torch.Tensor, np.ndarray, float]],
                 priorities: List[float] = None) -> None:
        """Add a batch of experiences to a specific buffer"""
        ray.get(self.buffers[buffer_idx].add_batch.remote(experiences, priorities))
        
    def sample(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample from all buffers"""
        # Get samples from all buffers
        samples = ray.get([buffer.sample.remote() for buffer in self.buffers])
        samples = [s for s in samples if s is not None]
        
        if not samples:
            return None
            
        # Combine samples
        states = torch.cat([s[0] for s in samples])
        policies = torch.cat([s[1] for s in samples])
        values = torch.cat([s[2] for s in samples])
        
        # Shuffle the combined batch
        indices = torch.randperm(len(states))
        return (
            states[indices][:self.batch_size],
            policies[indices][:self.batch_size],
            values[indices][:self.batch_size]
        )
        
    def update_priorities(self, buffer_idx: int, indices: List[int], 
                         priorities: List[float]) -> None:
        """Update priorities in a specific buffer"""
        ray.get(self.buffers[buffer_idx].update_priorities.remote(indices, priorities))
        
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the buffers"""
        sizes = ray.get([buffer.size.remote() for buffer in self.buffers])
        total_samples = ray.get([buffer.total_samples_seen.remote() for buffer in self.buffers])
        
        return {
            'buffer_sizes': sizes,
            'total_samples': sum(total_samples),
            'avg_buffer_size': sum(sizes) / len(sizes),
            'min_buffer_size': min(sizes),
            'max_buffer_size': max(sizes)
        } 
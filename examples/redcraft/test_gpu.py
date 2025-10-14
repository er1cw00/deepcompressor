import torch
import time

def measure_gpu_performance(device_id=0, matrix_size=4096, warmups=10, iterations=100):
    """
    测试指定 GPU 的矩阵乘法性能 (GFLOPS) 和内存拷贝速度 (GB/s)。

    Args:
        device_id (int): 要测试的 GPU 设备 ID。
        matrix_size (int): 用于矩阵乘法测试的矩阵边长 (N)。
        warmups (int): 热身运行次数。
        iterations (int): 实际计时迭代次数。
    """
    if not torch.cuda.is_available():
        print("CUDA 不可用，请检查 GPU 配置。")
        return

    device = torch.device(f'cuda:{device_id}')
    
    # ----------------------------------------------------
    # 1. 矩阵乘法性能测试 (GFLOPS)
    # ----------------------------------------------------
    print(f"\n--- 1. 矩阵乘法性能测试 (GFLOPS) ---")
    
    # 使用 FP16 (Half precision) 进行测试，这在现代 GPU 上更快
    dtype = torch.float16 
    
    # 创建方阵 A 和 B (N x N)
    N = matrix_size
    A = torch.randn(N, N, dtype=dtype, device=device)
    B = torch.randn(N, N, dtype=dtype, device=device)
    
    # 每次矩阵乘法的浮点运算次数 (FLOPS)： 2 * N^3
    flops_per_op = 2 * N**3
    
    # 热身运行 (Warmup)
    for _ in range(warmups):
        torch.matmul(A, B)
    
    # 开始计时
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(iterations):
        torch.matmul(A, B)
    end_event.record()
    
    # 等待 GPU 任务完成
    torch.cuda.synchronize()
    
    # 计算总时间 (毫秒)
    total_time_ms = start_event.elapsed_time(end_event)
    total_time_s = total_time_ms / 1000.0
    
    # 计算 GFLOPS
    total_flops = flops_per_op * iterations
    gflops = total_flops / (total_time_s * 1e9)
    
    print(f"矩阵大小 (N): {N}x{N}, 数据类型: {dtype}")
    print(f"平均 GFLOPS:   {gflops:.2f} GFLOPS")


    # ----------------------------------------------------
    # 2. 内存拷贝速度测试 (GB/s)
    # ----------------------------------------------------
    print(f"\n--- 2. 内存拷贝速度测试 (GB/s) ---")
    
    # 使用一个大 Tensor (4GB)
    tensor_size_bytes = 4 * 1024**3 
    elements = tensor_size_bytes // torch.float32.itemsize 
    
    # 创建 CPU Tensor
    cpu_tensor = torch.randn(elements, dtype=torch.float32)
    
    # 热身运行
    for _ in range(warmups):
        cpu_tensor.to(device)
    
    # 开始计时
    start_event.record()
    for _ in range(iterations):
        # 测量 CPU -> GPU 拷贝时间
        cpu_tensor.to(device)
    end_event.record()
    
    torch.cuda.synchronize()
    total_time_ms = start_event.elapsed_time(end_event)
    total_time_s = total_time_ms / 1000.0
    
    # 计算带宽
    total_data_bytes = tensor_size_bytes * iterations
    gb_per_sec = total_data_bytes / total_time_s / 1e9
    
    print(f"拷贝数据大小: {tensor_size_bytes / 1024**3:.0f} GB")
    print(f"平均带宽 (CPU -> GPU): {gb_per_sec:.2f} GB/s")


if __name__ == '__main__':
    # 检查并选择 GPU
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("未检测到 CUDA 设备。")
    elif num_gpus == 1:
        print(f"检测到 1 个 GPU: {torch.cuda.get_device_name(0)}")
        measure_gpu_performance(device_id=0)
    else:
        print(f"检测到 {num_gpus} 个 GPU。")
        for i in range(num_gpus):
            print(f"[{i}]: {torch.cuda.get_device_name(i)}")
        
        try:
            target_gpu = int(input("请输入要测试的 GPU ID: "))
            if 0 <= target_gpu < num_gpus:
                measure_gpu_performance(device_id=target_gpu)
            else:
                print("无效的 GPU ID。")
        except ValueError:
            print("输入无效。")

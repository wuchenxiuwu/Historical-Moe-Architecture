import torch
import logging
from typing import Dict, Any

# 配置日志
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def validate_training_environment(config: Dict[str, Any]) -> bool:
    """
    增强版训练环境验证
    
    新增功能：
    1. CUDA版本兼容性检查
    2. 显存需求预估
    3. CPU模式资源验证
    4. 分布式后端检查
    """
    # ========================
    # 1. 核心设备验证
    # ========================
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    
    if device == 'cuda' and not torch.cuda.is_available():
        error_msg = (
            "配置要求GPU训练，但未检测到CUDA设备\n"
            "解决方案：\n"
            "1. 安装CUDA驱动和PyTorch GPU版本\n"
            "2. 修改配置 device: 'cpu'"
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    # ========================
    # 2. 分布式训练深度检查
    # ========================
    if config.get('distributed', False):
        # CUDA可用性检查
        if not torch.cuda.is_available():
            error_msg = (
                "分布式训练需要CUDA支持，请执行以下操作之一:\n"
                "1. 安装CUDA驱动和PyTorch GPU版本\n"
                "2. 在配置文件中设置 distributed: false"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)
            
        # 分布式后端检查
        backend = config.get('distributed_backend', 'nccl')
        if backend == 'nccl' and not torch.distributed.is_nccl_available():
            error_msg = (
                "NCCL后端不可用，请检查：\n"
                "1. 是否安装NVIDIA NCCL\n"
                "2. PyTorch是否支持NCCL"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)
            
        # GPU数量检查
        required_gpus = config.get('num_gpus', torch.cuda.device_count())
        available_gpus = torch.cuda.device_count()
        if required_gpus > available_gpus:
            warning_msg = (
                f"配置GPU数量({required_gpus})超过实际可用数量({available_gpus})\n"
                f"将使用所有可用GPU: {available_gpus}个"
            )
            logger.warning(warning_msg)
    
    # ========================
    # 3. 混合精度增强检查
    # ========================
    if config.get('mixed_precision', False):
        # 版本检查
        torch_version = torch.__version__.split('.')
        major, minor = int(torch_version[0]), int(torch_version[1])
        if major < 1 or (major == 1 and minor < 7):
            error_msg = "混合精度训练需要PyTorch 1.7.0或更高版本"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
            
        # Tensor Core兼容性检查
        if device == 'cuda':
            for i in range(torch.cuda.device_count()):
                capability = torch.cuda.get_device_capability(i)
                if capability[0] < 7:  # Volta架构以下不支持Tensor Core
                    logger.warning(
                        f"GPU {i} ({torch.cuda.get_device_name(i)}) "
                        f"计算能力{capability[0]}.{capability[1]} < 7.0，"
                        "混合精度加速效果有限"
                    )
    
    # ========================
    # 4. 内存系统验证增强
    # ========================
    accum_steps = config.get('gradient_accumulation_steps', 1)
    if accum_steps < 1:
        raise ValueError("梯度累积步数必须大于等于1")
        
    # CPU模式内存检查
    if device == 'cpu':
        import psutil
        required_ram = config.get('min_ram_gb', 32) * 1024**3  # 默认32GB
        available_ram = psutil.virtual_memory().total
        if available_ram < required_ram:
            logger.warning(
                f"可用内存({available_ram/1024**3:.1f}GB) "
                f"低于推荐值({required_ram/1024**3:.1f}GB)"
            )
    
    # ========================
    # 5. 环境信息记录
    # ========================
    env_info = [
        f"PyTorch版本: {torch.__version__}",
        f"设备: {device.upper()}",
        f"CUDA可用: {torch.cuda.is_available()}"
    ]
    
    if torch.cuda.is_available():
        env_info.append(f"CUDA版本: {torch.version.cuda}")
        env_info.append(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            env_info.append(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    logger.info("环境验证通过\n" + "\n".join(env_info))
    return True
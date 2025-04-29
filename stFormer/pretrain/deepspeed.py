"""
Utility to launch stFormer training with DeepSpeed.

Example usage:
    from stformer_deepspeed import launch_deepspeed

    launch_deepspeed(
        script_path="path/to/stformer_pretrainer.py",
        deepspeed_config="ds_config.json",
        num_gpus=12,
        num_nodes=3,
    )
"""

import subprocess


def launch_deepspeed(
    script_path: str,
    deepspeed_config: str,
    num_gpus: int = 12,
    num_nodes: int = 1,
) -> None:
    """
    Invoke DeepSpeed to run the given training script.

    Parameters:
      script_path: Path to the Python training module.
      deepspeed_config: DeepSpeed JSON config file path.
      num_gpus: Number of GPUs per node.
      num_nodes: Number of nodes.
    """
    cmd = [
        'deepspeed',
        f'--num_gpus={num_gpus}',
        f'--num_nodes={num_nodes}',
        script_path,
        '--deepspeed', deepspeed_config,
    ]
    subprocess.run(cmd, check=True)
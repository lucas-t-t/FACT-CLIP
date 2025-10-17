#!/usr/bin/env python3
import os, subprocess, getpass

# Import torch only AFTER the mask is in the environment
import torch                               # torch doc: turn0search1

def main():
    user = getpass.getuser()
    visible = os.getenv("CUDA_VISIBLE_DEVICES", "All GPUs visible")
    print(f"User           : {user}")
    print(f"CUDA_VISIBLE_DEVICES = {visible}\n")

    n = torch.cuda.device_count()
    print(f"PyTorch sees   : {n} GPU(s)\n")

    for i in range(n):
        name = torch.cuda.get_device_name(i)
        total = torch.cuda.get_device_properties(i).total_memory // (1024**2)
        print(f"Logical GPU {i}: {name} ({total} MiB total)")

    print("\nActive GPU processes (nvidia-smi):")
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL)
        print(out.strip() or "  <none>")
    except Exception as e:
        print("  Could not run nvidia-smi:", e)

if __name__ == "__main__":
    main()



#!/bin/bash
# Record host + GPU memory while a server starts and serves, to measure what a recipe really needs.
# Start it BEFORE launching the server; stop it (Ctrl-C) after your sweep. Writes one CSV line per second.
#   tools/memwatch.sh memwatch.csv
# Columns: unix_s, MemAvailable_GB, SwapUsed_GB, gpu_used_MiB (nvidia-smi, all GPUs summed; empty on non-NVIDIA)
OUT=${1:-memwatch.csv}
echo "t,mem_available_gb,swap_used_gb,gpu_used_mib" > "$OUT"
while true; do
  read -r ma st sf < <(awk '/MemAvailable/{a=$2}/SwapTotal/{t=$2}/SwapFree/{f=$2}END{print a,t,f}' /proc/meminfo)
  g=$(timeout 5 nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | paste -sd+ | bc 2>/dev/null)
  echo "$(date +%s),$(echo "scale=2;$ma/1000000" | bc),$(echo "scale=2;($st-$sf)/1000000" | bc),$g" >> "$OUT"
  sleep 1
done

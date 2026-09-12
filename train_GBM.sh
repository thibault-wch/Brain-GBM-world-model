# copy pre-trained model
# mkdir /data/chwang/experiments/glioma/template;
# cd ./glioma/template;
source /etc/network_turbo
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export HF_ENDPOINT=https://hf-mirror.com


CUDA_VISIBLE_DEVICES=0,1 \
accelerate launch \
  --config_file /root/remoteproject/glioma/configs/accelerate_configs/multi_nodes/8_gpus_node_2.yaml \
  /root/remoteproject/glioma/train_stage_two.py \
  config=/root/remoteproject/glioma/configs/showo2_1.5b_stage_2_a.yaml

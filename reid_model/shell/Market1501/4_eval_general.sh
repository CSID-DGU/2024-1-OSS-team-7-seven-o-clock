cd 
cd amd/reid_model

gpus='1'

# ps -ef | grep "python3 ./projects/InterpretationReID" | awk '{print $2}' | xargs kill
CUDA_VISIBEVICES=$gpus python3 ./projects/InterpretationReID/general_evaluation.py  --config-file ./projects/InterpretationReID/configs/Market1501_Circle/circle_R50_ip_eval_only.yml --eval-only  MODEL.DEVICE "cuda:0" 
# CUDA_VISIBEVICES='0' --config-file ./projects/InterpretationReID/configs/Market1501_Circle/circle_R50_ip.yml  MODEL.DEVICE "cuda:0"
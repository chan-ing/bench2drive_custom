#!/bin/bash
BASE_PORT=20200
BASE_TM_PORT=40200
IS_BENCH2DRIVE=True

CONFIG_NAME=hipad_b2d_stage3_tiny-official_ACT-Quant-regress-resume_debug

TEAM_AGENT=bench2drive/leaderboard/team_code/hipad_b2d_agent.py
TEAM_CONFIG=/mnt/sda/HiP-AD-custom/projects/configs/hipad_b2d_stage3_tiny-official_ACT-Quant-regress-resume.py+\
/mnt/sda/HiP-AD-custom/eval_ckpts/hipad_b2d_stage3_tiny-official_ACT-Quant-regress-resume/latest.pth

PLANNER_TYPE=traj
BASE_ROUTES=bench2drive/leaderboard/data/splits4/bench2drive220

SAVE_PATH=evaluation/$CONFIG_NAME
BASE_CHECKPOINT_ENDPOINT=evaluation/$CONFIG_NAME/$CONFIG_NAME

if [ ! -d "$SAVE_PATH" ]; then
    mkdir -p "$SAVE_PATH"
fi

echo -e "**************Please Manually adjust GPU or TASK_ID **************"
GPU_RANK_LIST=(0)
TASK_LIST=(0)

echo -e "TASK_LIST: $TASK_LIST"
echo -e "GPU_RANK_LIST: $GPU_RANK_LIST"
echo -e "\033[36m***********************************************************************************\033[0m"

length=${#TASK_LIST[@]}
for ((i=0; i<$length; i++ )); do
    PORT=$((BASE_PORT + i * 200))
    TM_PORT=$((BASE_TM_PORT + i * 200))
    ROUTES="${BASE_ROUTES}_${TASK_LIST[$i]}.xml"
    CHECKPOINT_ENDPOINT="${BASE_CHECKPOINT_ENDPOINT}_${TASK_LIST[$i]}.json"
    GPU_RANK=${GPU_RANK_LIST[$i]}

    echo -e "TASK_ID: $i"
    echo -e "PORT: $PORT"
    echo -e "TM_PORT: $TM_PORT"
    echo -e "CHECKPOINT_ENDPOINT: $CHECKPOINT_ENDPOINT"
    echo -e "GPU_RANK: $GPU_RANK"
    echo -e "bash bench2drive/leaderboard/scripts/run_evaluation.sh $PORT $TM_PORT $IS_BENCH2DRIVE $ROUTES $TEAM_AGENT $TEAM_CONFIG $CHECKPOINT_ENDPOINT $SAVE_PATH $PLANNER_TYPE $GPU_RANK"
    echo -e "\033[36m***********************************************************************************\033[0m"
    bash -e bench2drive/leaderboard/scripts/run_evaluation.sh $PORT $TM_PORT $IS_BENCH2DRIVE $ROUTES $TEAM_AGENT $TEAM_CONFIG $CHECKPOINT_ENDPOINT $SAVE_PATH $PLANNER_TYPE $GPU_RANK # 2>&1 >> ${BASE_CHECKPOINT_ENDPOINT}_${TASK_LIST[$i]}.log &
    sleep 5
done
wait

# bash -e bench2drive/leaderboard/scripts/run_evaluation.sh \
#   20200 40200 True \
#   bench2drive/leaderboard/data/splits4/bench2drive220_0.xml \
#   bench2drive/leaderboard/team_code/hipad_b2d_agent.py \
#   /mnt/sdc/hc/HiP-AD/projects/configs/hipad_b2d_stage2_fp32_neck_eval.py+/mnt/sdc/hc/HiP-AD/work_dirs/hipad_b2d_stage2_fp32_neck_eval/latest.pth \
#   evaluation/hipad_b2d_stage2_fp32_neck_eval/hipad_b2d_stage2_fp32_neck_eval_0.json \
#   evaluation/hipad_b2d_stage2_fp32_neck_eval traj 0 \
#   2>&1 > evaluation/hipad_b2d_stage2_fp32_neck_eval/hipad_b2d_stage2_fp32_neck_eval_0.log &

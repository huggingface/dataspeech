#!/bin/bash
#SBATCH --job-name=llm-swarm
#SBATCH --partition hopper-prod
#SBATCH --gpus={{gpus}}
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=11G
#SBATCH -o slurm/logs/%x_%j.out

source ~/.bashrc

# For HF cluster internal users: Check if /fsx directory exists
if [ -d "/fsx/.cache" ]; then
    export volume="/fsx/.cache"
else
    export volume=".cache"
fi

export model={{model}}
export revision={{revision}}

function unused_port() {
    N=${1:-1}
    comm -23 \
        <(seq "1025" "65535" | sort) \
        <(ss -Htan |
            awk '{print $4}' |
            cut -d':' -f2 |
            sort -u) |
        shuf |
        head -n "$N"
}
export PORT=$(unused_port)

if [ -z "$HUGGING_FACE_HUB_TOKEN" ]; then
    # try reading from file
    export HUGGING_FACE_HUB_TOKEN=$(cat "${HF_HOME}"/token)
fi

echo "Starting TGI container port $PORT"
echo "http://$(hostname -I | awk '{print $1}'):$PORT" >> {{slurm_hosts_path}}

# unset cache dirs to avoid pyxis having host env var somehow get into the container
unset HF_HUB_CACHE HF_ASSETS_CACHE HF_DATASETS_CACHE HF_MODULES_CACHE
srun --container-image='ghcr.io/huggingface/text-generation-inference:2.0' \
    --container-env=HUGGING_FACE_HUB_TOKEN,PORT \
    --container-mounts="$volume:/data" \
    --no-container-mount-home \
    --qos normal \
    /usr/local/bin/text-generation-launcher \
    --model-id $model \
    --revision $revision \
    --max-concurrent-requests 2000 \
    --max-total-tokens {{model_max_length}} \
    --max-input-length {{model_input_length}} \
    --max-batch-prefill-tokens {{model_max_length}} \

echo "End of job"
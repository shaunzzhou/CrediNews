#!/bin/sh

BASEDIR=$(dirname "$0")
cd $BASEDIR/..
echo Current Directory:
pwd

export DATA_PATH=$1
export RESULTS_PATH=results/llm_rewrite_results_${DATA_PATH}

echo "Evaluating LLM on $DATA_PATH dataset; output: $RESULTS_PATH"

python llm_toolkit/eval_openai.py

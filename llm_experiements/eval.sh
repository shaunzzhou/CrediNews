#!/bin/sh

BASEDIR=$(dirname "$0")
cd $BASEDIR/..
echo Current Directory:
pwd

DATA_FILE=$1
export DATA_PATH=dataset/$DATA_FILE
export RESULTS_PATH=results/llm_rewrite_results_${DATA_FILE}

echo "Evaluating LLM on $DATA_PATH dataset; output: $RESULTS_PATH"

python llm_toolkit/eval_openai.py

# export http_proxy=http://127.0.0.1:7890
# export https_proxy=http://127.0.0.1:7890
# export HF_HOME=/userhome/huggingface
export LMUData=/userhome/Dataset/LMUData
# torchrun --nproc-per-node=8 run.py \
#     --data MMBench_DEV_EN_V11 MMBench_DEV_CN_V11 AI2D CCBench HallusionBench MME MMStar OCRBench POPE RealWorldQA ScienceQA SEEDBench_IMG TextVQA_VAL \
#     --model llava_llama3_8b_graph_prompt
torchrun --nproc-per-node=8 run.py --data=MME --model=llava_v1.5_7b --work-dir=./results
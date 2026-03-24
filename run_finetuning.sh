# ===================== 批量运行脚本 =====================
# 运行对抗微调脚本
echo -e "\nRunning adversarial_fine-tuning.py ..."
python adversarial_fine-tuning.py --dataset animals10  --victim byol --gpu 0

# 运行标准微调脚本
echo -e "\nRunning standard_finetuning.py ..."
python standard_finetuning.py --dataset animals10 --victim byol --gpu 0










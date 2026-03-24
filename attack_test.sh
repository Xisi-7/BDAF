DATASET="animals10"
VICTIM="byol"
GPU="0"

echo "Dataset: $DATASET"
echo "Victim: $VICTIM"

python UAP_rob.py --dataset $DATASET --victim $VICTIM --gpu $GPU
python UAPEPGD_rob.py --dataset $DATASET --victim $VICTIM --gpu $GPU
python SSP_rob.py --dataset $DATASET --victim $VICTIM --gpu $GPU
python PAP_rob.py --dataset $DATASET --victim $VICTIM --gpu $GPU
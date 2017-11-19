cd models
bash download_atten.sh
cd ..
echo 'done'
python model_atten.py "$@"
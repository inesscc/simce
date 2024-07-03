#!/bin/bash



# Run finetune.py with the first config file
python prueba_finetune.py --resume /home/ubuntu/simce/saved/models/maxvit_t/0702_211954/model_best.pt

# Run finetune.py with the first config file
python prueba_finetune.py --config config/model_m_r.json

# Run finetune.py with the first config file
python prueba_finetune.py --config config/model_l_nr.json

# Run finetune.py with the second config file
python prueba_finetune.py --config config/model_l_r.json
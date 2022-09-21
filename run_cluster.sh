#!/bin/bash 

#===============================================================================
# exemples d'options
 
#SBATCH --partition=court     # choix de la partition où soumettre le job
#SBATCH --ntasks=1            # nb de tasks total pour le job
#SBATCH --cpus-per-task=20    # nb CPU pour une task
#SBATCH --mem=64000           # mémoire nécessaire (par noeud) en Mo
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=helene.tran@doctorant.uca.fr
 
#===============================================================================

python3 main.py \
--pickle_name_dataset cmu_mosei_aligned_text_no_averaged \
--pickle_name_fold cmu_mosei \
--align_to_text \
--append_label_to_data \
--image_feature facet \
--model_name ef_williams \
--num_layers 1 \
--num_nodes 100 \
--dropout_rate 0.1 \
--final_activ linear \
--num_epochs 1000 \
--patience 10 \
--batch_size 64 \
--fixed_num_steps 15 \
--label_type present \
--optimizer adam \
--loss_function mean_absolute_error \
--learning_rate 0.001 \
--val_metric loss \
--threshold_emo_present 0 0.25 0.5 \
--round_decimals 3 \
--save_confusion_matrix \
--display_fig \
--save_predictions
#--predict_neutral_class \ #Do not use this parameter for the moment
#--with_custom_split \
#--display_fig

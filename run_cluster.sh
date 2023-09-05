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
--image_feature facet \
--model_name ef_williams \
--label_type all \
--set_length avg \
--threshold_emo_present 0 0.25 0.5 \
--round_decimals 3 \
--plot_confusion_matrix \
--save_predictions
#--predict_neutral_class \
#--with_custom_split \
#--display_fig

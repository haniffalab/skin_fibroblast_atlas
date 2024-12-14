#!/bin/bash
GSE_ID=$1
base_dir=/nfs/team298/ls34/reprocess_public_10x/$GSE_ID
echo $base_dir
IMAGE="/nfs/cellgeni/singularity/images/cellbender-0.3.0.sif"
for file in $base_dir/*; do
    SAMPLE=$(basename "$file")
    if [[ $SAMPLE == GSM* || $SAMPLE == SRS* ]] && [[ ! $SAMPLE == *.tsv ]]; then
        echo $SAMPLE
        OUTPUT_PATH=$base_dir/$SAMPLE/cellbender_v3/
        OUTPUT_FILE=$OUTPUT_PATH/${SAMPLE}_cellbender_out_filtered.h5
        if [[ -e $OUTPUT_FILE ]]; then
            echo "File $OUTPUT_FILE already exists, skipping..."
            continue
        fi
        if [[ ! -d $OUTPUT_PATH ]]; then
            mkdir -p $OUTPUT_PATH
        fi
        #INPUT=$base_dir/$SAMPLE/output/GeneFull/raw/ 
        #echo $INPUT
        /software/singularity/3.11.4/bin/singularity run --nv --bind /nfs,/lustre $IMAGE cellbender remove-background --low-count-threshold 100 \
             --input $base_dir/$SAMPLE/output/GeneFull/raw/ \
             --output $OUTPUT_FILE \
             --cuda 
        echo "DONE"      
    fi
done

CRITERION=sparsemax_exact
# CRITERION=cross_entropy
# CRITERION=entmax15_exact
# CRITERION=entmax_bisect
# CRITERION=entmax_nsect

# now you must configure this too.
# if this causes errors while validating, set it to 1
VALID_ALPHA=1.5

#export CUDA_VISIBLE_DEVICES=0

fairseq-train \
    /home/timpijnacker/.pyenv/versions/3.10.4/envs/venv3104/lib/python3.10/site-packages/fairseq/data-bin/iwslt14.tokenized.de-en \
    --user-dir /home/timpijnacker/Thesis/fairseq/sfseq \
    --arch transformer_sparse_out \
    --share-decoder-input-output-embed \
    --encoder-embed-dim 512 \
    --encoder-ffn-embed-dim 1024 \
    --encoder-attention-heads 4 \
    --encoder-layers 6 \
    --decoder-embed-dim 512 \
    --decoder-ffn-embed-dim 1024 \
    --decoder-attention-heads 4 \
    --decoder-layers 6 \
    --optimizer adam \
    --adam-betas '(0.9, 0.98)' \
    --clip-norm 0.0 \
    --lr 5e-4 \
    --lr-scheduler inverse_sqrt \
    --warmup-updates 4000 \
    --dropout 0.3 \
    --weight-decay 0.0001 \
    --criterion ${CRITERION} \
    --proba-mapping-alpha ${VALID_ALPHA} \
    --max-tokens 4096 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu \
    --maximize-best-checkpoint-metric \
    --wandb-project fast_sparse_seq2seq \
    --validate-interval-updates 1 \
    --save-dir ${CRITERION}

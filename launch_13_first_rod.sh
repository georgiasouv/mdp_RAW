cd ~/raw-mdp && for TAG in \
  solo_1_normgrad solo_2_normgrad solo_T_normgrad \
  pair_12_normgrad pair_T1_normgrad pair_T2_normgrad triple_normgrad \
  homo_1_normgrad homo_2_normgrad homo_T_normgrad \
  pair_12_sum pair_T1_sum triple_sum \
  pair_12_mgda; do \
  f="cluster_scripts/rod/run_${TAG}.sh"; \
  if [ -f "$f" ]; then echo "submitting $TAG"; sbatch "$f"; else echo "MISSING: $f"; fi; \
done
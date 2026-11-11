#!/usr/bin/env python3
"""Generate final two-panel summary figures from committed JSON summaries."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
DEV=json.loads((ROOT/'results/development/day5_audit.json').read_text())
FINAL=json.loads((ROOT/'results/final_holdout/final_holdout_summary.json').read_text())
OUT=ROOT/'paper/figures';OUT.mkdir(exist_ok=True,parents=True)

ordered=['c0','c1','c2'];labels=['Base (C0)','Success (C1)','Recovery (C2)']
d=DEV['fixed_state_after_excluding_officially_correct_initial_states']
dev_n=d['n']; hold_n=FINAL['true_failed_starts']
dev_hits=[d['by_model'][m]['passed'] for m in ordered]
hold_hits=[FINAL['by_model'][m]['passed'] for m in ordered]
fig,ax=plt.subplots(figsize=(6.2,3.05));w=.33
bs1=ax.bar([i-w/2 for i in range(3)], [100*v/dev_n for v in dev_hits], w,
           label=f'Development ({dev_n} incorrect starts)',color='#536477')
bs2=ax.bar([i+w/2 for i in range(3)], [100*v/hold_n for v in hold_hits], w,
           label=f'Frozen holdout ({hold_n} incorrect starts)',color='#bf895e')
for bars,nums,denom in [(bs1,dev_hits,dev_n),(bs2,hold_hits,hold_n)]:
 for b,h in zip(bars,nums): ax.annotate(f'{h}/{denom}',(b.get_x()+b.get_width()/2,b.get_height()),
                                        xytext=(0,3),textcoords='offset points',ha='center',va='bottom',fontsize=8)
ax.set_xticks(range(3),labels);ax.set_ylim(0,22);ax.set_ylabel('Official MBPP+ repaired (%)')
for s in ('top','right'):ax.spines[s].set_visible(False)
ax.legend(loc='upper right',frameon=False,fontsize=8);fig.tight_layout()
fig.savefig(OUT/'dev_vs_holdout.png',dpi=220,bbox_inches='tight');plt.close(fig)

v=DEV['visible_test_vs_official_score'];first=FINAL['first_attempt_official_pass'];fail=FINAL['visible_failed_starts'];fa=len(FINAL['officially_correct_visible_fail_ids']);n=FINAL['n_tasks']
conf=[[v['visible_pass_official_pass'],v['visible_pass_official_fail'],v['visible_fail_official_pass'],v['visible_fail_official_fail']],
      [first-fa,(n-fail)-(first-fa),fa,fail-fa]]
fig,ax=plt.subplots(figsize=(6.2,2.7));left=[0,0]
for k,(tag,col) in enumerate(zip(['Pass / pass','Pass / fail','Fail / pass','Fail / fail'],
                                   ['#536477','#b0bec5','#cb7b43','#8a9a8a'])):
 nums=[conf[i][k] for i in range(2)]
 ax.barh(['Dev (110)','Frozen (70)'],nums,left=left,height=.45,color=col,label=tag)
 for i,r in enumerate(nums):
  if r:ax.text(left[i]+r/2,i,str(r),ha='center',va='center',fontsize=8,
               color='white' if k in (0,2,3) else 'black')
 left=[left[i]+nums[i] for i in range(2)]
ax.invert_yaxis();ax.set_xlim(0,115);ax.set_xlabel('Number of C0 initial solutions')
ax.legend(loc='upper center',bbox_to_anchor=(.5,-.25),ncol=2,fontsize=8,frameon=False)
for s in ('top','right','left'):ax.spines[s].set_visible(False)
fig.tight_layout();fig.savefig(OUT/'visible_official_agreement.png',dpi=200,bbox_inches='tight');plt.close(fig)
print('Wrote', OUT/'dev_vs_holdout.png', OUT/'visible_official_agreement.png')

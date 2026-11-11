"""Create development-only research-paper figures using audited counts."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
D = json.loads((ROOT/'day5_audit.json').read_text())
FIG=ROOT/'figures'; FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                     'figure.dpi':180,'savefig.dpi':220})

models=['C0: base','C1: success-only','C2: recovery']
fixed=D['fixed_state_after_excluding_officially_correct_initial_states']['by_model']
vals=[fixed[c]['passed'] for c in ['c0','c1','c2']]
fig,ax=plt.subplots(figsize=(6.4,2.65))
bars=ax.bar(models, vals, width=.55)
ax.set_ylim(0,8.2);ax.set_ylabel('Officially correct repairs (of 29)')
for bar,v in zip(bars,vals):
    ax.text(bar.get_x()+bar.get_width()/2,v+.16,f'{v}/29 ({v/29:.1%})',ha='center',va='bottom',fontsize=9)
ax.set_title('Fixed-state recovery (development only)',loc='left',weight='bold',fontsize=11)
fig.tight_layout();fig.savefig(FIG/'fixed_recovery_audited.png',bbox_inches='tight');plt.close(fig)

fig,ax=plt.subplots(figsize=(6.4,2.65))
values=[D['feedback_ablation_c2']['detailed_pass'],D['feedback_ablation_c2']['vague_pass']]
bars=ax.bar(['Detailed test feedback','Vague retry instruction'],values,width=.56)
ax.set_ylim(0,7);ax.set_ylabel('Officially correct repairs (of 24)')
for bar,v in zip(bars,values):
    ax.text(bar.get_x()+bar.get_width()/2,v+.15,f'{v}/24 ({v/24:.1%})',ha='center',fontsize=9)
ax.set_title('C2 feedback ablation (development only)',loc='left',weight='bold',fontsize=11)
fig.tight_layout();fig.savefig(FIG/'feedback_ablation.png',bbox_inches='tight');plt.close(fig)

fig,ax=plt.subplots(figsize=(6.4,2.7))
cols=['C0: base','C1: success-only','C2: recovery']
final=[D['end_to_end'][c]['final_solved'] for c in ['c0','c1','c2']]
ever=[D['end_to_end'][c]['ever_solved_by_attempt']['3'] for c in ['c0','c1','c2']]
x=[0,1,2]; w=.34
ax.bar([q-w/2 for q in x],final,width=w,label='Final returned code')
ax.bar([q+w/2 for q in x],ever,width=w,label='Ever passed (oracle diagnostic)')
for i,v in enumerate(final): ax.text(i-w/2,v+.3,str(v),ha='center',fontsize=8)
for i,v in enumerate(ever): ax.text(i+w/2,v+.3,str(v),ha='center',fontsize=8)
ax.set_xticks(x,cols);ax.set_ylim(0,21);ax.set_ylabel('Successful tasks (of 30)')
ax.legend(loc='upper right',frameon=False,ncol=2,fontsize=8)
ax.set_title('End-to-end 3-attempt evaluation',loc='left',weight='bold',fontsize=11)
fig.tight_layout();fig.savefig(FIG/'end_to_end_final_vs_ever.png',bbox_inches='tight');plt.close(fig)
print('Generated',len(list(FIG.glob('*.png'))),'figures')

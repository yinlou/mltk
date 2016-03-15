ID0=6

all_nfeats=(600 400 300 200 150 100 70)
all_LR=(0.05 0.03 0.01 0.01 0.01 0.01 0.01)

DiagID=${ID0}
MGAM=100
# LGAM=0.01
# NFEATS=5

MGA2M=60
MaxPAIRS=300
LGA2M=0.01
NBAGS=70
 
# Gam package
export CLASSPATH=/home/sdubois/mltk/bin:$CLASSPATH

for ((idx=0;idx<${#all_nfeats[*]};idx++))

do
ID=${ID0}${idx}
LGAM=${all_LR[$idx]}
NFEATS=${all_nfeats[$idx]}

#... GAM ...#
echo "GAM${idx} - ${NFEATS} variables"

# build GAM
java mltk.predictor.gam.GAMLearner -b tr:3:100 -m $MGAM -l $LGAM -g c -e a -o gams/gam.model${ID} -f diags/gam${DiagID}.txt:N:${NFEATS} -w feats/gam${ID}

# test GAM
java mltk.predictor.evaluation.Evaluator -e a -m gams/gam.model${ID}

# compute residuals
java mltk.predictor.evaluation.Predictor -g c -m gams/gam.model${ID} -R res/residual${ID}.txt

# diagnostics
java mltk.predictor.gam.tool.Diagnostics -i gams/gam.model${ID} -o diags/gam${ID}.txt

#... FAST ...#
java mltk.predictor.gam.interaction.FAST -p 10 -R res/residual${ID}.txt -o ints/interactions${ID}.txt

echo "GA2M${idx}"
#... GA2M ...#
java mltk.predictor.gam.GA2MLearner -p $MaxPAIRS -m $MGA2M -b $NBAGS -l $LGA2M -g c -e a -i gams/gam.model${ID} -I ints/interactions${ID}.txt -o ga2ms/ga2m.model${ID}

# test
java mltk.predictor.evaluation.Evaluator -e a -m ga2ms/ga2m.model${ID}

#... DIAGS ...#
java mltk.predictor.gam.tool.Diagnostics -i ga2ms/ga2m.model${ID} -o diags/ga2m${ID}.txt

DiagID=$ID
done

ID0=66

all_npairs=(150 75 40 25 17 10)
all_LR=(0.01 0.01 0.01 0.01 0.01 0.01)

DiagID=${ID0}
MGA2M=150
MaxPAIRS=300
NBAGS=70
# LGA2M=0.01

# Gam package
export CLASSPATH=/home/sdubois/mltk/bin:$CLASSPATH

for ((idx=0;idx<${#all_npairs[*]};idx++))

do
ID=${ID0}${idx}
LGA2M=${all_LR[$idx]}
NPAIRS=${all_npairs[$idx]}

echo "GA2M${idx} - ${NPAIRS} interaction terms"
#... GA2M ...#
java mltk.predictor.gam.GA2MLearner -p $MaxPAIRS -m $MGA2M -b $NBAGS -l $LGA2M -g c -e a -i gams/gam.model${ID0} -I ints/interactions${ID0}.txt -o ga2ms/ga2m.model${ID} -f diags/ga2m${DiagID}.txt:N:${NPAIRS}

# test
java mltk.predictor.evaluation.Evaluator -e a -m ga2ms/ga2m.model${ID}

#... DIAGS ...#
java mltk.predictor.gam.tool.Diagnostics -i ga2ms/ga2m.model${ID} -o diags/ga2m${ID}.txt

DiagID=$ID
done

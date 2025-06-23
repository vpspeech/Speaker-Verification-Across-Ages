import random
with open("age1_age2_aged_tgntg_asv_score_hel_fem") as f11:
    trial1 = f11.readlines()

pp_0 = []
np_0 = []
pp_20 = []
np_20 = []
pp_40 = []
np_40 = []

for line in trial1:
    k1 = line.strip().split()[0]
    k2 = line.strip().split()[1]
    k3 = float(line.strip().split()[2])
    print(k3)
    k4 = int(line.strip().split()[3])
    k5 = float(line.strip().split()[4])
    
    if k3 == 0.0:
      if k4==1: 
         pp_0.append(k5)
      elif k4==0:
         np_0.append(k5) 

    elif k3 == 20.0:
      if k4==1: 
         pp_20.append(k5)
      elif k4==0:
         np_20.append(k5)    

    elif k3 == 40.0:
      if k4==1: 
         pp_40.append(k5)
      elif k4==0:
         np_40.append(k5)



def EER(positive_scores, negative_scores):
    # Computing candidate thresholds
    thresholds, _ = torch.sort(torch.cat([positive_scores, negative_scores]))
    thresholds = torch.unique(thresholds)
    # Adding intermediate thresholds
    interm_thresholds = (thresholds[0:-1] + thresholds[1:]) / 2
    thresholds, _ = torch.sort(torch.cat([thresholds, interm_thresholds]))
    # Variable to store the min FRR, min FAR and their corresponding index
    min_index = 0
    final_FRR = 0
    final_FAR = 0
    for i, cur_thresh in enumerate(thresholds):
        pos_scores_threshold = positive_scores <= cur_thresh
        FRR = (pos_scores_threshold.sum(0)).float() / positive_scores.shape[0]
        del pos_scores_threshold
        neg_scores_threshold = negative_scores > cur_thresh
        FAR = (neg_scores_threshold.sum(0)).float() / negative_scores.shape[0]
        del neg_scores_threshold
        # Finding the threshold for EER
        if (FAR - FRR).abs().item() < abs(final_FAR - final_FRR) or i == 0:
            min_index = i
            final_FRR = FRR.item()
            final_FAR = FAR.item()
    # It is possible that eer != fpr != fnr. We return (FAR  + FRR) / 2 as EER.
    EER = (final_FAR + final_FRR) / 2
    return float(EER), float(thresholds[min_index])

def minDCF(
    positive_scores, negative_scores, c_miss=1.0, c_fa=1.0, p_target=0.01
):
    # Computing candidate thresholds
    thresholds, _ = torch.sort(torch.cat([positive_scores, negative_scores]))
    thresholds = torch.unique(thresholds)
    # Adding intermediate thresholds
    interm_thresholds = (thresholds[0:-1] + thresholds[1:]) / 2
    thresholds, _ = torch.sort(torch.cat([thresholds, interm_thresholds]))
    # Computing False Rejection Rate (miss detection)
    positive_scores = torch.cat(
        len(thresholds) * [positive_scores.unsqueeze(0)]
    )
    pos_scores_threshold = positive_scores.transpose(0, 1) <= thresholds
    p_miss = (pos_scores_threshold.sum(0)).float() / positive_scores.shape[1]
    del positive_scores
    del pos_scores_threshold
    # Computing False Acceptance Rate (false alarm)
    negative_scores = torch.cat(
        len(thresholds) * [negative_scores.unsqueeze(0)]
    )
    neg_scores_threshold = negative_scores.transpose(0, 1) > thresholds
    p_fa = (neg_scores_threshold.sum(0)).float() / negative_scores.shape[1]
    del negative_scores
    del neg_scores_threshold
    c_det = c_miss * p_miss * p_target + c_fa * p_fa * (1 - p_target)
    c_min, min_index = torch.min(c_det, dim=0)
    return float(c_min), float(thresholds[min_index])

import torch
pp_0 = torch.tensor(pp_0)
np_0 = torch.tensor(np_0)
eer_0 = EER(pp_0,np_0)
mdcf_0 = minDCF(pp_0,np_0)

pp_20 = torch.tensor(pp_20)
np_20 = torch.tensor(np_20)
eer_20 = EER(pp_20,np_20)
mdcf_20 = minDCF(pp_20,np_20)

pp_40 = torch.tensor(pp_40)
np_40 = torch.tensor(np_40)
eer_40 = EER(pp_40,np_40)
mdcf_40 = minDCF(pp_40,np_40)


print("Age Difference =0, Female, EER :",eer_0[0]*100,"mDCF:", mdcf_0[0]*100)
print("Age Difference =20, Female,  EER :",eer_20[0]*100,"mDCF:", mdcf_20[0]*100)
print("Age Difference =40, Female, EER :",eer_40[0]*100,"mDCF:", mdcf_40[0]*100)




file1 = open('age1_age2_aged_tg_ntg_score_female_for_spyder','r')#'lmre_fem_aged_score_spkid_tgntg_gend_d0_diff_session','r')#'lmre_fem_aged_score_spkid_tgntg_gend', 'r')
Lines = file1.readlines()
 
df=[]
count = 0
# Strips the newline character
for line in Lines:
    k1 = line.strip().split()[0]
    k2 = line.strip().split()[1]
    k3 = line.strip().split()[2]
    k4 = line.strip().split()[3]
    k5 = line.strip().split()[4]
    
    k= [float(k1), float(k2), float(k3),  int (k4), float(k5) + 0.4]
    df.append(k)
    
df1 = pd.DataFrame(df, columns=['age1','age2','aged','tgntg','asvscore'])

#rp.summary_cont(df1.groupby(["treatment", "sex"])["weight"])

#boxplot = df1.boxplot(["asvscore"], by = ["aged"],
#                     figsize = (16, 9),
#                     showmeans = True,
#                     notch = True)

#boxplot.set_xlabel("Age Difference")
#boxplot.set_ylabel("Target ASVScore")

#boxplot.figure.savefig("boxplot_tmp.png")

df11=df1[df1['tgntg']==1]
df12=df1[df1['tgntg']==0]
#df111=df11[df1['session']==1]
rp.codebook(df1)
#rp.summary_cont(df1.groupby(["treatment", "sex"])["weight"])

boxplot = df1.boxplot(["asvscore"], by = ["aged"],
                     figsize = (16, 9),
                     showmeans = True,
                     notch = True)

boxplot.set_xlabel("Age Difference")
boxplot.set_ylabel("Target ASVScore")

boxplot.figure.savefig("boxplot_tmp.png")



model_tg = smf.mixedlm("asvscore ~ aged",
                    df11,
                    groups=df11["aged"]).fit()

model_tg.summary()

model_ntg = smf.mixedlm("asvscore ~ aged",
                    df12,
                    groups=df12["aged"]).fit()

model_ntg.summary()


from matplotlib import figure

fig = figure.Figure(figsize = (20, 12))
ax = sns.distplot(model_tg.fittedvalues, hist = False,kde_kws = {"shade" : True, "lw": 1},color='green')
#ax = sns.distplot(model_ntg.fittedvalues, hist = False,kde_kws = {"shade" : True, "lw": 1},label='Non-target Scores')
#ax.set_title(" Target Scores from LMRE (Male)")
ax.set_xlabel("ASV_Score_LME",fontweight='bold', fontsize=18)
ax.set_ylabel("Score Density",fontweight='bold',fontsize=18)
plt.rc('xtick', labelsize=16)    # fontsize of the tick labels
plt.rc('ytick', labelsize=16)


fig = figure.Figure(figsize = (20, 12))
#ax = sns.distplot(model_tg.fittedvalues, hist = False,kde_kws = {"shade" : True, "lw": 1},label='Target Scores')
ax = sns.distplot(model_ntg.fittedvalues, hist = False,kde_kws = {"shade" : True, "lw": 1},color='orange')
#ax.set_title(" Target Scores from LMRE (Male)")
ax.set_xlabel("ASV_Score_LME",fontweight='bold', fontsize=18)
ax.set_ylabel("Score Density",fontweight='bold',fontsize=18)
plt.rc('xtick', labelsize=16)    # fontsize of the tick labels
plt.rc('ytick', labelsize=16)





model = smf.mixedlm("asvscore ~ aged",
                    df1,
                    groups=df1["aged"]).fit()

model.summary()


from matplotlib import figure

fig = figure.Figure(figsize = (20, 12))

ax = sns.distplot(model.fittedvalues, hist = False)

#ax.set_title(" Target Scores from LMRE (Male)")
ax.set_xlabel("ASV_Score_LMRE")



########################################################




file1 = open('age1_age2_aged_tgntg_asv_score_hel_fem','r')#'lmre_fem_aged_score_spkid_tgntg_gend_d0_diff_session','r')#'lmre_fem_aged_score_spkid_tgntg_gend', 'r')
Lines = file1.readlines()
 
df=[]
count = 0
# Strips the newline character
for line in Lines:
    k1 = line.strip().split()[0]
    k2 = line.strip().split()[1]
    k3 = line.strip().split()[2]
    k4 = line.strip().split()[6]
    
    k= [float(k1), float(k2), int(k3),  int (k4)]
    df.append(k)
    
df1 = pd.DataFrame(df, columns=['aged','asvscore','tgntg','session'])
df11=df1[df1['tgntg']==1]
df12=df1[df1['tgntg']==0]
df111=df11[df1['session']==1]
rp.codebook(df1)
#rp.summary_cont(df1.groupby(["treatment", "sex"])["weight"])

boxplot = df1.boxplot(["asvscore"], by = ["aged"],
                     figsize = (16, 9),
                     showmeans = True,
                     notch = True)

boxplot.set_xlabel("Age Difference")
boxplot.set_ylabel("Target ASVScore")

boxplot.figure.savefig("boxplot_tmp.png")



model_tg = smf.mixedlm("asvscore ~ aged",
                    df11,
                    groups=df11["aged"]).fit()

model_tg.summary()

model_tg1 = smf.mixedlm("asvscore ~ aged",
                    df111,
                    groups=df111["aged"]).fit()

model_tg1.summary()

model_ntg = smf.mixedlm("asvscore ~ aged",
                    df12,
                    groups=df12["aged"]).fit()

model_ntg.summary()


from matplotlib import figure

fig = figure.Figure(figsize = (20, 12))
ax = sns.distplot(model_tg.fittedvalues, hist = False,kde_kws = {"shade" : True, "lw": 1},color='green')
#ax = sns.distplot(model_ntg.fittedvalues, hist = False,kde_kws = {"shade" : True, "lw": 1},label='Non-target Scores')
#ax.set_title(" Target Scores from LMRE (Male)")
ax.set_xlabel("ASV_Score_LME",fontweight='bold', fontsize=18)
ax.set_ylabel("Score Density",fontweight='bold',fontsize=18)
plt.rc('xtick', labelsize=16)    # fontsize of the tick labels
plt.rc('ytick', labelsize=16)

fig = figure.Figure(figsize = (20, 12))
ax = sns.distplot(model_tg1.fittedvalues, hist = False,kde_kws = {"shade" : True, "lw": 1},color='green')
#ax = sns.distplot(model_ntg.fittedvalues, hist = False,kde_kws = {"shade" : True, "lw": 1},label='Non-target Scores')
#ax.set_title(" Target Scores from LMRE (Male)")
ax.set_xlabel("ASV_Score_LME",fontweight='bold', fontsize=18)
ax.set_ylabel("Score Density",fontweight='bold',fontsize=18)
plt.rc('xtick', labelsize=16)    # fontsize of the tick labels
plt.rc('ytick', labelsize=16)


fig = figure.Figure(figsize = (20, 12))
#ax = sns.distplot(model_tg.fittedvalues, hist = False,kde_kws = {"shade" : True, "lw": 1},label='Target Scores')
ax = sns.distplot(model_ntg.fittedvalues, hist = False,kde_kws = {"shade" : True, "lw": 1},color='orange')
#ax.set_title(" Target Scores from LMRE (Male)")
ax.set_xlabel("ASV_Score_LME",fontweight='bold', fontsize=18)
ax.set_ylabel("Score Density",fontweight='bold',fontsize=18)
plt.rc('xtick', labelsize=16)    # fontsize of the tick labels
plt.rc('ytick', labelsize=16)

######################################################################
##########HEL PAPER##################################################
file1 = open('age1_age2_aged_tgntg_asv_score_hel_fem','r')#'lmre_fem_aged_score_spkid_tgntg_gend_d0_diff_session','r')#'lmre_fem_aged_score_spkid_tgntg_gend', 'r')
Lines = file1.readlines()
 
df=[]
count = 0
# Strips the newline character
for line in Lines:
    k1 = line.strip().split()[0]
    k2 = line.strip().split()[1]
    k3 = line.strip().split()[2]
    k4 = line.strip().split()[3]
    k5 = line.strip().split()[4]
    
    k= [float(k1), float(k2), float(k3),  int (k4), float(k5)]
    df.append(k)
    
df1 = pd.DataFrame(df, columns=['age1','age2','aged','tgntg','asvscore'])

df11=df1[df1['tgntg']==1]
df12=df1[df1['tgntg']==0]

rp.codebook(df1)
#rp.summary_cont(df1.groupby(["treatment", "sex"])["weight"])

boxplot = df1.boxplot(["asvscore"], by = ["aged"],
                     figsize = (16, 9),
                     showmeans = True,
                     notch = True)

boxplot.set_xlabel("Age Difference")
boxplot.set_ylabel("Target ASVScore")

boxplot.figure.savefig("boxplot_tmp.png")



model_tg = smf.mixedlm("asvscore ~ aged",
                    df11,
                    groups=df11["aged"]).fit()

model_tg.summary()


model_ntg = smf.mixedlm("asvscore ~ aged",
                    df12,
                    groups=df12["aged"]).fit()

model_ntg.summary()


from matplotlib import figure

fig = figure.Figure(figsize = (20, 12))
ax = sns.distplot(model_tg.fittedvalues, hist = False,kde_kws = {"shade" : True, "lw": 1},color='green')
#ax = sns.distplot(model_ntg.fittedvalues, hist = False,kde_kws = {"shade" : True, "lw": 1},label='Non-target Scores')
#ax.set_title(" Target Scores from LMRE (Male)")
ax.set_xlabel("ASV_Score_LME",fontweight='bold', fontsize=18)
ax.set_ylabel("Score Density",fontweight='bold',fontsize=18)
plt.rc('xtick', labelsize=16)    # fontsize of the tick labels
plt.rc('ytick', labelsize=16)


fig = figure.Figure(figsize = (20, 12))
#ax = sns.distplot(model_tg.fittedvalues, hist = False,kde_kws = {"shade" : True, "lw": 1},label='Target Scores')
ax = sns.distplot(model_ntg.fittedvalues -2.0, hist = False,kde_kws = {"shade" : True, "lw": 1},color='orange')
#ax.set_title(" Target Scores from LMRE (Male)")
ax.set_xlabel("ASV_Score_LME",fontweight='bold', fontsize=18)
ax.set_ylabel("Score Density",fontweight='bold',fontsize=18)
plt.rc('xtick', labelsize=16)    # fontsize of the tick labels
plt.rc('ytick', labelsize=16)


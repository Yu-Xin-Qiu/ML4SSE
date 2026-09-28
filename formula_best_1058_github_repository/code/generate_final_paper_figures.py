"""Generate the final manuscript/SI figures and cost-conductivity screening tables."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import joblib
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.base import clone
from pymatgen.core import Composition
from scipy.spatial.distance import cdist
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict

from matminer_composition_features import featurize_new_formulas

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data"/"modeling_matrix_matminer_188.csv"
COST_FILE=ROOT/"data"/"element_costs.xlsx"
MODEL=ROOT/"models"/"ten_model_comparison_bundle.joblib"
REPORT=ROOT/"reports"/"rf_rfe_10models"
PROVENANCE=ROOT/"reports"/"si_tables"/"SI_matminer_descriptor_provenance.csv"
MAIN=ROOT/"figures"/"final_main"
SI=ROOT/"figures"/"final_si"
TOP5_XY=ROOT/"figures"/"top5_xy_screening"
SOURCE=ROOT/"reports"/"final_figure_source_data"
TABLE=ROOT/"reports"/"final_si_tables"
for d in (MAIN,SI,TOP5_XY,SOURCE,TABLE): d.mkdir(parents=True,exist_ok=True)
CV=KFold(n_splits=5,shuffle=True,random_state=42)
COLORS=["#3C5488","#00A087","#F39B7F","#8491B4","#E64B35","#7E7E7E","#4DBBD5","#91D1C2","#DC0000","#B09C85"]
FOCUSED_CMAP=LinearSegmentedColormap.from_list("focused",["#2c3e8f","#2878b5","#35a6a0","#78c679","#f4e85c","#f6bd4b","#ef7d32","#d84a2f","#a71930"])
GRID_X_STEP=.5
GRID_Y_STEP=.25
HIGH_PLATEAU_MS_CM=14.5
HIGH_PLATEAU_LOG10=float(np.log10(HIGH_PLATEAU_MS_CM))

def style():
 mpl.rcParams.update({"font.family":"Arial","font.size":7.5,"axes.labelsize":8,"axes.titlesize":8.5,
 "xtick.labelsize":7,"ytick.labelsize":7,"axes.linewidth":0.8,"xtick.direction":"out","ytick.direction":"out",
 "pdf.fonttype":42,"ps.fonttype":42,"figure.facecolor":"white","axes.facecolor":"white"})

def boxed(ax):
 for s in ax.spines.values(): s.set_visible(True); s.set_linewidth(0.8)

def save(fig,folder,stem):
 fig.savefig(folder/f"{stem}.png",dpi=600,bbox_inches="tight",facecolor="white")
 fig.savefig(folder/f"{stem}.pdf",bbox_inches="tight",facecolor="white")
 plt.close(fig)

def figures_a_b(df):
 import figure_core as core
 core.nature_style(); core.figure_a_distribution(df); core.figure_b_periodic_table(df)
 for stem in ("a_conductivity_distribution","b_periodic_table_element_distribution"):
  for ext in ("png","pdf"):
   shutil.copy2(core.FIG_MAIN/f"{stem}.{ext}",MAIN/f"{stem}.{ext}")

def publish_figure1_aliases():
 mapping={"a_conductivity_distribution":"Figure1a_conductivity_distribution","b_periodic_table_element_distribution":"Figure1b_element_distribution","top10_selected_feature_correlations":"Figure1c_top10_correlations","model_performance_with_error_bars":"Figure1d_model_performance","final_extratrees_oof_scatter":"Figure1e_ExtraTrees_regression","top10_shap_scatter_and_importance":"Figure1f_SHAP","g_final_dopant_pair_violin":"Figure1g_dopant_violin","h_final_Se_Br_contour":"Figure1h_Se_Br_contour"}
 rows=[]
 for source,target in mapping.items():
  for ext in ("png","pdf"): shutil.copy2(MAIN/f"{source}.{ext}",MAIN/f"{target}.{ext}")
  rows.append({"panel":target[7],"file":target,"source_stem":source})
 pd.DataFrame(rows).to_csv(SOURCE/"Figure1_panel_manifest.csv",index=False)

def slug(x): return x.lower().replace(" ","_")

def load():
 df=pd.read_csv(DATA); bundle=joblib.load(MODEL)
 features=bundle["selected_features"]; y=df.target_log10_mS_cm.to_numpy(float)
 oof=pd.read_csv(REPORT/"ten_model_oof_predictions.csv")
 return df,bundle,features,y,oof

def predict_best(bundle,formulas):
 x=featurize_new_formulas(formulas,expected_columns=bundle["selected_features"]).fillna(0.0)
 best_name=bundle["model_order"][0]
 best_model=next(item["model"] for item in bundle["models"] if item["name"]==best_name)
 return x,best_model.predict(x.to_numpy(float))

def performance_bar(y,oof,bundle):
 rows=[]
 for fold,(_,test) in enumerate(CV.split(y),1):
  for name in bundle["model_order"]:
   pred=oof[f"pred_{slug(name)}_log10_mS_cm"].to_numpy(float)
   rows.append({"model":name,"fold":fold,"r2":r2_score(y[test],pred[test]),"mae_log10":mean_absolute_error(y[test],pred[test])})
 scores=pd.DataFrame(rows); scores.to_csv(SOURCE/"model_fold_performance.csv",index=False)
 summary=scores.groupby("model").agg(mean_r2=("r2","mean"),sd_r2=("r2","std"),mean_mae=("mae_log10","mean")).reset_index().sort_values("mean_r2")
 summary.to_csv(SOURCE/"model_performance_mean_sd.csv",index=False)
 fig,ax=plt.subplots(figsize=(6.4,3.45)); x=np.arange(len(summary))
 for i,(xi,val) in enumerate(zip(x,summary.mean_r2)):
  cmap=LinearSegmentedColormap.from_list(f"g{i}",["white",COLORS[i%len(COLORS)]])
  ax.imshow(np.linspace(0,1,256).reshape(-1,1),extent=(xi-.39,xi+.39,0,val),origin="lower",aspect="auto",cmap=cmap,zorder=2)
  ax.add_patch(plt.Rectangle((xi-.39,0),.78,val,fill=False,edgecolor=COLORS[i%len(COLORS)],lw=.8,zorder=3))
 ax.errorbar(x,summary.mean_r2,yerr=summary.sd_r2,fmt="none",ecolor="black",lw=.7,capsize=2.3,zorder=4)
 ax.set_xticks(x,summary.model,rotation=38,ha="right"); ax.set_ylabel(r"Five-fold $R^2$ (mean $\pm$ s.d.)"); ax.set_ylim(0,.9); boxed(ax); fig.tight_layout(); save(fig,MAIN,"model_performance_with_error_bars")

def best_model_scatter(y,oof,bundle):
 name=bundle["model_order"][0]; pred=oof[f"pred_{slug(name)}_log10_mS_cm"].to_numpy(float)
 r2=r2_score(y,pred); mae=mean_absolute_error(y,pred); rmse=mean_squared_error(y,pred)**.5
 pd.DataFrame({"observed_log10_mS_cm":y,"extratrees_oof_log10_mS_cm":pred,"model":name}).to_csv(SOURCE/"final_extratrees_oof_predictions.csv",index=False)
 lo=min(y.min(),pred.min())-.15; hi=max(y.max(),pred.max())+.15
 fig=plt.figure(figsize=(4.25,4.15)); gs=GridSpec(2,2,figure=fig,width_ratios=(4,1.05),height_ratios=(1.05,4),hspace=.04,wspace=.04)
 top=fig.add_subplot(gs[0,0]); ax=fig.add_subplot(gs[1,0],sharex=top); right=fig.add_subplot(gs[1,1],sharey=ax)
 ax.scatter(y,pred,s=10,alpha=.52,color=COLORS[0],edgecolor="none",rasterized=True); ax.plot([lo,hi],[lo,hi],color="black",lw=.8)
 ax.set(xlim=(lo,hi),ylim=(lo,hi),xlabel=r"Observed log$_{10}$ conductivity (mS cm$^{-1}$)",ylabel=r"ExtraTrees prediction log$_{10}$ conductivity (mS cm$^{-1}$)")
 ax.text(.04,.96,f"ExtraTrees\n$R^2$ = {r2:.3f}\nMAE = {mae:.3f}\nRMSE = {rmse:.3f}",transform=ax.transAxes,ha="left",va="top"); boxed(ax)
 sns.histplot(x=y,bins=28,stat="density",color=COLORS[0],alpha=.4,ax=top); sns.kdeplot(x=y,color=COLORS[0],lw=1,ax=top)
 sns.histplot(y=pred,bins=28,stat="density",color=COLORS[1],alpha=.4,ax=right); sns.kdeplot(y=pred,color=COLORS[1],lw=1,ax=right)
 top.tick_params(axis="x",labelbottom=False); top.tick_params(axis="y",left=False,labelleft=False); right.tick_params(axis="y",labelleft=False); right.tick_params(axis="x",bottom=False,labelbottom=False)
 for a in (top,right): a.set_xlabel(""); a.set_ylabel(""); boxed(a)
 fig.subplots_adjust(left=.17,right=.95,bottom=.14,top=.95); save(fig,MAIN,"final_extratrees_oof_scatter")

def all_model_scatter(y,oof,bundle):
 lo,hi=float(y.min()-.15),float(y.max()+.15); rows=[]
 fig,axes=plt.subplots(2,5,figsize=(13.2,5.3),sharex=True,sharey=True)
 for ax,name,color in zip(axes.flat,bundle["model_order"],COLORS):
  pred=oof[f"pred_{slug(name)}_log10_mS_cm"].to_numpy(float); r2=r2_score(y,pred); mae=mean_absolute_error(y,pred); rmse=mean_squared_error(y,pred)**.5
  rows.append({"model":name,"r2":r2,"mae_log10":mae,"rmse_log10":rmse})
  ax.scatter(y,pred,s=7,alpha=.45,color=color,edgecolor="none",rasterized=True); ax.plot([lo,hi],[lo,hi],color="black",lw=.7)
  ax.set_xlim(lo,hi); ax.set_ylim(lo,hi); ax.set_title(f"{name}\n$R^2$={r2:.3f}; MAE={mae:.3f}",fontsize=7.5); boxed(ax)
 for ax in axes[-1,:]: ax.set_xlabel(r"Observed log$_{10}$ conductivity")
 for ax in axes[:,0]: ax.set_ylabel(r"OOF-predicted log$_{10}$ conductivity")
 pd.DataFrame(rows).to_csv(SOURCE/"all_model_oof_scatter_metrics.csv",index=False)
 fig.tight_layout(); save(fig,SI,"all_ten_model_oof_regression_scatter")

def residual_diagnostics(df,y):
 pred=pd.read_csv(SOURCE/"final_extratrees_oof_predictions.csv")["extratrees_oof_log10_mS_cm"].to_numpy(float); residual=y-pred
 out=df[["source_database","source_doi","formula_raw","formula_clean","composition_key"]].copy(); out["observed_log10_mS_cm"]=y; out["predicted_log10_mS_cm"]=pred; out["residual_log10"]=residual; out["absolute_error_log10"]=np.abs(residual); out.sort_values("absolute_error_log10",ascending=False).to_csv(SOURCE/"extratrees_oof_residuals_and_top_errors.csv",index=False,encoding="utf-8-sig")
 source_rows=[]
 for source,g in out.groupby("source_database"):
  source_rows.append({"source_database":source,"n":len(g),"r2":r2_score(g.observed_log10_mS_cm,g.predicted_log10_mS_cm),"mae_log10":mean_absolute_error(g.observed_log10_mS_cm,g.predicted_log10_mS_cm),"rmse_log10":mean_squared_error(g.observed_log10_mS_cm,g.predicted_log10_mS_cm)**.5})
 pd.DataFrame(source_rows).to_csv(TABLE/"SI_source_specific_oof_performance.csv",index=False,encoding="utf-8-sig")
 fig,(ax,hist)=plt.subplots(1,2,figsize=(7.2,3.05),gridspec_kw={"width_ratios":[1.45,1]})
 ax.scatter(pred,residual,s=8,alpha=.45,color=COLORS[0],edgecolor="none",rasterized=True); ax.axhline(0,color="black",lw=.75); ax.set_xlabel(r"ExtraTrees-predicted log$_{10}$ conductivity"); ax.set_ylabel("OOF residual (observed - predicted)"); boxed(ax)
 hist.hist(residual,bins=32,color=COLORS[1],alpha=.75,edgecolor="white",linewidth=.4); hist.axvline(0,color="black",lw=.75); hist.set_xlabel("OOF residual"); hist.set_ylabel("Count"); boxed(hist)
 fig.tight_layout(); save(fig,SI,"extratrees_residual_diagnostics")

def doi_group_sensitivity(df,y,bundle,oof):
 doi=df["source_doi"].fillna("").astype(str).str.strip().str.lower()
 fallback=df.get("source_ref",pd.Series([""]*len(df))).fillna("").astype(str).str.strip().str.lower()
 groups=np.array([d if d and d!="nan" else (f"ref:{r}" if r and r!="nan" else f"row:{i}") for i,(d,r) in enumerate(zip(doi,fallback))])
 best_name=bundle["model_order"][0]; best_model=next(item["model"] for item in bundle["models"] if item["name"]==best_name); x=df[bundle["selected_features"]].to_numpy(float)
 grouped=cross_val_predict(clone(best_model),x,y,cv=GroupKFold(n_splits=5),groups=groups,n_jobs=1)
 random=oof[f"pred_{slug(best_name)}_log10_mS_cm"].to_numpy(float)
 rows=[]
 for scheme,pred in [("Random shuffled KFold",random),("DOI-grouped KFold sensitivity",grouped)]:
  rows.append({"validation":scheme,"model":best_name,"r2":r2_score(y,pred),"mae_log10":mean_absolute_error(y,pred),"rmse_log10":mean_squared_error(y,pred)**.5,"n_groups":len(np.unique(groups))})
 pd.DataFrame(rows).to_csv(TABLE/"SI_random_vs_DOI_grouped_validation.csv",index=False,encoding="utf-8-sig")
 pd.DataFrame({"formula":df["formula_clean"],"source_doi":df["source_doi"],"group":groups,"observed_log10_mS_cm":y,"grouped_oof_prediction":grouped}).to_csv(SOURCE/"doi_grouped_oof_predictions.csv",index=False,encoding="utf-8-sig")
 fig,ax=plt.subplots(figsize=(4.2,3.1)); xpos=np.arange(2); vals=[r["r2"] for r in rows]; ax.bar(xpos,vals,color=[COLORS[0],COLORS[4]],edgecolor="black",linewidth=.65); ax.set_xticks(xpos,["Random five-fold","DOI-grouped\nsensitivity"]); ax.set_ylabel(r"OOF $R^2$"); ax.set_ylim(min(0, min(vals)-.08),max(vals)+.08); [ax.text(i,v+.012,f"{v:.3f}",ha="center",va="bottom",fontsize=7) for i,v in enumerate(vals)]; boxed(ax); fig.tight_layout(); save(fig,SI,"random_vs_doi_grouped_validation")

def final_model_shap(df,bundle):
 rng=np.random.default_rng(42); idx=np.sort(rng.choice(len(df),size=min(500,len(df)),replace=False)); features=bundle["selected_features"]
 values=df.iloc[idx][features].reset_index(drop=True); total=np.zeros((len(idx),len(features)))
 lookup={item["name"]:item["model"] for item in bundle["models"]}
 # Use the best validated individual model for a direct, model-exact SHAP
 # explanation.  The heterogeneous Ridge stack contains non-tree members and
 # does not admit a single exact TreeSHAP decomposition.
 shap_model_name=bundle["model_order"][0]
 expl=shap.TreeExplainer(lookup[shap_model_name]); total=np.asarray(expl.shap_values(values.to_numpy(float),check_additivity=False))
 shapdf=pd.DataFrame(total,columns=features); importance=shapdf.abs().mean().sort_values(ascending=False); top10=importance.head(10).index.tolist()
 pd.concat({"feature_value":values,"shap_value":shapdf},axis=1).to_csv(SOURCE/"best_model_shap_values.csv",index=False)
 importance.rename("mean_abs_shap").to_csv(SOURCE/"best_model_shap_importance.csv")
 def short_name(name):
  return (name.replace("matminer_ElementProperty__MagpieData ", "Magpie ")
          .replace("matminer_ElementFraction__", "Element fraction: ")
          .replace("matminer_Stoichiometry__", "Stoichiometry: ")
          .replace("matminer_IonProperty__", "Ion property: ")
          .replace("matminer_ValenceOrbital__", "Valence orbital: ")
          .replace("matminer_BandCenter__", "Band center: ")
          .replace("matminer_TMetalFraction__", "TM fraction: "))
 fig,(ax,bar)=plt.subplots(2,1,figsize=(6.4,6.2),gridspec_kw={"height_ratios":[2.2,1.15],"hspace":.38}); cmap=mpl.colormaps["coolwarm"]
 for row,f in enumerate(top10[::-1]):
  sv=shapdf[f].to_numpy(); fv=values[f].to_numpy(); ranks=pd.Series(fv).rank(pct=True).to_numpy(); jitter=rng.normal(0,.09,len(sv))
  ax.scatter(sv,row+jitter,c=ranks,cmap=cmap,vmin=0,vmax=1,s=7,alpha=.7,edgecolor="none",rasterized=True)
 ax.axvline(0,color="#777",lw=.6); ax.set_yticks(range(10),[short_name(f) for f in top10[::-1]]); ax.set_xlabel(r"SHAP value (log$_{10}$ mS cm$^{-1}$)"); boxed(ax)
 sm=mpl.cm.ScalarMappable(cmap=cmap,norm=mpl.colors.Normalize(0,1)); cb=fig.colorbar(sm,ax=ax,fraction=.035,pad=.02); cb.set_ticks([0,1],labels=["Low","High"]); cb.set_label("Feature value")
 vals=importance.loc[top10].sort_values(); ypos=np.arange(10); grad=mpl.colormaps["YlOrRd"](np.linspace(.28,.9,10))
 bar.barh(ypos,vals.values,color=grad,edgecolor="#333333",linewidth=.55); bar.set_yticks(ypos,[short_name(f) for f in vals.index]); bar.set_xlabel("Mean |SHAP value|"); boxed(bar)
 fig.suptitle(f"SHAP analysis: {shap_model_name}",y=.995,fontsize=8.5); fig.subplots_adjust(left=.32,right=.93,bottom=.08,top=.96); save(fig,MAIN,"top10_shap_scatter_and_importance")
 return top10,importance

def correlations(df,features,top10):
 all_features=[c for c in df.columns if c.startswith("matminer_") and df[c].nunique()>1]
 for cols,stem,title,size,tick in [(all_features,"correlation_before_188","188 descriptors before RF-RFE",(13.5,12.5),2),(features,f"correlation_after_{len(features)}",f"{len(features)} descriptors after RF-RFECV",(11.5,10.5),2.5)]:
  corr=df[cols].corr(); corr.to_csv(SOURCE/f"{stem}_matrix.csv")
  fig,ax=plt.subplots(figsize=size); im=ax.imshow(corr,cmap="vlag",vmin=-1,vmax=1,aspect="equal",interpolation="nearest"); ax.set_xticks(range(len(cols)),cols,rotation=-45,ha="left",fontsize=tick); ax.set_yticks(range(len(cols)),cols,fontsize=tick); ax.tick_params(length=0); ax.set_title(title); boxed(ax); cb=fig.colorbar(im,ax=ax,fraction=.028,pad=.02); cb.set_label("Pearson correlation coefficient"); fig.tight_layout(); save(fig,SI,stem)
 corr=df[top10].corr(); corr.to_csv(SOURCE/"top10_selected_feature_correlation_matrix.csv")
 def compact(name):
  return (name.replace("matminer_ElementProperty__MagpieData ","Magpie ").replace("matminer_ElementFraction__","Element fraction: ").replace("matminer_Stoichiometry__","Stoichiometry: ").replace("matminer_IonProperty__","Ion property: ").replace("matminer_ValenceOrbital__","Valence orbital: "))
 labels=[compact(c) for c in top10]; display=corr.copy(); display.index=labels; display.columns=labels
 fig,ax=plt.subplots(figsize=(6.2,5.5)); sns.heatmap(display,cmap="vlag",vmin=-1,vmax=1,center=0,square=True,annot=True,fmt=".2f",annot_kws={"fontsize":5.5},cbar_kws={"label":"Pearson correlation coefficient","shrink":.75},ax=ax); ax.tick_params(axis="x",rotation=-45,length=0,labelsize=6); [t.set_ha("left") for t in ax.get_xticklabels()]; ax.tick_params(axis="y",rotation=0,length=0,labelsize=6); boxed(ax); fig.tight_layout(); save(fig,MAIN,"top10_selected_feature_correlations")

def hyperparameter_heatmaps():
 search=pd.read_csv(REPORT/"ten_model_all_search_results.csv"); configs={
 "Ridge":("param_ridge__alpha",None),"ElasticNet":("param_elasticnet__alpha","param_elasticnet__l1_ratio"),"KNN":("param_kneighborsregressor__n_neighbors","param_kneighborsregressor__p"),"SVR":("param_svr__C","param_svr__gamma"),"RandomForest":("param_max_features","param_min_samples_leaf"),"ExtraTrees":("param_max_features","param_min_samples_leaf"),"AdaBoost":("param_n_estimators","param_learning_rate"),"GradientBoosting":("param_learning_rate","param_max_depth"),"XGBoost":("param_learning_rate","param_max_depth"),"LightGBM":("param_learning_rate","param_num_leaves")}
 fig,axes=plt.subplots(2,5,figsize=(18,6.7)); sources=[]
 for ax,(name,(xc,yc)) in zip(axes.flat,configs.items()):
  sub=search[search.model.eq(name)].copy(); x=pd.to_numeric(sub[xc],errors="coerce")
  def exact(v): return v.map(lambda z:f"{z:g}" if pd.notna(z) else "NA")
  sub["x_plot"]=exact(x)
  if yc is None: sub["y_plot"]="all"
  else: sub["y_plot"]=exact(pd.to_numeric(sub[yc],errors="coerce"))
  pivot=sub.pivot_table(index="y_plot",columns="x_plot",values="mean_test_score",aggfunc="max")
  local_min=float(np.nanmin(pivot.to_numpy(float))); local_max=float(np.nanmax(pivot.to_numpy(float))); pad=max((local_max-local_min)*.04,.002)
  hm=sns.heatmap(pivot,cmap=FOCUSED_CMAP,vmin=local_min-pad,vmax=local_max+pad,annot=True,fmt=".3f",annot_kws={"fontsize":4.7},linewidths=.35,linecolor="white",cbar=True,cbar_kws={"label":r"Mean CV $R^2$","shrink":.72,"pad":.025},ax=ax)
  hm.collections[0].colorbar.ax.tick_params(labelsize=4.8,length=2); hm.collections[0].colorbar.set_label(r"Mean CV $R^2$",fontsize=5.5)
  ax.set_title(name); ax.set_xlabel(xc.replace("param_","").replace("__","\n"),fontsize=6.5); ax.set_ylabel("" if yc is None else yc.replace("param_","").replace("__","\n"),fontsize=6.5); ax.tick_params(axis="x",rotation=-60,labelsize=4.7); ax.tick_params(axis="y",rotation=0,labelsize=5); boxed(ax)
  long=pivot.stack(future_stack=True).rename("best_mean_cv_r2").reset_index(); long.insert(0,"model",name); sources.append(long)
 pd.concat(sources,ignore_index=True).to_csv(SOURCE/"hyperparameter_heatmap_source.csv",index=False); fig.tight_layout(); save(fig,SI,"ten_model_hyperparameter_search_heatmaps")

def top5_xy_screening(bundle,prices):
 xs=np.arange(.25,4.501,.25); ys=np.arange(.25,1.501,.25)
 grid=pd.DataFrame([{"x":float(x),"y":float(y),"formula":formula_pair("Se","Br",float(x),float(y))} for y in ys for x in xs])
 feat=featurize_new_formulas(grid.formula.tolist(),expected_columns=bundle["selected_features"]).fillna(0.0)
 lookup={item["name"]:item["model"] for item in bundle["models"]}; top5=bundle["model_order"][:5]
 experiments=pd.DataFrame([{"id":"A","x":2.,"y":.5},{"id":"B","x":2.,"y":.75},{"id":"C","x":1.,"y":.5},{"id":"D","x":1.,"y":.75}])
 cmap=LinearSegmentedColormap.from_list("focused_original",["#2c3e8f","#2878b5","#35a6a0","#78c679","#f4e85c","#f4a43a","#b8322a"])
 combined=plt.figure(figsize=(11.8,6.7)); axes=combined.subplots(2,3); axes=axes.flat; markers=["o","s","^","D"]
 rows=[]; scale_rows=[]
 for panel,(name,ax) in enumerate(zip(top5,axes)):
  pred=lookup[name].predict(feat.to_numpy(float)); grid[f"pred_{slug(name)}_log10_mS_cm"]=pred
  z=pred.reshape(len(ys),len(xs))
  levels=np.linspace(float(z.min()),float(z.max()),18)
  cf=ax.contourf(xs,ys,z,levels=levels,cmap=cmap,extend="both")
  lines=ax.contour(xs,ys,z,levels=levels[::4],colors="#4f4300",linewidths=.35,alpha=.65); ax.clabel(lines,fontsize=4.5,fmt="%.2f")
  for marker,(_,r) in zip(markers,experiments.iterrows()): ax.scatter(r.x,r.y,s=32,marker=marker,facecolor="white",edgecolor="black",linewidth=.8,zorder=5); ax.annotate(r.id,(r.x,r.y),xytext=(4,4),textcoords="offset points",fontsize=6,weight="bold")
  cb=combined.colorbar(cf,ax=ax,pad=.02,fraction=.046); cb.set_label(r"Predicted log$_{10}$ conductivity",fontsize=6); cb.ax.tick_params(labelsize=5)
  ax.set_title(name); ax.set_xlabel(r"Se content, $x$"); ax.set_ylabel(r"Br content, $y$"); ax.set_xlim(.25,4.5); ax.set_ylim(.23,1.52); boxed(ax)
  # Also publish a full-size standalone panel with its own color scale.
  single,sax=plt.subplots(figsize=(4.7,3.65)); scf=sax.contourf(xs,ys,z,levels=levels,cmap=cmap,extend="both"); sl=sax.contour(xs,ys,z,levels=levels[::4],colors="#4f4300",linewidths=.4,alpha=.65); sax.clabel(sl,fontsize=5,fmt="%.2f")
  for marker,(_,r) in zip(markers,experiments.iterrows()): sax.scatter(r.x,r.y,s=42,marker=marker,facecolor="white",edgecolor="black",linewidth=.9,zorder=5); sax.annotate(r.id,(r.x,r.y),xytext=(4,4),textcoords="offset points",fontsize=6.5,weight="bold")
  scb=single.colorbar(scf,ax=sax,pad=.02); scb.set_label(r"Predicted log$_{10}$ conductivity (mS cm$^{-1}$)"); sax.set(title=name,xlabel=r"Se content, $x$",ylabel=r"Br content, $y$"); sax.set_xlim(.25,4.5); sax.set_ylim(.23,1.52); boxed(sax); single.tight_layout(); save(single,TOP5_XY,f"{panel+1}_{slug(name)}_Se_Br_xy")
  scale_rows.extend({"model":name,"boundary_index":i,"log10_mS_cm_boundary":v} for i,v in enumerate(levels))
  rows.extend({"model":name,"x":r.x,"y":r.y,"formula":r.formula,"predicted_log10_mS_cm":p,"predicted_mS_cm":10**p,"elemental_cost_USD_per_kg":elemental_cost(r.formula,prices)} for (_,r),p in zip(grid.iterrows(),pred))
 axes[5].axis("off"); combined.subplots_adjust(left=.07,right=.98,bottom=.10,top=.94,wspace=.34,hspace=.34); save(combined,SI,"top5_model_Se_Br_xy_screening")
 all_predictions=pd.DataFrame(rows); all_predictions.to_csv(SOURCE/"top5_model_Se_Br_xy_predictions.csv",index=False,encoding="utf-8-sig")
 pd.DataFrame(scale_rows).to_csv(SOURCE/"top5_model_Se_Br_linear_color_boundaries.csv",index=False,encoding="utf-8-sig")
 mean_grid=all_predictions.groupby(["x","y","formula"],as_index=False).agg(mean_log10_mS_cm=("predicted_log10_mS_cm","mean"),sd_log10_mS_cm=("predicted_log10_mS_cm","std"),geometric_mean_mS_cm=("predicted_mS_cm",lambda s:10**np.log10(s).mean()),arithmetic_mean_mS_cm=("predicted_mS_cm","mean"),elemental_cost_USD_per_kg=("elemental_cost_USD_per_kg","first"))
 mean_grid.to_csv(SOURCE/"top5_model_mean_Se_Br_xy_predictions.csv",index=False,encoding="utf-8-sig")
 mz=mean_grid.mean_log10_mS_cm.to_numpy().reshape(len(ys),len(xs)); mlevels=np.unique(np.quantile(mz,np.linspace(0,1,33)))
 if len(mlevels)<12: mlevels=np.linspace(float(mz.min()),float(mz.max()),25)
 mnorm=BoundaryNorm(mlevels,cmap.N,clip=False); mline=np.unique(np.quantile(mz,[.08,.20,.35,.50,.65,.78,.88,.95])); mticks=mlevels[np.unique(np.linspace(0,len(mlevels)-1,6).round().astype(int))]
 mfig,maxis=plt.subplots(figsize=(5.0,3.75)); mcf=maxis.contourf(xs,ys,mz,levels=mlevels,cmap=cmap,norm=mnorm,extend="both"); ml=maxis.contour(xs,ys,mz,levels=mline,colors="#4f4300",linewidths=.45,alpha=.7); maxis.clabel(ml,fontsize=5.2,fmt="%.2f")
 for marker,(_,r) in zip(markers,experiments.iterrows()): maxis.scatter(r.x,r.y,s=46,marker=marker,facecolor="white",edgecolor="black",linewidth=.9,zorder=5); maxis.annotate(r.id,(r.x,r.y),xytext=(4,4),textcoords="offset points",fontsize=7,weight="bold")
 mcb=mfig.colorbar(mcf,ax=maxis,pad=.02,ticks=mticks,format="%.2f"); mcb.set_label(r"TOP5 mean predicted log$_{10}$ conductivity (mS cm$^{-1}$)"); maxis.set(title="TOP5 model mean",xlabel=r"Se content, $x$",ylabel=r"Br content, $y$"); maxis.set_xlim(.25,4.5); maxis.set_ylim(.23,1.52); boxed(maxis); mfig.tight_layout(); save(mfig,TOP5_XY,"0_top5_mean_Se_Br_xy")
 for ext in ("png","pdf"): shutil.copy2(TOP5_XY/f"0_top5_mean_Se_Br_xy.{ext}",SI/f"top5_model_mean_Se_Br_xy.{ext}")
 candidate_keys={(2.,.5):"A",(2.,.75):"B",(1.,.5):"C",(1.,.75):"D"}; candidate_summary=all_predictions[all_predictions.apply(lambda r:(r.x,r.y) in candidate_keys,axis=1)].copy(); candidate_summary["candidate_id"]=[candidate_keys[(r.x,r.y)] for _,r in candidate_summary.iterrows()]; candidate_summary.sort_values(["candidate_id","model"]).to_csv(SOURCE/"top5_model_four_experimental_candidates.csv",index=False,encoding="utf-8-sig")
 candidate_wide=candidate_summary.pivot(index=["candidate_id","formula","x","y"],columns="model",values="predicted_mS_cm").reset_index()
 candidate_stats=candidate_summary.groupby(["candidate_id","formula","x","y"],as_index=False).agg(mean_log10_mS_cm=("predicted_log10_mS_cm","mean"),sd_log10_mS_cm=("predicted_log10_mS_cm","std"),geometric_mean_mS_cm=("predicted_mS_cm",lambda s:10**np.log10(s).mean()),arithmetic_mean_mS_cm=("predicted_mS_cm","mean"))
 candidate_wide=candidate_wide.merge(candidate_stats,on=["candidate_id","formula","x","y"]).sort_values("candidate_id"); candidate_wide.to_csv(SOURCE/"top5_model_four_experimental_candidates_wide.csv",index=False,encoding="utf-8-sig")

def descriptor_table(features,importance):
 prov=pd.read_csv(PROVENANCE); rank=pd.read_csv(REPORT/"rfecv_feature_ranking.csv"); out=prov.merge(rank,on="descriptor",how="left"); out["selected_final"]=out.descriptor.isin(features); out["mean_abs_shap_best_model"]=out.descriptor.map(importance).fillna(0); out["shap_rank"]=out["mean_abs_shap_best_model"].rank(ascending=False,method="min").astype(int); out.to_csv(TABLE/"SI_descriptor_dictionary_RF_RFE_final.csv",index=False,encoding="utf-8-sig")

def formula_pair(chalcogen,halogen,x,y):
 parts=["Li5.5","P",f"S{4.5-x:.4g}"]; parts += [] if x<1e-12 else [f"{chalcogen}{x:.4g}"]; parts += [] if 1.5-y<1e-12 else [f"Cl{1.5-y:.4g}"]; parts += [] if y<1e-12 else [f"{halogen}{y:.4g}"]; return "".join(parts)

def cost_map():
 c=pd.read_excel(COST_FILE); return dict(zip(c.formula.astype(str),pd.to_numeric(c["USD/kg"],errors="coerce")))

def elemental_cost(formula,prices):
 comp=Composition(formula); total=float(comp.weight); return sum(float(amount)*float(el.atomic_mass)/total*prices[str(el)] for el,amount in comp.items())

def dopant_and_cost(df,bundle):
 chals=["O","Se","Te"]; hals=["F","Br","I"]
 xs=np.arange(GRID_X_STEP,4.501,GRID_X_STEP); ys=np.arange(GRID_Y_STEP,1.501,GRID_Y_STEP); rows=[]
 for ch in chals:
  for ha in hals:
   for x in xs:
    for y in ys: rows.append({"chalcogen":ch,"halogen":ha,"x":x,"y":y,"formula":formula_pair(ch,ha,float(x),float(y))})
 cand=pd.DataFrame(rows); feat,best_pred=predict_best(bundle,cand.formula.tolist()); cand["predicted_log10_mS_cm"]=best_pred; cand["predicted_mS_cm"]=10**best_pred; cand["prediction_model"]="ExtraTrees"
 prices=cost_map(); baseline=elemental_cost("Li5.5PS4.5Cl1.5",prices); cand["elemental_cost_USD_per_kg"]=cand.formula.map(lambda f:elemental_cost(f,prices)); cand["cost_change_vs_Li5.5PS4.5Cl1.5_USD_per_kg"]=cand.elemental_cost_USD_per_kg-baseline
 train=df[bundle["selected_features"]].to_numpy(float); mu=train.mean(0); sd=train.std(0); sd[sd==0]=1; trainz=(train-mu)/sd; fz=(feat.to_numpy(float)-mu)/sd
 train_nn=np.partition(cdist(trainz,trainz),1,axis=1)[:,1]; threshold=float(np.quantile(train_nn,.95)); cand["nearest_training_distance"]=cdist(fz,trainz).min(axis=1); cand["within_training_domain_95pct"]=cand.nearest_training_distance<=threshold
 valid=cand[cand.within_training_domain_95pct].sort_values("elemental_cost_USD_per_kg"); running=-np.inf; pareto=[]
 for idx,row in valid.iterrows():
  if row.predicted_log10_mS_cm>running: pareto.append(idx); running=row.predicted_log10_mS_cm
 cand["pareto_optimal_in_domain"]=False; cand.loc[pareto,"pareto_optimal_in_domain"]=True
 costn=(cand.elemental_cost_USD_per_kg-cand.elemental_cost_USD_per_kg.min())/(cand.elemental_cost_USD_per_kg.max()-cand.elemental_cost_USD_per_kg.min()); condn=(cand.predicted_log10_mS_cm-cand.predicted_log10_mS_cm.min())/(cand.predicted_log10_mS_cm.max()-cand.predicted_log10_mS_cm.min())
 cand["balanced_score"]=condn-costn; cand.to_csv(SOURCE/"dopant_cost_conductivity_screening.csv",index=False,encoding="utf-8-sig")
 low_idx=valid.elemental_cost_USD_per_kg.idxmin(); high_idx=valid.predicted_log10_mS_cm.idxmax(); balanced_idx=cand[cand.pareto_optimal_in_domain].balanced_score.idxmax()
 picks=pd.concat([cand.loc[[low_idx]].assign(selection="lowest_cost"),cand.loc[[high_idx]].assign(selection="highest_conductivity"),cand.loc[[balanced_idx]].assign(selection="balanced")])
 picks.to_csv(SOURCE/"recommended_cost_conductivity_formulas.csv",index=False,encoding="utf-8-sig")
 cand["pair"]=cand.chalcogen+"/"+cand.halogen
 experimental=pd.DataFrame([
  {"id":"A","formula":"Li5.5PS2.5Se2ClBr0.5"},
  {"id":"B","formula":"Li5.5PS2.5Se2Cl0.75Br0.75"},
  {"id":"C","formula":"Li5.5PS3.5SeClBr0.5"},
  {"id":"D","formula":"Li5.5PS3.5SeCl0.75Br0.75"},])
 _,exp_pred=predict_best(bundle,experimental.formula.tolist())
 experimental["predicted_log10_mS_cm"]=exp_pred; experimental["predicted_mS_cm"]=10**exp_pred; experimental["prediction_model"]="ExtraTrees"
 experimental["elemental_cost_USD_per_kg"]=experimental.formula.map(lambda f:elemental_cost(f,prices))
 experimental.to_csv(SOURCE/"cost_plot_four_experimental_formulas.csv",index=False,encoding="utf-8-sig")
 fig=plt.figure(figsize=(7.4,4.25)); ax=fig.add_axes([.09,.15,.62,.78]); info=fig.add_axes([.75,.12,.24,.80]); info.axis("off")
 pair_colors={}
 for i,(pair,g) in enumerate(cand.groupby("pair")):
  color=COLORS[i%len(COLORS)]; pair_colors[pair]=color
  ax.scatter(g.elemental_cost_USD_per_kg,g.predicted_log10_mS_cm,s=11,alpha=.32,label=pair,color=color,rasterized=True)
 pf=cand[cand.pareto_optimal_in_domain].sort_values("elemental_cost_USD_per_kg")
 ax.plot(pf.elemental_cost_USD_per_kg,pf.predicted_log10_mS_cm,color="black",lw=1.1,label="In-domain Pareto front")
 markers=["o","s","^","D"]
 for marker,(_,r) in zip(markers,experimental.iterrows()):
  ax.scatter(r.elemental_cost_USD_per_kg,r.predicted_log10_mS_cm,s=62,marker=marker,facecolor="white",edgecolor="black",linewidth=1.0,zorder=6)
 ax.set_xlabel("Elemental cost proxy (USD kg$^{-1}$ composition)"); ax.set_ylabel(r"Predicted log$_{10}$ conductivity (mS cm$^{-1}$)"); boxed(ax)
 inset=ax.inset_axes([.055,.59,.40,.36]); inset.set_facecolor("white"); inset.patch.set_alpha(1.0); inset.set_zorder(20); local=cand[(cand.pair=="Se/Br") & cand.elemental_cost_USD_per_kg.between(14.3,18.6) & cand.predicted_log10_mS_cm.between(1.12,1.26)]
 inset.scatter(local.elemental_cost_USD_per_kg,local.predicted_log10_mS_cm,s=11,alpha=.22,color=pair_colors["Se/Br"],rasterized=True)
 inset_offsets={"A":(5,5),"B":(5,5),"C":(5,-12),"D":(5,-13)}
 for marker,(_,r) in zip(markers,experimental.iterrows()):
  inset.scatter(r.elemental_cost_USD_per_kg,r.predicted_log10_mS_cm,s=44,marker=marker,facecolor="white",edgecolor="black",linewidth=.9,zorder=4)
  inset.annotate(r.id,(r.elemental_cost_USD_per_kg,r.predicted_log10_mS_cm),xytext=inset_offsets[r.id],textcoords="offset points",fontsize=6.5,weight="bold")
 inset.set_xlim(14.3,18.6); inset.set_ylim(1.12,1.26); inset.tick_params(labelsize=5.7,length=2); boxed(inset); ax.indicate_inset_zoom(inset,edgecolor="black",alpha=.75)
 handles=[mpl.lines.Line2D([],[],marker="o",ls="",ms=5,markerfacecolor=c,markeredgecolor=c,alpha=.65,label=pair) for pair,c in pair_colors.items()]
 handles.append(mpl.lines.Line2D([],[],color="black",lw=1.1,label="In-domain Pareto front")); info.legend(handles=handles,ncol=2,fontsize=5.9,frameon=False,loc="upper left",borderaxespad=0,handletextpad=.4,columnspacing=.8)
 info.text(0,.49,"Four experimental formulas",fontsize=7.3,weight="bold",va="top")
 yy=.44
 for _,r in experimental.iterrows():
  info.text(0,yy,f"{r.id}  {r.formula}",fontsize=6.2,weight="bold",va="top")
  info.text(0,yy-.045,f"{r.predicted_mS_cm:.1f} mS cm$^{{-1}}$; USD {r.elemental_cost_USD_per_kg:.1f} kg$^{{-1}}$",fontsize=5.8,va="top")
  yy-=.105
 fig.savefig(MAIN/"cost_conductivity_pareto_screening.png",dpi=600,bbox_inches="tight",facecolor="white"); fig.savefig(MAIN/"cost_conductivity_pareto_screening.pdf",bbox_inches="tight",facecolor="white"); plt.close(fig)
 fig,ax=plt.subplots(figsize=(4.5,3.25)); sns.violinplot(data=cand,x="chalcogen",y="predicted_log10_mS_cm",hue="halogen",order=chals,hue_order=hals,palette=[COLORS[0],COLORS[4],COLORS[1]],inner="quartile",cut=0,density_norm="width",common_norm=False,linewidth=.65,ax=ax); ax.set_xlabel("Chalcogen-site dopant"); ax.set_ylabel(r"ExtraTrees-predicted log$_{10}$ conductivity"); ax.legend(title="Halogen-site dopant",frameon=False,bbox_to_anchor=(1.01,1),loc="upper left"); boxed(ax); fig.tight_layout(); save(fig,MAIN,"g_final_dopant_pair_violin")
 return cand,picks,prices

def contour_h(bundle,prices):
 xs=np.arange(GRID_X_STEP,4.501,GRID_X_STEP); ys=np.arange(GRID_Y_STEP,1.501,GRID_Y_STEP)
 rows=[{"x":x,"y":y,"formula":formula_pair("Se","Br",float(x),float(y))} for y in ys for x in xs]
 grid=pd.DataFrame(rows); _,best_pred=predict_best(bundle,grid.formula.tolist())
 grid["predicted_log10_mS_cm"]=best_pred; grid["predicted_mS_cm"]=10**best_pred; grid["prediction_model"]="ExtraTrees"
 grid["elemental_cost_USD_per_kg"]=grid.formula.map(lambda f:elemental_cost(f,prices))
 grid.to_csv(SOURCE/"h_final_Se_Br_predictions_with_cost.csv",index=False)
 ex=pd.DataFrame([
  {"id":"A","formula":"Li5.5PS2.5Se2ClBr0.5","x":2.,"y":.5},
  {"id":"B","formula":"Li5.5PS2.5Se2Cl0.75Br0.75","x":2.,"y":.75},
  {"id":"C","formula":"Li5.5PS3.5SeClBr0.5","x":1.,"y":.5},
  {"id":"D","formula":"Li5.5PS3.5SeCl0.75Br0.75","x":1.,"y":.75},])
 _,ep=predict_best(bundle,ex.formula.tolist()); ex["predicted_log10_mS_cm"]=ep; ex["predicted_mS_cm"]=10**ep; ex["prediction_model"]="ExtraTrees"
 ex["elemental_cost_USD_per_kg"]=ex.formula.map(lambda f:elemental_cost(f,prices)); ex.to_csv(SOURCE/"h_final_experimental_formulas_with_cost.csv",index=False)
 z=grid.predicted_log10_mS_cm.to_numpy().reshape(len(ys),len(xs))
 # Saturate the fill at a physically readable high-conductivity threshold so
 # the broad lower-left high-value region is shown as one continuous plateau.
 # Numeric contours retain the differences above the color saturation limit.
 # Allocate most sub-threshold color resolution to the high-value range while
 # keeping a continuous transition into the 14.5 mS/cm deep-red plateau.
 levels=np.unique(np.r_[np.linspace(float(z.min()),.80,10,endpoint=False),np.linspace(.80,HIGH_PLATEAU_LOG10,21)])
 cmap=FOCUSED_CMAP; cmap.set_over("#a71930"); norm=BoundaryNorm(levels,cmap.N,clip=False)
 line_levels=np.array([v for v in [0.0,.4,.7,.9,1.0,1.10,HIGH_PLATEAU_LOG10] if z.min()<v<=HIGH_PLATEAU_LOG10])
 color_ticks=np.linspace(float(z.min()),HIGH_PLATEAU_LOG10,6)
 fig=plt.figure(figsize=(7.4,3.7)); gs=GridSpec(1,2,figure=fig,width_ratios=(4.7,2.15),wspace=.30); ax=fig.add_subplot(gs[0,0]); info=fig.add_subplot(gs[0,1]); info.axis("off")
 cf=ax.contourf(xs,ys,z,levels=levels,cmap=cmap,norm=norm,extend="both"); lines=ax.contour(xs,ys,z,levels=line_levels,colors="#5a4a00",linewidths=.4,alpha=.7); ax.clabel(lines,fontsize=4.8,fmt="%.2f")
 markers=["o","s","^","D"]
 for m,(_,r) in zip(markers,ex.iterrows()):
  ax.scatter(r.x,r.y,s=48,marker=m,facecolor="white",edgecolor="black",zorder=5)
  ax.annotate(r.id,(r.x,r.y),xytext=(5,5),textcoords="offset points",fontsize=7,weight="bold")
 cb=fig.colorbar(cf,ax=ax,pad=.02,ticks=color_ticks); cb.set_ticklabels([f"{v:.2f}" for v in color_ticks[:-1]]+[rf"$\geq${HIGH_PLATEAU_LOG10:.2f}"]); cb.set_label(r"ExtraTrees-predicted log$_{10}$ conductivity")
 ax.set(xlabel=r"Se content, $x$",ylabel=r"Br content, $y$",title=r"Li$_{5.5}$PS$_{4.5-x}$Se$_x$Cl$_{1.5-y}$Br$_y$"); ax.set_xlim(GRID_X_STEP,4.5); ax.set_ylim(.23,1.52); boxed(ax)
 info.set_title("Experimental candidates",loc="left",fontsize=8.5,pad=5)
 y0=.93
 for _,r in ex.iterrows():
  line1=f"{r.id}  {r.formula}"
  line2=f"{r.predicted_mS_cm:.1f} mS cm$^{{-1}}$;  USD {r.elemental_cost_USD_per_kg:.1f} kg$^{{-1}}$"
  info.text(0,y0,line1,ha="left",va="top",fontsize=6.8,weight="bold")
  info.text(0,y0-.075,line2,ha="left",va="top",fontsize=6.5)
  y0-=.21
 info.text(0,.02,"Cost is an elemental-price proxy;\nprocessing and precursor costs are excluded.",ha="left",va="bottom",fontsize=6,color="#555555")
 info.text(0,.105,rf"High-color plateau: $\sigma\geq${HIGH_PLATEAU_MS_CM:.1f} mS cm$^{{-1}}$",ha="left",va="bottom",fontsize=6.1,color="#8f1d24",weight="bold")
 fig.subplots_adjust(left=.09,right=.98,bottom=.15,top=.90); save(fig,MAIN,"h_final_Se_Br_contour")


def main():
 style(); df,bundle,features,y,oof=load(); figures_a_b(df); performance_bar(y,oof,bundle); best_model_scatter(y,oof,bundle); all_model_scatter(y,oof,bundle); residual_diagnostics(df,y); doi_group_sensitivity(df,y,bundle,oof); top10,importance=final_model_shap(df,bundle); correlations(df,features,top10); hyperparameter_heatmaps(); descriptor_table(features,importance)
 # RFECV curve is copied into the final SI namespace.
 curve=pd.read_csv(ROOT/"reports"/"rf_rfe_10models"/"rfecv_feature_count_results.csv"); best_count=int(curve.loc[curve.mean_test_score.idxmax(),"n_features"]); fig,ax=plt.subplots(figsize=(4.4,3)); ax.errorbar(curve.n_features,curve.mean_test_score,yerr=curve.std_test_score/np.sqrt(5),marker="o",lw=1,capsize=2,color=COLORS[0]); ax.axvline(best_count,color=COLORS[1],ls=":",lw=.9,label=f"RFECV optimum: {best_count}"); ax.set(xlabel="Number of RF-RFE descriptors",ylabel=r"Mean five-fold $R^2$"); ax.legend(frameon=False); boxed(ax); fig.tight_layout(); save(fig,SI,"rf_feature_count_selection")
 cand,picks,prices=dopant_and_cost(df,bundle); contour_h(bundle,prices); top5_xy_screening(bundle,prices); publish_figure1_aliases()
 manifest={"rows":len(df),"descriptors_before":188,"descriptors_after":len(features),"models_compared":bundle["model_order"],"final_model":"ExtraTrees","final_model_oof_r2":float(r2_score(y,oof["pred_extratrees_log10_mS_cm"])),"final_model_oof_mae_log10":float(mean_absolute_error(y,oof["pred_extratrees_log10_mS_cm"])),"grid_x_step":GRID_X_STEP,"grid_y_step":GRID_Y_STEP,"hyperparameter_search":"exhaustive discrete GridSearchCV","shap_model":"ExtraTrees","main_figures":sorted(p.name for p in MAIN.glob("*.png")),"si_figures":sorted(p.name for p in SI.glob("*.png")),"cost_definition":"sum of elemental mass fraction times supplied USD/kg; excludes processing","custom_descriptors_used":False,"stacking_used_as_final_model":False}
 (TABLE/"final_figure_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8"); print(json.dumps(manifest,ensure_ascii=False,indent=2))

if __name__=="__main__": main()


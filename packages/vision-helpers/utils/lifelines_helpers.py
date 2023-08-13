import lifelines
from lifelines import CoxPHFitter, KaplanMeierFitter
import pandas as pd

def run_cph_comparison(df, duration_col, event_col, baseline_covariates, additional_covariates, penalizer=0.1, l1_ratio=0.1):
    results = {}
    
    # baseline 
    cph = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
    cph.fit(df=df, 
            duration_col=duration_col, 
            event_col=event_col, 
            formula=' + '.join(baseline_covariates)
           )
    results['baseline_covariates'] = cph
    
    for var in additional_covariates:
        cph = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
        cph.fit(df=df.dropna(subset=[var]), 
                duration_col=duration_col, 
                event_col=event_col, 
                formula=' + '.join(baseline_covariates + [var])
               )
        results[var] = cph
    
    return results

def extract_cph_results(results):
    agg = pd.DataFrame(data=[x.log_likelihood_ for x in results.values()], index=results.keys(), columns=['log_likelihood_'])
    agg['AIC_partial_'] = [x.AIC_partial_ for x in results.values()]
    agg['concordance_index_'] = [x.concordance_index_ for x in results.values()]

    for key, val in results.items():
        temp = [x for x in val.summary.index.values if key in x]
        if len(temp) > 1:
            print('detected categorical: ', key)
        agg.loc[key, 'p'] = val.summary.loc[temp, 'p'].mean()
        agg.loc[key, 'hazard_ratio'] = val.summary.loc[temp, 'exp(coef)'].mean()

    return agg.sort_values('concordance_index_', ascending=False)

def run_cph_comparison_multivar(df, duration_col, event_col, covariate_formula, penalizer=0.1, l1_ratio=0.1):
    results = {}
    
    # baseline 
    cph = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
    cph.fit(df=df, 
            duration_col=duration_col, 
            event_col=event_col, 
            formula=covariate_formula
           )
    return cph
# -*- coding: utf-8 -*-
"""

@author: ssundukov
"""


import pandas as pd
import os
import numpy as np
from pulp import *
import datetime
import ConfigParser

applications = pd.read_csv('applications.csv', sep=',', header =0)
app_components = pd.read_csv('app_components.csv', sep=',', header =0)
component_cost = pd.read_csv('component_cost.csv', sep=',', header =0)
waves = pd.read_csv('waves.csv', sep=',', header =0)
fixed_apps = pd.read_csv('fixed_apps.csv', sep=',', header =0)

fixed_apps =  pd.merge(fixed_apps, waves, on=['wave'], how='inner')


conf = ConfigParser.RawConfigParser()
conf.read(os.path.dirname(os.path.abspath(__file__)) + '\\app_scheduling.conf')
max_app_per_wave  = conf.get("main", "max_app_per_wave")
early_end_for_penalty = conf.get("main", "early_end_for_penalty")
late_end_for_penalty =  conf.get("main", "late_end_for_penalty")
penalty_rate = conf.get("main", "penalty_rate")
fence_date = pd.to_datetime(conf.get("main", "fence_date"), dayfirst=True)

var_i = pd.DataFrame(columns=applications.columns.values.tolist() + waves.columns.values.tolist())

ind = 0

for a in applications.index.values:
    for w in waves.index.values:        
        var_i.loc[ind] = pd.concat([applications.ix[a], waves.ix[w]])
        ind = ind + 1


var_i['cost'] = 0

for v in var_i.index.values:
    if np.timedelta64(pd.to_datetime(var_i['end_date'][v], dayfirst=True)- pd.to_datetime(var_i['preferred_date'][v], dayfirst=True), 'D').astype(int) > int(late_end_for_penalty):
        var_i.loc[v, 'cost'] = var_i.loc[v, 'cost'] + int(penalty_rate)
            
    if np.timedelta64(pd.to_datetime(var_i['preferred_date'][v], dayfirst=True) - pd.to_datetime(var_i['end_date'][v], dayfirst=True), 'D').astype(int) > int(early_end_for_penalty):
        var_i.loc[v, 'cost'] = var_i.loc[v, 'cost'] + int(penalty_rate)
    
    for c in app_components[app_components['application'] == var_i['application'][v]].index.values:
        for cs in component_cost[component_cost['component'] == app_components['component'][c]].index.values:
            
            if (pd.to_datetime(component_cost['start_date'][cs], dayfirst=True) < pd.to_datetime(var_i['end_date'][v], dayfirst=True) and pd.to_datetime(component_cost['end_date'][cs], dayfirst=True) >= pd.to_datetime(var_i['start_date'][v], dayfirst=True)):
                    cost_st_date = max(pd.to_datetime(component_cost['start_date'][cs]), pd.to_datetime(var_i['start_date'][v], dayfirst=True))
                    cost_en_date = min(pd.to_datetime(var_i['end_date'][v], dayfirst=True), pd.to_datetime(component_cost['end_date'][cs], dayfirst=True))
                    var_i.loc[v, 'cost'] = var_i.loc[v, 'cost'] + (np.timedelta64(cost_en_date - cost_st_date, 'D').astype(int)/30.41666667)*component_cost['cost_per_month'][cs]


x = pulp.LpVariable.dicts('Schedule', (var_i.index), 
                                        lowBound = 0,
                                        upBound = 1,
                                        cat = pulp.LpInteger)

scheduling_model = pulp.LpProblem("App Scheduling", pulp.LpMinimize)

scheduling_model += sum(x[v]*var_i['cost'][v] for v in var_i.index)




for a in applications.index:
    scheduling_model += sum(x[v] for v in var_i[var_i['application']==applications['application'][a]].index) == 1, ""
     
for w in waves.index:
    scheduling_model += lpSum(x[v] for v in var_i[var_i['wave']==waves['wave'][w]].index) <= int(max_app_per_wave), ""

for f in fixed_apps[fixed_apps['fixed'] == 1].index:
    scheduling_model += lpSum(x[v] for v in var_i[(var_i['wave']==fixed_apps['wave'][f]) & (var_i['application']==fixed_apps['application'][f])].index) == 1, ""

for f in fixed_apps[pd.to_datetime(fixed_apps['start_date'], dayfirst=True) <= fence_date].index:
    scheduling_model += lpSum(x[v] for v in var_i[(var_i['wave']==fixed_apps['wave'][f]) & (var_i['application']==fixed_apps['application'][f])].index) == 1, ""



scheduling_model.solve(COIN_CMD(msg = 1), use_mps = False)

applications['wave'] = 'www'

for a in applications.index:
    for v in var_i[var_i['application'] == applications['application'][a]].index:
        if x[v].value() == 1:
            applications.loc[a, 'wave'] = var_i['wave'][v]
                
applications.to_csv(os.path.dirname(os.path.abspath(__file__)) + "\\result.csv", index=False)

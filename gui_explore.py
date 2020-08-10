import attr
import numpy as np
import pandas as pd
import streamlit as st

import gui_config as conf
from gui_utils import load_previous_configs, make_st_widget, ParamDescriptor, ParamType, VisType

'''
# All runs
'''

configs = load_previous_configs()

if len(configs) == 0:
    st.warning("No data found.")
else:
    def get_config_as_dict(c):
        d = attr.asdict(c)
        rr = c.load_results()
        errors = np.array([r.error for r in rr.results])
        d['file_path'] = f"/{d['file_path'].stem}/"
        d['min_error'] = errors.min()
        d['final_error'] = errors[-1]
        return d


    configs_as_dict = [get_config_as_dict(c) for c in configs if c.is_favorite]
    df = pd.DataFrame(configs_as_dict)


    def cast_numpy_type_to_python(arg0, *args):
        if isinstance(arg0, int):
            return map(int, (arg0, *args))
        elif isinstance(arg0, float):
            return map(float, (arg0, *args))
        raise Exception


    feature_map_paths = conf.FEATURE_MAP_DIR.glob("*.png")
    feature_map_names = [p.name for p in feature_map_paths]
    st.sidebar.multiselect(label="feature_map_path",
                           options=feature_map_names,
                           default=feature_map_names)

    for param, descriptor in conf.PARAM_DESCRIPTOR_MAP.items():
        filter_descriptor = None
        if descriptor.param_type == ParamType.INTERVAL:
            min_val = df[param].min()
            max_val = df[param].max()
            if min_val == max_val or pd.isna(min_val) or pd.isna(max_val):
                min_val = descriptor.min_value
                max_val = descriptor.max_value
                step = descriptor.step
            else:
                step, min_val, max_val = cast_numpy_type_to_python(descriptor.step, min_val, max_val)
            filter_descriptor = ParamDescriptor(param_type=ParamType.INTERVAL,
                                                min_value=min_val,
                                                max_value=max_val,
                                                value=(min_val, max_val),
                                                step=step,
                                                vis_type=VisType.SLIDER)
        elif descriptor.param_type == ParamType.CATEGORICAL:
            options = list(set(df[param]))
            filter_descriptor = ParamDescriptor(param_type=ParamType.CATEGORICAL,
                                                options=options,
                                                vis_type=VisType.MULTISELECT,
                                                default=options)
        if filter_descriptor:
            make_st_widget(filter_descriptor, label=param, returns_iterable=True)

    st.dataframe(df)

import sys
import os
import time
import numpy as np

def _strip_quotes(s: str) -> str:
    s = s.strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    return s

def parse_simple_yaml(cfg_path: str):
    """
    Parse only the keys used in your generated yaml:
      src_pc: "./data/xxx_src.ply"
      tgt_pc: "./data/xxx_tgt.ply"
      visualize_nricp: True/False
      nricp_result_path: "./log/xxx_nricp.ply"
      cost_time_path: "./log/xxx_nricp_time.txt"
    """
    kv = {}
    with open(cfg_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if ':' not in line:
                continue
            k, v = line.split(':', 1)
            kv[k.strip()] = _strip_quotes(v.strip())
    for k in ['src_pc', 'tgt_pc', 'nricp_result_path', 'cost_time_path']:
        if k not in kv:
            raise RuntimeError('missing key {} in {}'.format(k, cfg_path))
    # visualize_nricp default to True if not specified
    viz = kv.get('visualize_nricp', 'True').strip()
    kv['visualize_nricp'] = (viz.lower() == 'true')
    return kv

def swap_ext(path: str, new_ext: str) -> str:
    base, _ = os.path.splitext(path)
    return base + new_ext

# --- main ---
name_of_objects = []
user_input = input('input object name:')
while user_input != '-1':
    name_of_objects.append(user_input)
    user_input = input('input object name:')

rotate_ids = []
rotate_ids.append(0)

first_view_ids = []
first_view_ids.append(0)

for object_name in name_of_objects:
    print('testing ' + object_name)
    for rotate_id in rotate_ids:
        for view_id in first_view_ids:
            cfg_path = './data/' + object_name + '_r' + str(rotate_id) + '_v' + str(view_id) + '_config.yaml'
            while os.path.isfile(cfg_path) == False:
                pass
            time.sleep(1)

            # read yaml parse config
            cfg = parse_simple_yaml(cfg_path)
            # visualize_nricp=False -> --no-vis
            cmd = '~/cilantro_nicp/src/build/apps/cilantro_nicp ' \
                  + cfg['src_pc'] + ' ' + cfg['tgt_pc'] + ' ' \
                  + '--out ' + cfg['nricp_result_path'] + ' ' \
                  + '--time ' + cfg['cost_time_path'] + ' '
            if cfg['visualize_nricp'] == False:
                cmd += '--no-vis '
            # call nricp c++ app
            os.system(cmd)

            f = open('./log/' + object_name + '_r' + str(rotate_id) + '_v' + str(view_id) + '_ready.txt', 'a')
            f.close()
            print('testing ' + object_name + '_r' + str(rotate_id) + '_v' + str(view_id) + ' over.')
print('all over.')

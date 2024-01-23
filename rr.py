# screen
from utilities.utils import screen_lib
#screen_lib("analysis/MRAS/desc/mras_desc.npy", "analysis/KRAS/desc/binder_desc.npy", "./pep_library_one/pep_feat_7to15_names.pkl", "analysis/MRAS/mras_screen.txt", topN=1000000, target_site=None)
screen_lib("analysis/MRAS/desc/shoc2_desc.npy", "analysis/KRAS/desc/binder_desc.npy", "./pep_library_one/pep_feat_7to15_names.pkl", "analysis/MRAS/shoc2_screen.txt", topN=1000000, target_site=None)
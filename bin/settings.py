import os

survey_dir = os.getenv('LEGACY_SURVEY_DIR')
randoms_input_fn = '/global/cfs/cdirs/desi/target/catalogs/dr9m/0.42.0/randoms/resolve/randoms-randomized-1.fits'
truth_fn = '/project/projectdirs/desi/users/ajross/MCdata/seed.fits'
output_dir = os.getenv('OUTPUT_DIR')
randoms_fn = os.getenv('RANDOMS_FN')
bricklist_fn = 'bricklist.txt'
bricknames = ['0954p487'] #'0954p487','1500m242','1986p432'
randoms_matched_fn = os.getenv('RANDOMS_FN').replace('.fits','_matched.fits')
dir_plot = './plots/'

import os

survey_dir = os.getenv('LEGACY_SURVEY_DIR')
output_dir = os.path.join(os.getenv('CSCRATCH'),'Obiwan','dr9','test')
randoms_input_fn = '/global/cfs/cdirs/desi/target/catalogs/dr9m/0.42.0/randoms/resolve/randoms-randomized-1.fits'
truth_fn = '/project/projectdirs/desi/users/ajross/MCdata/seed.fits'
randoms_fn = os.path.join(output_dir,'randoms','randoms.fits')
bricklist_fn = 'bricklist.txt'
def get_bricknames():
    #return [brickname[:-len('\n')] for brickname in open(bricklist_fn,'r')]
    return ['0954p487','1986p432'][:1] #'0954p487','1500m242','1986p432'
run = 'north'
fileid = 0
rowstart = 0
skipid = 0
randoms_matched_fn = randoms_fn.replace('.fits','_matched.fits')
dir_plot = './plots/'

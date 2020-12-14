import os

survey_dir = os.getenv('LEGACY_SURVEY_DIR')
output_dir = os.path.join(os.getenv('CSCRATCH'),'Obiwan','dr9','test')
randoms_input_fn = '/global/cfs/cdirs/desi/target/catalogs/dr9m/0.44.0/randoms/resolve/randoms-1-0.fits'
truth_fn = '/project/projectdirs/desi/users/ajross/MCdata/seed.fits'
randoms_fn = os.path.join(output_dir,'randoms','randoms.fits')
bricklist_fn = 'bricklist.txt'
def get_bricknames():
    return [brickname.replace('\n','') for brickname in open(bricklist_fn,'r')]
run = 'north'
fileid = 0
rowstart = 0
skipid = 0
#legacypipe_survey_dir = os.getenv('LEGACYPIPE_SURVEY_DIR')
#legacypipe_output_dir = os.path.join(os.getenv('LEGACYPIPE_SURVEY_DIR'),run)

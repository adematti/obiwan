import os

survey_dir = os.getenv('LEGACY_SURVEY_DIR')
output_dir = os.path.join(os.getenv('CSCRATCH'),'Obiwan','dr9','ebv100')
randoms_input_fn = '/global/cfs/cdirs/desi/target/catalogs/dr9m/0.44.0/randoms/resolve/randoms-1-0.fits'
truth_fn = '/project/projectdirs/desi/users/ajross/MCdata/seed.fits'
randoms_fn = os.path.join(output_dir,'randoms','randoms.fits')
bricklist_fn = 'bricklist_ebv.txt'
def get_bricknames():
    tmp = [brickname[:-len('\n')] for brickname in open(bricklist_fn,'r')]
    return tmp
    #excl = ['1320p385', '1034p332', '1513p757', '1347p387', '0951p562', '1119p530', '0988p520', '1013p632', '0963p565', '1004p442']
    #excl += ['2884p560', '0988p520', '0963p435', '0994p432', '1915p795', '1004p442', '1011p407', '1011p627', '0964p440', '1011p407', '2915p527', '0978p485', '1013p365']
    #return [t for t in tmp if t not in excl]
run = 'north'
fileid = 0
rowstart = 0
skipid = 0
kwargs_file = {'fileid':fileid,'rowstart':rowstart,'skipid':skipid}
merged_dir = os.path.join(output_dir,'merged')
dir_plot = './plots/'
legacypipe_survey_dir = os.getenv('LEGACYPIPE_SURVEY_DIR')
legacypipe_output_dir = os.path.join(os.getenv('LEGACYPIPE_SURVEY_DIR'),run)

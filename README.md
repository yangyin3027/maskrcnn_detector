# maskrcnn_detector

# get stated
git clone https://github.com/yangyin3027/maskrcnn_detector.git

# create a python virtual environment
# use conda
conda create -n [virtual environment name]
# activate the environment
conda activate [virtual environment name]

# install all the prerequiste packages
pip install -r requirements.txt

# run command line
# default is detect all objects with confidence over 0.8
python human_detector.py --img [img file] 

# could specify human_only detection by add an argument
python human_detector.py --img [img_file] --human True

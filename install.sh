cd cuda
tr '\n' '\0' < files.txt | xargs -0 sudo rm -f --
rm -rf build/ dist/ *.egg* 
python setup.py install --record files.txt

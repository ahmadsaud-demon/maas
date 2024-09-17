# maas

if you will face nltk download error, run below code on your python console

import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download()


if you'll face pdfminer issue, run below

pip3 uninstall pdfminer
pip3 uninstall pdfminer-six
pip3 install pdfminer-six 



we will need "brew install tesseract"

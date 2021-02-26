import argparse

from app import ppfe
from app import dtt

__STEP__ = ['ppfe', 'train', 'test']
__DEST__ = 'command'

parser = argparse.ArgumentParser()
subparser = parser.add_subparsers(dest=__DEST__)

ppfe_parser = subparser.add_parser(__STEP__[0])
ppfe_parser.add_argument('-cfg', type=str)

train_parser = subparser.add_parser(__STEP__[1])
h5_train_help = "h5 file name to train, located in the root"
train_parser.add_argument('-h5', type=str, help=h5_train_help)
clsf_help = "choose one of these classifications"
train_parser.add_argument('-clsf', type=str, choices=dtt.CLSF, required=True, help=clsf_help)
attr_help = "format attr should follow the sequence and without parenthesis:\nsvm -> C=(float) kernel=({linear, poly, rbf, sigmoid, precomputed}) tol=(float), knn -> n=(int). coef0=(float) leave it if you doesnt use poly or sigmoid"
train_parser.add_argument('-attr', type=str, nargs='*', required=True, help=attr_help)

test_parser = subparser.add_parser(__STEP__[2])
h5_test_help = "h5 file name to test, located in the root"
test_parser.add_argument('-h5', type=str, help=h5_test_help)
joblib_test_help = "joblib model file name, located in the root"
test_parser.add_argument('-md', type=str, help=joblib_test_help)

args = vars(parser.parse_args())

if (args[__DEST__] == __STEP__[0]):
    path_sections = args['cfg'].split('/')
    file_name = path_sections[len(path_sections)-1]
    ppfe.apply(file_name)
elif (args[__DEST__] == __STEP__[1]):
    h5_file_name = args['h5']
    clsf = args['clsf']
    attrs = args['attr']
    dtt.train(h5_file_name, classification=clsf, attributes=attrs)
elif (args[__DEST__] == __STEP__[2]):
    h5_file_name = args['h5']
    joblib_model_file_name = args['md']
    dtt.test(h5_file_name, joblib_model_file_name)
else:
    print('unsupported step')


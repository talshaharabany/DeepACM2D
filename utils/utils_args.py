import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-it', '--DeepIt', default=3, help='number of snake iteration', required=False)
    parser.add_argument('-lr', '--learning_rate', default=0.001, help='learning_rate', required=False)
    parser.add_argument('-bs', '--Batch_size', default=20, help='batch_size', required=False)
    parser.add_argument('-ep', '--epochs', default=25000, help='number of epoches', required=False)
    parser.add_argument('-drop', '--drop', default=0, help='dropout value', required=False)
    parser.add_argument('-R', '--Radius', default=64, help='initial guess circle radius', required=False)
    parser.add_argument('-D_rate', '--D_rate', default=50, help='initial guess circle radius', required=False)
    parser.add_argument('-opt', '--opt', default='adam', help='initial guess circle radius', required=False)
    parser.add_argument('-a', '--a', default=0.5, help='initial guess circle radius', required=False)
    parser.add_argument('-nW', '--nW', default=0, help='number os workers', required=False)
    parser.add_argument('-WD', '--WD', default=0.00005, help='number os workers', required=False)
    parser.add_argument('-nP', '--nP', default=32, help='number of points', required=False)
    parser.add_argument('-order', '--order', default=85, help='backbone dimension', required=False)
    parser.add_argument('-depth_wise', '--depth_wise', default=0, help='backbone dimension', required=False)
    parser.add_argument('-outlayer', '--outlayer', default=3, help='backbone dimension', required=False)
    parser.add_argument('-im_size', '--im_size', default=256, help='backbone dimension', required=False)
    parser.add_argument('-task', '--task', default='bing', help='which dataset to use?', required=False)
    args = vars(parser.parse_args())
    return args


def save_args(args):
    path = r'results/' + args['task'] + '/params.csv'
    f = open(path, 'w')
    keys = list(args.keys())
    vals = list(args.values())
    for i in range(len(keys)):
        f.write(str(keys[i])+','+str(vals[i])+'\n')
        f.flush()

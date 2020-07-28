import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-it', '--DeepIt', default=3, help='number of snake iteration', required=False)
    parser.add_argument('-lr', '--learning_rate', default=0.001, help='learning_rate', required=False)
    parser.add_argument('-bs', '--Batch_size', default=50, help='batch_size', required=False)
    parser.add_argument('-ep', '--epochs', default=25000, help='number of epoches', required=False)
    parser.add_argument('-drop', '--drop', default=0.1, help='dropout value', required=False)
    parser.add_argument('-R', '--Radius', default=8, help='initial guess circle radius', required=False)
    parser.add_argument('-nW', '--nW', default=0, help='number os workers', required=False)
    parser.add_argument('-dim', '--dim', default=64, help='input dimension', required=False)
    parser.add_argument('-wM', '--wM', default=1, help='mask loss coeff', required=False)
    parser.add_argument('-wB', '--wB', default=0.01, help='Ballon loss coeff', required=False)
    parser.add_argument('-wNN', '--wNN', default=0.1, help='NN loss coeff', required=False)
    parser.add_argument('-folder', '--folder', default=0, help='name of save folder', required=False)
    parser.add_argument('-nP', '--nP', default=16, help='number of points', required=False)
    parser.add_argument('-AEdim', '--AEdim', default=512, help='backbone dimension', required=False)
    parser.add_argument('-is_res', '--is_res', default=0, help='is resnet backbone?', required=False)
    parser.add_argument('-is_load', '--is_load', default=0, help='is load check point?', required=False)
    parser.add_argument('-CP', '--CP', default=1, help='check point epoch', required=False)
    parser.add_argument('-CHP', '--CHP', default=2, help='evaluation iteration', required=False)
    parser.add_argument('-task', '--task', default='viah', help='which dataset to use?', required=False)
    args = vars(parser.parse_args())
    args = vars(parser.parse_args())
    return args


def save_args(args):
    path = 'results/gpu' + str(args['folder'])+'/params.csv'
    f = open(path, 'w')
    keys = list(args.keys())
    vals = list(args.values())
    for i in range(len(keys)):
        f.write(str(keys[i])+','+str(vals[i])+'\n')
        f.flush()

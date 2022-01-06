import sys
sys.path.append("..")
from RecQ import RecQ
from tool.config import Config
from visual.display import Display



if __name__ == '__main__':

    print '='*80
    print '   RecQ: An effective python-based recommender algorithm library.   '
    print '='*80
    print '0. Analyze the input data.(Configure the visual.conf in config/visual first.)'
    print '-' * 80
    print 'Generic Recommenders:'
    print '1. BG_Rec'

    print '='*80
    algor = -1
    conf = -1
    order = raw_input('please enter the num of the algorithm to run it:')
    #order = sys.argv[1]
    import time
    s = time.time()
    if order == '0':
        try:
            import seaborn as sns
        except ImportError:
            print '!!!To obtain nice data charts, ' \
                  'we strongly recommend you to install the third-party package <seaborn>!!!'
        conf = Config('../config/visual/visual.conf')
        Display(conf).render()
        exit(0)

    algorthms = {'1':'BG_Rec'}

    try:
        conf = Config('../config/'+algorthms[order]+'.conf')
    except KeyError:
        print 'Error num!'
        exit(-1)
    recSys = RecQ(conf)
    recSys.execute()
    e = time.time()
    print "Run time: %f s" % (e - s)